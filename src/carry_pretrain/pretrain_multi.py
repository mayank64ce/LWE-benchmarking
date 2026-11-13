"""Step 3a: multi-secret pretraining with secret-ID tokens (Fresca-style).

Fresca (ePrint 2024/150, Section 5.2) pretrains one model on many invented
secrets at once by giving each secret a learned token t_si in the input
sequence, so the model knows which secret a sample belongs to. We copy that
protocol and add our wrap supervision on top:

  arm A: plain b labels          -> this IS Fresca's protocol (the baseline)
  arm B: b + carry head (k)      -> ours
  arm C: aux-modulus label switch -> Kikuchi, see AUXMOD_RECIPE.md

Data fairness: Fresca pretrains on a separately generated reduced dataset.
At toy scale we only have one reduced pool, so we split by ROWS instead:
pretraining uses rows [0, 25000) of train_A; the victim fine-tune
(finetune.py) uses only rows [25000, end). Disjoint rows, same reduction
run — that deviation from Fresca is deliberate and documented here.

The checkpoint this script saves is consumed by finetune.py.

Run (one arm, ~40 min on the 3090):
  python3 pretrain_multi.py --arm C --npz ../../../carry_pretrain_data/synth_h8_m64_K4_seed0.npz \
      --grid_dir ../../../testn80_grid --out_dir ../../../carry_pretrain_runs
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src.salsa.train.model.encoder import Encoder  # noqa: E402

from make_synthetic_labels import transform_A  # noqa: E402
from pretrain import (CarryEncoder, DualAngularEncoder,  # noqa: E402
                      to_circle, decode_circle)

PRETRAIN_ROWS = 25000  # rows [0, 25000) for pretraining; the rest is victim's


class SecretTokenModel(nn.Module):
    """Wraps one of the arm models and prepends a learned secret-ID token
    to the input sequence (Fresca's t_si). The embedding has one extra slot
    (index num_secrets) reserved for the fine-tune victim's fresh token."""

    def __init__(self, base, num_secrets, emb_dim):
        super().__init__()
        self.base = base
        self.sec_emb = nn.Embedding(num_secrets + 1, emb_dim)

    def forward(self, A, sec_idx):
        b = self.base
        tok = b.tok_emb(A)                              # (batch, 80, dim)
        sec = self.sec_emb(sec_idx).unsqueeze(1)        # (batch, 1, dim)
        x = torch.cat([sec, tok], dim=1)                # secret token first
        pos = torch.arange(0, x.shape[1], device=A.device).unsqueeze(0)
        x = b.drop(x + b.pos_emb(pos))
        for layer in b.layers:
            x = layer(x)
        x = b.ln_f(x)
        pooled = torch.max(x, dim=1)[0]
        out = b.head(pooled)
        if hasattr(b, "k_head"):
            return out, b.k_head(pooled)
        return out


def build_base(arm, emb_dim, layers, heads, q, n_k_classes):
    """One place that maps an arm name to its architecture, shared with
    finetune.py so checkpoints always load into the right shapes."""
    class P:
        pass
    mp = P()
    mp.enc_emb_dim = emb_dim
    mp.n_enc_layers = layers
    mp.n_enc_heads = heads
    mp.dropout = 0.0
    mp.attention_dropout = 0.0
    mp.Q = q
    if arm == "A":
        return Encoder(mp)
    if arm == "B":
        return CarryEncoder(mp, n_k_classes)
    return DualAngularEncoder(mp)  # arm C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["A", "B", "C"], required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--grid_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--num_secrets", type=int, default=32,
                   help="how many invented secrets to pretrain across")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux_r", type=float, default=0.3)
    p.add_argument("--carry_weight", type=float, default=1.0)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=2000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.npz)
    K = int(z["K"])
    A_shift = int(z["A_shift"])
    q = args.Q
    b_t = torch.from_numpy(z["b"])
    k_t = torch.from_numpy(z["k"])
    bK_t = torch.from_numpy(z["bK"])
    n_k_classes = int(z["k"].max()) + 1

    A = np.load(os.path.join(args.grid_dir, "train_A.npy"))
    X = torch.from_numpy(transform_A(A, A.shape[1], q, A_shift))

    base = build_base(args.arm, args.emb_dim, args.layers, args.heads,
                      q, n_k_classes)
    model = SecretTokenModel(base, args.num_secrets, args.emb_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def encode_inputs(rows):
        if args.arm == "C":
            return torch.cat([to_circle(rows, q), to_circle(rows, K * q)], dim=-1)
        return to_circle(rows, q)

    # In-pretraining eval: held-out rows of pretraining secret 0 (sanity only;
    # the real evaluation is the fine-tune in finetune.py).
    eval_rows = rng.choice(np.arange(20000, PRETRAIN_ROWS), 2000, replace=False)
    eval_X = encode_inputs(X[eval_rows]).to(dev)
    eval_sec = torch.zeros(len(eval_rows), dtype=torch.long, device=dev)
    eval_true = b_t[0, eval_rows]

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = model(eval_X, eval_sec)
        if args.arm == "B":
            out = out[0]
        pred = decode_circle(out[:, :2], q).cpu()
        d = (pred - eval_true) % q
        d = torch.minimum(d, q - d)
        model.train()
        return (pred == eval_true).float().mean().item(), (d <= 6).float().mean().item()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"pretrain_arm{args.arm}_m{args.num_secrets}_seed{args.seed}"
    log = open(os.path.join(args.out_dir, f"{tag}.jsonl"), "a")
    t0 = time.time()

    # Training rows [0, 20000): disjoint from the sanity-eval rows above
    # and from the victim rows used by finetune.py.
    train_rows_pool = np.arange(0, 20000)

    for step in range(1, args.steps + 1):
        j = int(rng.integers(0, args.num_secrets))
        rows = rng.choice(train_rows_pool, size=args.batch_size)
        xb = encode_inputs(X[rows]).to(dev)
        sec = torch.full((args.batch_size,), j, dtype=torch.long, device=dev)

        if args.arm == "A":
            loss = mse(model(xb, sec), to_circle(b_t[j, rows], q).to(dev))
        elif args.arm == "B":
            out_b, out_k = model(xb, sec)
            loss = (mse(out_b, to_circle(b_t[j, rows], q).to(dev))
                    + args.carry_weight * ce(out_k, k_t[j, rows].to(dev)))
        else:  # arm C, Eq. (2) label switch
            out = model(xb, sec)
            sel = torch.from_numpy(rng.random(args.batch_size) < args.aux_r).to(dev)
            t_q = to_circle(b_t[j, rows], q).to(dev)
            t_K = to_circle(bK_t[j, rows], K * q).to(dev)
            loss = (mse(out[~sel, :2], t_q[~sel]) * (~sel).float().mean()
                    + mse(out[sel, 2:], t_K[sel]) * sel.float().mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0:
            acc, near = evaluate()
            rec = {"step": step, "loss": float(loss.item()),
                   "sec0_match_acc": acc, "sec0_within_2sigma": near,
                   "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

    torch.save({"model": model.state_dict(), "args": vars(args),
                "n_k_classes": n_k_classes},
               os.path.join(args.out_dir, f"{tag}.pt"))
    print(f"saved {tag}.pt")


if __name__ == "__main__":
    main()
