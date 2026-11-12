"""Step 2 of the carry-pretraining experiment: the three-arm pretraining run.

Question this script answers: on REAL BKZ-reduced rows (the victim's train_A),
does supervising wrap information make the LWE prediction task learnable?

  arm A (baseline):  predict b = (X @ s + e) mod q.  Plain SALSA-style target.
  arm B (carry):     predict b AND the wrap count k with a second head (ours).
  arm C (aux mod):   per-sample label switch: with probability r the target is
                     b mod Kq instead of b mod q (Kikuchi et al., Eq. 2 —
                     see AUXMOD_RECIPE.md; dual angular readout).

All arms share the same model trunk (the repo's angular Encoder), the same
optimizer, the same batch schedule, the same seeds. The ONLY difference is
the label supervision. So any gap between arms is caused by the labels.

Each run trains on ONE invented secret (b depends on the secret, so the task
is only well-posed per secret — same as FRESCA). Evaluation: match accuracy
(predicted b rounded to nearest integer mod q equals true b) on HELD-OUT ROWS
of that secret. Chance level is 1/q ~ 0.9% exact, ~11.5% within 2 sigma. We
report both, since exact match is capped by the LWE noise itself.

We deliberately do NOT touch any repo file: the arm-specific parts live in
small subclasses here. Repo pieces reused: Encoder, ProjectedEmbedding,
CoordinatesHead (src/salsa/train/model/encoder.py).

Run (one arm, ~minutes on the 3090):
  python3 pretrain.py --arm A --npz ../../../carry_pretrain_data/synth_h8_m64_K4_seed0.npz \
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

# Repo imports: we run from src/carry_pretrain, repo root is two levels up.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from src.salsa.train.model.encoder import Encoder  # noqa: E402

from make_synthetic_labels import transform_A  # noqa: E402


# ---------------------------------------------------------------------------
# Models. Arm A uses the repo Encoder unchanged. B and C are tiny subclasses.
# ---------------------------------------------------------------------------

class CarryEncoder(Encoder):
    """Arm B: the repo encoder plus a second linear head that classifies the
    wrap count k from the same pooled representation. Everything else equal."""

    def __init__(self, params, n_k_classes):
        super().__init__(params)
        self.k_head = nn.Linear(params.enc_emb_dim, n_k_classes, bias=False)

    def forward(self, A):
        # Same body as BaseEncoder.forward, but we also return the k logits.
        tok_emb = self.tok_emb(A)
        pos = torch.arange(0, A.shape[1], device=A.device).unsqueeze(0)
        x = self.drop(tok_emb + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        pooled = torch.max(x, dim=1)[0]
        return self.head(pooled), self.k_head(pooled)


class DualAngularEncoder(Encoder):
    """Arm C: the repo encoder with 4-dim input projection and 4-dim readout,
    one circle for mod q and one for mod Kq (paper's Approach 2)."""

    def get_input_embedding(self, params):
        emb = super().get_input_embedding(params)
        emb.proj = nn.Linear(4, params.enc_emb_dim)  # two circles in
        return emb

    def get_head(self, params):
        head = super().get_head(params)
        head.proj = nn.Linear(params.enc_emb_dim, 4, bias=False)  # two circles out
        return head


class HybridEncoder(DualAngularEncoder):
    """Arm D: hybrid of C and B — dual circles AND the carry head. The two
    tricks act in different places (targets vs an extra output), so they
    compose without interfering."""

    def __init__(self, params, n_k_classes):
        super().__init__(params)
        self.k_head = nn.Linear(params.enc_emb_dim, n_k_classes, bias=False)

    def forward(self, A):
        tok_emb = self.tok_emb(A)
        pos = torch.arange(0, A.shape[1], device=A.device).unsqueeze(0)
        x = self.drop(tok_emb + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        pooled = torch.max(x, dim=1)[0]
        return self.head(pooled), self.k_head(pooled)


# ---------------------------------------------------------------------------
# Angular encoding helpers (same convention as the repo's AngularEncoder).
# ---------------------------------------------------------------------------

def to_circle(x, modulus):
    """Map integers to (cos, sin) on the circle of the given modulus."""
    rad = x.to(torch.float32) / modulus * 2 * torch.pi
    return torch.stack((torch.cos(rad), torch.sin(rad)), dim=-1)


def decode_circle(xy, modulus):
    """Inverse of to_circle: (cos, sin) -> nearest integer mod modulus.
    Same decode the paper and the repo use (angle -> scale -> round)."""
    ang = (torch.atan2(xy[..., 1], xy[..., 0]) + 2 * torch.pi) % (2 * torch.pi)
    return torch.round(ang / (2 * torch.pi) * modulus).long() % modulus


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["A", "B", "C", "Cp", "D"], required=True,
                   help="A=plain b; B=+carry head; C=aux modulus (Kikuchi); "
                        "Cp=ablation, dual inputs but mod-q labels only; "
                        "D=hybrid C+B (label switch AND carry head)")
    p.add_argument("--npz", required=True, help="output of make_synthetic_labels.py")
    p.add_argument("--grid_dir", required=True, help="dir with train_A.npy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--secret_idx", type=int, default=0,
                   help="which invented secret to train on (one per run)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux_r", type=float, default=0.3,
                   help="arm C: probability a sample's label is mod Kq (Eq. 2)")
    p.add_argument("--carry_weight", type=float, default=1.0,
                   help="arm B: weight of the k cross-entropy term")
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data -------------------------------------------------------------
    z = np.load(args.npz)
    secrets, b_all, k_all, bK_all = z["secrets"], z["b"], z["k"], z["bK"]
    K = int(z["K"])
    A_shift = int(z["A_shift"])
    q = args.Q

    A = np.load(os.path.join(args.grid_dir, "train_A.npy"))
    X = torch.from_numpy(transform_A(A, A.shape[1], q, A_shift))

    # IMPORTANT (caught in review): b depends on the secret, and the model
    # only sees the row — so predicting b for an unseen secret is impossible
    # by construction. The well-posed task (and what FRESCA does) is
    # per-secret: train on one secret's samples, evaluate on held-out ROWS
    # of that same secret. The arm comparison is unchanged.
    j_secret = args.secret_idx
    b_t = torch.from_numpy(b_all)
    k_t = torch.from_numpy(k_all)
    bK_t = torch.from_numpy(bK_all)
    n_k_classes = int(k_all.max()) + 1

    # Split ROWS: last 10% are eval-only, never trained on.
    n_rows = len(A)
    split = int(n_rows * 0.9)
    train_rows_pool = np.arange(split)
    eval_rows_pool = np.arange(split, n_rows)

    # ---- model ------------------------------------------------------------
    # Minimal params namespace with just the fields the repo Encoder reads.
    class P:
        pass
    mp = P()
    mp.enc_emb_dim = args.emb_dim
    mp.n_enc_layers = args.layers
    mp.n_enc_heads = args.heads
    mp.dropout = 0.0
    mp.attention_dropout = 0.0
    mp.Q = q

    if args.arm == "A":
        model = Encoder(mp)
    elif args.arm == "B":
        model = CarryEncoder(mp, n_k_classes)
    elif args.arm in ("C", "Cp"):
        model = DualAngularEncoder(mp)
    else:  # D
        model = HybridEncoder(mp, n_k_classes)
    model = model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def encode_inputs(rows):
        """A row of X -> angular tokens. Dual-circle arms get 4-dim inputs."""
        if args.arm in ("C", "Cp", "D"):
            return torch.cat([to_circle(rows, q), to_circle(rows, K * q)], dim=-1)
        return to_circle(rows, q)

    # ---- evaluation on held-out rows of the SAME secret --------------------
    eval_rows = rng.choice(eval_rows_pool, size=2000, replace=False)
    eval_X = encode_inputs(X[eval_rows]).to(dev)
    eval_true = b_t[j_secret, eval_rows]

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = model(eval_X)
        if args.arm in ("B", "D"):
            out = out[0]
        pred = decode_circle(out[:, :2], q).cpu()
        acc = (pred == eval_true).float().mean().item()
        # within-noise accuracy: |pred - true| <= 6 (2 sigma) circularly,
        # since exact match is capped by the LWE noise itself
        d = (pred - eval_true) % q
        d = torch.minimum(d, q - d)
        near = (d <= 6).float().mean().item()
        model.train()
        return acc, near

    # ---- training loop -----------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"arm{args.arm}_s{args.secret_idx}_seed{args.seed}"
    log_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    log = open(log_path, "a")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # One batch = random training rows of the one secret. Simple on
        # purpose; all arms share this exact schedule and seed.
        j = j_secret
        rows = rng.choice(train_rows_pool, size=args.batch_size)
        xb = encode_inputs(X[rows]).to(dev)

        if args.arm == "A":
            target = to_circle(b_t[j, rows], q).to(dev)
            loss = mse(model(xb), target)

        elif args.arm == "B":
            target = to_circle(b_t[j, rows], q).to(dev)
            kb = k_t[j, rows].to(dev)
            out_b, out_k = model(xb)
            loss = mse(out_b, target) + args.carry_weight * ce(out_k, kb)

        elif args.arm == "Cp":
            # Ablation: dual-circle inputs like C, but ONLY mod-q labels.
            # Isolates how much of arm C's gain is the richer input encoding.
            target = to_circle(b_t[j, rows], q).to(dev)
            loss = mse(model(xb)[:, :2], target)

        else:  # arms C and D: per-sample label switch, Eq. (2) of the paper
            out = model(xb)
            if args.arm == "D":
                out, out_k = out
            # (batch, 4): dims 0:2 = mod q circle, dims 2:4 = mod Kq circle
            use_aux = torch.from_numpy(
                rng.random(args.batch_size) < args.aux_r)
            t_q = to_circle(b_t[j, rows], q).to(dev)
            t_K = to_circle(bK_t[j, rows], K * q).to(dev)
            # Loss on the selected circle only, per sample.
            sel = use_aux.to(dev)
            loss = (mse(out[~sel, :2], t_q[~sel]) * (~sel).float().mean()
                    + mse(out[sel, 2:], t_K[sel]) * sel.float().mean())
            if args.arm == "D":
                # Hybrid: add our carry-head loss on top of the label switch.
                kb = k_t[j, rows].to(dev)
                loss = loss + args.carry_weight * ce(out_k, kb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0:
            acc, near = evaluate()
            rec = {"step": step, "loss": float(loss.item()),
                   "heldout_match_acc": acc, "heldout_within_2sigma": near,
                   "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(args.out_dir, f"{tag}.pt"))
    print(f"saved {tag}.pt and {tag}.jsonl in {args.out_dir}")


if __name__ == "__main__":
    main()
