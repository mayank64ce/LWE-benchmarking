"""Step 3b: fine-tune a pretrained model on a REAL victim secret, b-only.

This is the attack-relevant test. The victim gives us only (row, b) pairs —
no wrap counts, no aux labels. Protocol copied from Fresca Section 5.2:
take the pretrained weights, give the victim a FRESH secret token (the
reserved embedding slot, randomly initialized), and train on b labels only.

Arms compared (same fine-tune procedure for all):
  cold      random init, no pretraining       (lower control)
  A         Fresca-style b-only pretraining   (the real baseline to beat)
  B         carry-head pretraining (ours)     (k head unused here: no labels)
  C         aux-modulus pretraining (Kikuchi) (loss on the mod-q circle only)

Victim = one of the repo's real h=8 secrets (train_b_8_{seed}.npy), the
regime where the published SALSA recipe recovers 0/15. Fine-tune uses only
rows [25000, end) of train_A — disjoint from the pretraining rows — with the
last 2000 rows held out for eval.

Run:
  python3 finetune.py --checkpoint ../../../carry_pretrain_runs/pretrain_armC_m32_seed0.pt \
      --grid_dir ../../../testn80_grid --hamming 8 --secret_seed 0 \
      --out_dir ../../../carry_pretrain_runs
  (or --cold --arm A for the no-pretraining control)
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

from make_synthetic_labels import transform_A  # noqa: E402
from pretrain import to_circle, decode_circle  # noqa: E402
from pretrain_multi import SecretTokenModel, build_base, PRETRAIN_ROWS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="",
                   help="pretrain_multi.py checkpoint; empty with --cold")
    p.add_argument("--cold", action="store_true",
                   help="no pretraining: random init control")
    p.add_argument("--arm", choices=["A", "B", "C"], default="",
                   help="required with --cold; otherwise read from checkpoint")
    p.add_argument("--grid_dir", required=True)
    p.add_argument("--hamming", type=int, required=True)
    p.add_argument("--secret_seed", type=int, required=True)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--A_shift", type=int, default=42)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = args.Q

    # ---- rebuild the model exactly as pretrained --------------------------
    if args.cold:
        assert args.arm, "--cold needs --arm"
        arm, num_secrets, K, n_k = args.arm, 32, 4, 8
        emb_dim, layers, heads = 256, 4, 8
    else:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ca = ck["args"]
        arm = ca["arm"]
        num_secrets, K = ca["num_secrets"], 4
        n_k = ck["n_k_classes"]
        emb_dim, layers, heads = ca["emb_dim"], ca["layers"], ca["heads"]

    base = build_base(arm, emb_dim, layers, heads, q, n_k)
    model = SecretTokenModel(base, num_secrets, emb_dim)
    if not args.cold:
        model.load_state_dict(ck["model"])
    model = model.to(dev)

    # The victim's fresh secret token: reserved slot num_secrets, which the
    # pretraining never used, so it keeps its random initialization (Fresca:
    # "randomly initialize the new token embedding").
    victim_tok = num_secrets

    # ---- victim data: REAL b files from the repo grid ----------------------
    A = np.load(os.path.join(args.grid_dir, "train_A.npy"))
    X = torch.from_numpy(transform_A(A, A.shape[1], q, args.A_shift))
    b_file = os.path.join(args.grid_dir, "binary_secrets_h6_12",
                          f"train_b_{args.hamming}_{args.secret_seed}.npy")
    # Column ~shift pairs with the transformed rows (repo convention).
    b = torch.from_numpy(np.load(b_file)[:, ~args.A_shift].astype(np.int64))

    # Victim rows: disjoint from pretraining; last 2000 held out for eval.
    victim_rows = np.arange(PRETRAIN_ROWS, len(X) - 2000)
    eval_rows = np.arange(len(X) - 2000, len(X))

    def encode_inputs(rows):
        if arm == "C":
            return torch.cat([to_circle(rows, q), to_circle(rows, K * q)], dim=-1)
        return to_circle(rows, q)

    eval_X = encode_inputs(X[eval_rows]).to(dev)
    eval_sec = torch.full((len(eval_rows),), victim_tok, dtype=torch.long,
                          device=dev)
    eval_true = b[eval_rows]

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = model(eval_X, eval_sec)
        if arm == "B":
            out = out[0]
        pred = decode_circle(out[:, :2], q).cpu()
        d = (pred - eval_true) % q
        d = torch.minimum(d, q - d)
        model.train()
        return (pred == eval_true).float().mean().item(), (d <= 6).float().mean().item()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    mse = nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    init = "cold" if args.cold else "pre"
    tag = f"ft_{init}{arm}_h{args.hamming}_s{args.secret_seed}_seed{args.seed}"
    log = open(os.path.join(args.out_dir, f"{tag}.jsonl"), "a")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        rows = rng.choice(victim_rows, size=args.batch_size)
        xb = encode_inputs(X[rows]).to(dev)
        sec = torch.full((args.batch_size,), victim_tok, dtype=torch.long,
                         device=dev)
        out = model(xb, sec)
        if arm == "B":
            out = out[0]       # k head has no victim labels; unused
        # b-only loss. For arm C that means the mod-q circle only (r = 0):
        # the victim never hands us a mod-Kq label.
        loss = mse(out[:, :2], to_circle(b[rows], q).to(dev))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0:
            acc, near = evaluate()
            rec = {"step": step, "loss": float(loss.item()),
                   "victim_match_acc": acc, "victim_within_2sigma": near,
                   "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

    print(f"done: {tag}")


if __name__ == "__main__":
    main()
