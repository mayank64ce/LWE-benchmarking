#!/usr/bin/env python3
"""
Probe the positional embeddings of trained SALSA models for secret
information - including models whose secret recovery FAILED.

Background: Saxena & Lauter ("How do Small Transformer Models Learn Hard
Math Tasks?") show that in models that successfully recover the secret, the
secret is linearly readable from the positional embeddings. They only probe
successful models. This script asks the complementary question: do FAILED
models' embeddings still hold (partial) secret information?

Method: load each run's checkpoint, take the first N rows of
pos_emb.weight (position i = coordinate i of the transformed a), and score
each position three ways: row L2 norm, |projection on PC1|, |projection on
PC2|. Compare the scores against the true secret (diagnostic only):
  - AUC: probability a random secret position outscores a random
    non-secret position (0.5 = chance, 1.0 = perfect, sign-agnostic)
  - median and max rank of the true bits under the better direction.

Usage:  python3 src/em_wrap/probe_pe.py --logs ../../b1_runs/logs/b1 \
            --secret_dir ../../testn80_grid/binary_secrets_h6_12
(defaults match the toy grid layout)
"""

import argparse
import os

import numpy as np
import torch


def pe_scores(checkpoint_path, n):
    """Per-position scores from one checkpoint's positional embedding."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pe = ck["model"]["pos_emb.weight"][:n].numpy()
    centered = pe - pe.mean(0)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    return {
        "norm": np.linalg.norm(pe, axis=1),
        "pc1": np.abs(U[:, 0] * S[0]),
        "pc2": np.abs(U[:, 1] * S[1]),
    }


def auc_and_ranks(score, secret):
    """Sign-agnostic AUC and (median, max) rank of the true bits."""
    pos, neg = score[secret == 1], score[secret == 0]
    auc = (pos[:, None] > neg[None, :]).mean()
    flip = auc < 0.5
    auc = max(auc, 1 - auc)
    order = np.argsort(score) if flip else np.argsort(-score)
    ranks = np.where(np.isin(order, np.where(secret == 1)[0]))[0] + 1
    return round(float(auc), 2), int(np.median(ranks)), int(ranks.max())


def zscore(x):
    return (x - x.mean()) / x.std()


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=os.path.join(base, "../../../b1_runs/logs/b1"))
    p.add_argument("--secret_dir", default=os.path.join(
        base, "../../../testn80_grid/binary_secrets_h6_12"))
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--hs", type=int, nargs="+", default=[6, 8, 10])
    p.add_argument("--secret_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--model_seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    print("secret | per-score (AUC, med rank, max rank) for norm/pc1/pc2, "
          "ensembled over model seeds")
    for h in args.hs:
        for ss in args.secret_seeds:
            secret = np.load(os.path.join(
                args.secret_dir, f"secret_{h}_{ss}.npy")).flatten()
            runs = []
            for m in args.model_seeds:
                path = os.path.join(args.logs, f"h{h}_s{ss}_m{m}", "checkpoint.pth")
                if os.path.exists(path):
                    runs.append(pe_scores(path, args.N))
            if not runs:
                continue
            report = {}
            for key in ("norm", "pc1", "pc2"):
                ensemble = np.mean([zscore(r[key]) for r in runs], axis=0)
                report[key] = auc_and_ranks(ensemble, secret)
            print(f"h{h}_s{ss} (runs={len(runs)})", report, flush=True)


if __name__ == "__main__":
    main()
