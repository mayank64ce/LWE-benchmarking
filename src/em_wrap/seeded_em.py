#!/usr/bin/env python3
"""
Pair-seeded EM attack.

Motivation (measured, see basin test): every secret on the toy grid is a
stable attractor for the EM loop with a basin radius of ~4 wrong bits.
So we do not need a clever initializer - we need ANY start within ~4 bits
of the truth. Enumerating all pairs of coordinates provides that:

  seed(i,j) = the binary vector with ones exactly at positions i and j.

If both seeded positions are true secret bits, the seed is (h-2) flips from
the truth: inside the basin for h=6 (4 flips, ~70% per try), marginal for
h=8 (6 flips, ~15% per try, but C(h,2)=28 correct pairs exist), out of
range for h=10 via pairs.

There are C(80,2)=3160 pairs; each EM run takes ~1.3 s, so a full sweep is
~70 minutes per secret. Every candidate is verified with the residual test
on the original samples (never the true secret). We stop at first success.

Usage:  python3 src/em_wrap/seeded_em.py --hamming 8 --secret_seed 3
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import em_fit, binary_candidates, residual_std, load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secret_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../testn80_grid/binary_secrets_h6_12"))
    p.add_argument("--hamming", type=int, required=True)
    p.add_argument("--secret_seed", type=int, required=True)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--A_shift", type=int, default=42)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--em_iters", type=int, default=60)
    args = p.parse_args()

    X, y, verify_A, verify_b = load_data(
        args.secret_dir, args.hamming, args.secret_seed, args.N, args.Q, args.A_shift)

    t0 = time.time()
    n_seeds = 0
    # Enumerate all pairs (i, j). Order is plain lexicographic - no oracle.
    for i, j in itertools.combinations(range(args.N), 2):
        seed = np.zeros(args.N)
        seed[i] = seed[j] = 1.0
        estimate = em_fit(X, y, args.Q, n_iters=args.em_iters, init=seed)
        n_seeds += 1

        for guess in binary_candidates(estimate):
            std = residual_std(verify_A, verify_b, guess, args.Q)
            if std < 2 * args.sigma:
                print(json.dumps({
                    "matched": True,
                    "h": args.hamming, "secret_seed": args.secret_seed,
                    "winning_seed_pair": [i, j],
                    "n_seeds_tried": n_seeds,
                    "wall_sec": round(time.time() - t0, 1),
                    "resid_std": round(float(std), 2),
                    "recovered_support": sorted(np.where(guess == 1)[0].tolist()),
                }))
                return

        if n_seeds % 200 == 0:
            print(f"# {n_seeds} seeds tried, {time.time()-t0:.0f}s", flush=True)

    print(json.dumps({
        "matched": False,
        "h": args.hamming, "secret_seed": args.secret_seed,
        "n_seeds_tried": n_seeds,
        "wall_sec": round(time.time() - t0, 1),
    }))


if __name__ == "__main__":
    main()
