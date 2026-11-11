#!/usr/bin/env python3
"""
Dead-core-seeded EM attack.

The insight (measured, 2026-11-11): the EM basin is NOT a Hamming-distance
budget. EM self-heals missing bits on high-signal columns; what it cannot fix
are missing bits on "dead" columns. Which columns are dead is computable from
A alone, via the per-column characteristic function:

    eps_j = | mean over samples of exp(2*pi*i * a_j / q) |

eps_j measures how much signal survives when secret bit j is unaccounted for
(1 = full signal, 0 = one missing bit flattens every statistic). On this
dataset the 80 columns split into ~38 strong (eps>0.5), ~12 lukewarm
(0.02<eps<=0.5), ~30 dead (eps<=0.02, at the sampling noise floor).

Attack: enumerate small patterns over the dead (optionally dead+lukewarm)
columns in ascending weight; run EM from each pattern (all other coordinates
start at 0); verify every result with the residual test on original samples.
A pattern that gets the dead-core bits right is inside the basin even though
its Hamming distance to the truth is large (all strong bits still missing).

Measured result: h10_s2 - which Cool&Cruel failed in a 20-minute budget -
recovered in 10 EM runs / 80 s (its single dead-core bit is found at
weight 1). Cost scales with C(|hard set|, #hard ones), not with
C(53, #cruel ones) as in Cool&Cruel.

Usage:  python3 src/em_wrap/deadcore_em.py --hamming 10 --secret_seed 2
        python3 src/em_wrap/deadcore_em.py --hamming 8 --secret_seed 3 \
            --include_lukewarm --max_weight 4
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


def column_eps(X, q):
    """Per-column blurring factor from A alone (no secret involved)."""
    return np.abs(np.exp(2j * np.pi * X / q).mean(axis=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secret_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../testn80_grid/binary_secrets_h6_12"))
    p.add_argument("--hamming", type=int, required=True)
    p.add_argument("--secret_seed", type=int, required=True)
    p.add_argument("--max_weight", type=int, default=2,
                   help="max number of ones in the enumerated seed pattern")
    p.add_argument("--include_lukewarm", action="store_true",
                   help="enumerate over dead+lukewarm columns (eps<=0.5) "
                        "instead of dead only (eps<=0.02)")
    p.add_argument("--dead_eps", type=float, default=0.02)
    p.add_argument("--lukewarm_eps", type=float, default=0.5)
    p.add_argument("--em_iters", type=int, default=60)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--A_shift", type=int, default=42)
    p.add_argument("--sigma", type=float, default=3.0)
    args = p.parse_args()

    X, y, verify_A, verify_b = load_data(
        args.secret_dir, args.hamming, args.secret_seed, args.N, args.Q, args.A_shift)

    eps = column_eps(X, args.Q)
    threshold = args.lukewarm_eps if args.include_lukewarm else args.dead_eps
    hard = np.where(eps <= threshold)[0]
    # Enumerate hardest columns first inside each weight class.
    hard = hard[np.argsort(eps[hard])]

    t0, n_runs = time.time(), 0
    for weight in range(args.max_weight + 1):
        for pattern in itertools.combinations(hard, weight):
            seed = np.zeros(args.N)
            seed[list(pattern)] = 1.0
            estimate = em_fit(X, y, args.Q, n_iters=args.em_iters, init=seed)
            n_runs += 1
            for guess in binary_candidates(estimate):
                std = residual_std(verify_A, verify_b, guess, args.Q)
                if std < 2 * args.sigma:
                    print(json.dumps({
                        "matched": True,
                        "h": args.hamming, "secret_seed": args.secret_seed,
                        "winning_pattern": [int(i) for i in pattern],
                        "n_runs": n_runs,
                        "wall_sec": round(time.time() - t0, 1),
                        "hard_set_size": int(len(hard)),
                        "recovered_support": sorted(np.where(guess == 1)[0].tolist()),
                    }))
                    return
            if n_runs % 500 == 0:
                print(f"# {n_runs} runs, {time.time()-t0:.0f}s, "
                      f"at weight {weight}", flush=True)

    print(json.dumps({
        "matched": False,
        "h": args.hamming, "secret_seed": args.secret_seed,
        "n_runs": n_runs, "hard_set_size": int(len(hard)),
        "wall_sec": round(time.time() - t0, 1),
    }))


if __name__ == "__main__":
    main()
