#!/usr/bin/env python3
"""
Measure the EM attack's basin of attraction, in bits.

Question: how close (Hamming distance) must EM's starting point be to the
true secret for EM to converge to it? We start EM from the true secret with
k bits deliberately flipped (k = 0..8, random positions, several trials)
and report the recovery rate per k.

This uses the true secret BY DESIGN - it is a diagnostic of the method, not
an attack. Recovery is still judged by the residual test on original
samples, same as everywhere else.

Usage:  python3 src/em_wrap/basin_test.py --hamming 8 --secret_seed 3
"""

import argparse
import os
import sys

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
    p.add_argument("--max_flips", type=int, default=8)
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--A_shift", type=int, default=42)
    p.add_argument("--sigma", type=float, default=3.0)
    args = p.parse_args()

    X, y, verify_A, verify_b = load_data(
        args.secret_dir, args.hamming, args.secret_seed, args.N, args.Q, args.A_shift)
    s_true = np.load(os.path.join(
        args.secret_dir, f"secret_{args.hamming}_{args.secret_seed}.npy")
    ).flatten().astype(float)

    rng = np.random.RandomState(0)
    print(f"h{args.hamming}_s{args.secret_seed}:", end=" ")
    for flips in range(args.max_flips + 1):
        wins, trials = 0, (1 if flips == 0 else args.n_trials)
        for _ in range(trials):
            init = s_true.copy()
            idx = rng.choice(args.N, flips, replace=False)
            init[idx] = 1 - init[idx]  # flip -> start is `flips` bits from truth
            estimate = em_fit(X, y, args.Q, init=init)
            for guess in binary_candidates(estimate):
                if residual_std(verify_A, verify_b, guess, args.Q) < 2 * args.sigma:
                    wins += 1
                    break
        print(f"{flips}flips:{wins}/{trials}", end="  ", flush=True)
    print()


if __name__ == "__main__":
    main()
