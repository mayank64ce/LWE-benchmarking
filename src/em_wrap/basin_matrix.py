#!/usr/bin/env python3
"""
2D basin measurement: EM success as a function of (d_minus, d_plus).

d_minus = number of TRUE bits missing from the start (removed ones)
d_plus  = number of FALSE bits added to the start (extra ones)
Hamming distance = d_minus + d_plus, but the two error types may behave
differently (observed: pair seeds, which are pure-d_minus starts, do worse
than the mixed-error basin test predicted at equal distance).

Additionally, each trial records how many of its errors landed on CRUEL
columns (the 53 high-variance ones), so we can test whether the basin is
really a cruel-error budget rather than a global Hamming budget.

Diagnostic: uses the true secret to construct starts; success is still
judged only by the residual test.

Usage:  python3 src/em_wrap/basin_matrix.py --hamming 8 --secret_seed 3
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
    p.add_argument("--n_trials", type=int, default=15)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--A_shift", type=int, default=42)
    p.add_argument("--sigma", type=float, default=3.0)
    args = p.parse_args()

    X, y, vA, vb = load_data(args.secret_dir, args.hamming, args.secret_seed,
                             args.N, args.Q, args.A_shift)
    s_true = np.load(os.path.join(
        args.secret_dir, f"secret_{args.hamming}_{args.secret_seed}.npy")
    ).flatten().astype(float)
    ones = np.where(s_true == 1)[0]
    zeros = np.where(s_true == 0)[0]

    # Cruel columns = the high-variance ones (Z-shape); used only to report
    # where each trial's errors landed.
    col_std = X.std(axis=0)
    cruel = set(np.argsort(col_std)[27:].tolist())  # top 53 by variance

    rng = np.random.RandomState(1000 * args.hamming + args.secret_seed + 7)

    print("d_minus,d_plus,trials,successes,avg_cruel_errors_success,"
          "avg_cruel_errors_fail")
    for d_minus in range(0, min(args.hamming, 6) + 1):
        for d_plus in range(0, 4):
            if d_minus == 0 and d_plus == 0:
                continue
            wins, cruel_s, cruel_f = 0, [], []
            for _ in range(args.n_trials):
                init = s_true.copy()
                removed = rng.choice(ones, d_minus, replace=False)
                added = rng.choice(zeros, d_plus, replace=False)
                init[removed] = 0
                init[added] = 1
                n_cruel_err = sum(1 for i in list(removed) + list(added)
                                  if i in cruel)
                est = em_fit(X, y, args.Q, init=init)
                ok = any(residual_std(vA, vb, g, args.Q) < 2 * args.sigma
                         for g in binary_candidates(est))
                wins += ok
                (cruel_s if ok else cruel_f).append(n_cruel_err)
            avg = lambda v: round(float(np.mean(v)), 2) if v else ""
            print(f"{d_minus},{d_plus},{args.n_trials},{wins},"
                  f"{avg(cruel_s)},{avg(cruel_f)}", flush=True)


if __name__ == "__main__":
    main()
