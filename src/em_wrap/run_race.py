#!/usr/bin/env python3
"""
Horse race: run every initialization strategy on every (h, secret) pair of
the toy kill-test grid, and record which strategy recovers which secret.

For each pair we run ALL strategies (we want the comparison table, not just
a win), verify every candidate against the original samples (residual std
< 2*sigma, never reading the true secret), and write one CSV row per
(h, secret, strategy).

Resumable: rows already in the CSV are skipped.

Usage:  python3 src/em_wrap/run_race.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import em_fit, binary_candidates, residual_std, load_data
from strategies import em_fit_annealed, cool_start, greedy_pursuit, restart_inits

# --- toy grid configuration (same as the baselines) ---
SECRET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../testn80_grid/binary_secrets_h6_12",
)
N, Q, SHIFT, SIGMA = 80, 113, 42, 3.0
HS = [6, 8, 10, 12]
SECRET_SEEDS = range(5)
N_RESTARTS = 200
OUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../experiments/em_wrap/race_results.csv",
)


def verify_estimate(s_hat, verify_A, verify_b):
    """Round a real-valued estimate every way we know and check each guess.

    Returns (matched, best_residual_std). Checking is the cheap part.
    """
    best = float("inf")
    for guess in binary_candidates(s_hat):
        std = residual_std(verify_A, verify_b, guess, Q)
        best = min(best, std)
        if std < 2 * SIGMA:
            return True, best
    return False, best


def run_strategy(name, X, y, verify_A, verify_b):
    """Run one named strategy. Returns (matched, best_std, n_attempts)."""
    if name == "cold":
        # The known baseline: single EM run from plain OLS.
        matched, best = verify_estimate(em_fit(X, y, Q), verify_A, verify_b)
        return matched, best, 1

    if name == "anneal":
        matched, best = verify_estimate(em_fit_annealed(X, y, Q), verify_A, verify_b)
        return matched, best, 1

    if name == "cool_start":
        matched, best = verify_estimate(cool_start(X, y, Q), verify_A, verify_b)
        return matched, best, 1

    if name == "greedy":
        # Verify every greedy prefix, and also EM-refine each prefix and
        # verify that too (the greedy guess may be close but not exact).
        best = float("inf")
        for i, s in enumerate(greedy_pursuit(X, y, Q)):
            m, b1 = verify_estimate(s, verify_A, verify_b)
            best = min(best, b1)
            if m:
                return True, best, i + 1
            m, b2 = verify_estimate(em_fit(X, y, Q, init=s), verify_A, verify_b)
            best = min(best, b2)
            if m:
                return True, best, i + 1
        return False, best, 20

    if name == "restarts":
        best = float("inf")
        for i, init in enumerate(restart_inits(X, y, Q, N_RESTARTS)):
            m, b = verify_estimate(em_fit(X, y, Q, init=init), verify_A, verify_b)
            best = min(best, b)
            if m:
                return True, best, i + 1
        return False, best, N_RESTARTS

    raise ValueError(name)


STRATEGIES = ["cold", "anneal", "cool_start", "greedy", "restarts"]


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add((int(row["h"]), int(row["secret_seed"]), row["strategy"]))
    else:
        with open(OUT_CSV, "w") as f:
            f.write("h,secret_seed,strategy,matched,best_resid_std,wall_sec,n_attempts\n")

    for h in HS:
        for ss in SECRET_SEEDS:
            X, y, verify_A, verify_b = load_data(SECRET_DIR, h, ss, N, Q, SHIFT)
            for strat in STRATEGIES:
                if (h, ss, strat) in done:
                    continue
                t0 = time.time()
                matched, best, n_att = run_strategy(strat, X, y, verify_A, verify_b)
                wall = time.time() - t0
                with open(OUT_CSV, "a") as f:
                    f.write(f"{h},{ss},{strat},{int(matched)},{best:.2f},{wall:.1f},{n_att}\n")
                print(f"h{h}_s{ss} {strat}: matched={int(matched)} "
                      f"best_std={best:.2f} wall={wall:.0f}s attempts={n_att}",
                      flush=True)
    print("race complete")


if __name__ == "__main__":
    main()
