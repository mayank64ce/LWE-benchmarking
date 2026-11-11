#!/usr/bin/env python3
"""
Slack test at n=256 Kyber scale with ~100k samples (vs 14.5k before).

Question: with more equations per unknown, can EM tolerate MISSING dead-column
entries in its seed? At 14.5k samples the answer was 0 (every dead entry had
to be seeded exactly). At n=80 (625 eqs/unknown) the slack was ~2 bits, which
is what made enumeration affordable.

Protocol per secret:
  seed = true secret restricted to dead columns (strong columns zeroed),
  then zero out k of the dead nonzero entries (all C(n_dead_nz, k) ways),
  run EM, round, apply one-coordinate verified repair, check exact recovery.
Diagnostic: uses true secret for seed construction; success is judged by the
residual test on original (unreduced) samples only.

Usage: python3 n256_slack_test.py <hamming> <seed> [grid_dir]
"""
import itertools
import sys
import time

import numpy as np

Q = 179067461
N, K, RING_N = 512, 2, 256


def transform(A, shift=0):
    A2 = np.flip(A.reshape((len(A), K, RING_N)), axis=2)
    A2 = np.concatenate((A2[..., shift:], -A2[..., :shift]), axis=-1) % Q
    return A2.reshape((len(A2), N))


def center(x):
    return ((x + Q // 2) % Q) - Q // 2


def em_fit(X, y, s0, iters=25):
    s = s0.astype(float).copy()
    for _ in range(iters):
        r = center((y - X @ s) % Q)
        yu = X @ s + r
        sc = np.median(np.abs(r)) / 0.6745 + 1e-9
        w = (1 - (r / (6 * sc)) ** 2) ** 2
        w[np.abs(r) > 6 * sc] = 0.0
        Xw = X * w[:, None]
        s = np.linalg.solve(Xw.T @ X + 1e-6 * np.eye(N), Xw.T @ yu)
    return s


def main():
    h, sd = int(sys.argv[1]), int(sys.argv[2])
    grid = sys.argv[3] if len(sys.argv) > 3 else "testn256_28_grid"

    X = transform(np.load(f"{grid}/train_A.npy").astype(np.int64)).astype(float)
    tb = np.load(f"{grid}/binomial_secrets_h13_19/train_b_{h}_{sd}.npy")
    y = tb[:, ~0].astype(float)
    vA = transform(np.load(f"{grid}/orig_A.npy").astype(np.int64))
    vb = np.load(f"{grid}/binomial_secrets_h13_19/orig_b_{h}_{sd}.npy")[:, ~0]
    secrets = np.load(f"{grid}/binomial_secrets_h13_19/secret.npy").astype(np.int64)
    s_true = secrets[:, (h - 13) * 10 + sd].astype(float)

    # eps profile in attack (transformed) coordinates, from this grid's own A
    eps = np.abs(np.exp(2j * np.pi * X / Q).mean(axis=0))
    dead = eps <= 0.1
    dead_nz = np.where((s_true != 0) & dead)[0]
    print(f"h={h} s={sd}: {len(X)} samples, {int(dead.sum())} dead cols, "
          f"{len(dead_nz)} dead nonzeros {dead_nz.tolist()}", flush=True)

    def resid_vec(g):
        return center((vA @ g - vb) % Q)

    def repair(g):
        if resid_vec(g).std() < 2.0:
            return g, True
        base = resid_vec(g)
        for j in range(N):
            for v in (-2, -1, 0, 1, 2):
                if v == g[j]:
                    continue
                if center((base + (v - g[j]) * vA[:, j]) % Q).std() < 2.0:
                    g2 = g.copy()
                    g2[j] = v
                    return g2, True
        return g, False

    full_seed = np.where(dead, s_true, 0.0)
    for k in range(0, min(3, len(dead_nz)) + 1):
        wins = tot = 0
        t0 = time.time()
        combos = list(itertools.combinations(range(len(dead_nz)), k))
        if len(combos) > 15:  # cap work at k=3 for large supports
            rng = np.random.RandomState(0)
            combos = [combos[i] for i in
                      rng.choice(len(combos), 15, replace=False)]
        for drop in combos:
            s0 = full_seed.copy()
            s0[dead_nz[list(drop)]] = 0.0
            est = em_fit(X, y, s0)
            g = np.clip(np.round(est), -2, 2).astype(np.int64)
            g, ok = repair(g)
            wins += int(ok and (g == s_true).all())
            tot += 1
        print(f"  {k} dead entries missing: {wins}/{tot} recovered "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
