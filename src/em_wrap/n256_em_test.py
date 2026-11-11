#!/usr/bin/env python3
"""
First EM test at Kyber scale (n=512 total dimension, q=3329, binomial secrets).

Question: does the dead-core-budget theory from n=80 transfer?
Theory predicts:
  (a) EM started AT the true secret stays there (sanity).
  (b) EM started with all STRONG-column entries zeroed, but DEAD-column
      entries correct, recovers the secret (EM self-heals strong columns).
  (c) EM started with all DEAD-column entries zeroed, but STRONG-column
      entries correct, FAILS (dead columns give EM no signal to heal).
If (b) holds, the attack logic transfers and total cost = cost of guessing
the dead-column entries. If (b) fails, the basin is smaller at this scale.

Diagnostic only: uses true secrets to build starting points. Success is
still judged by the residual test on original (unreduced) samples.
"""
import numpy as np
import sys

Q = 3329
N, K, RING_N = 512, 2, 256
ERR_STD = 1.0          # measured: true-secret residual std on orig data
GRID = "testn256_grid"


def transform(A, shift=0):
    """Repo's MLWEiDataset.transform for k=2: flip each block, negacyclic shift."""
    A2 = np.flip(A.reshape((len(A), K, RING_N)), axis=2)
    A2 = np.concatenate((A2[..., shift:], -A2[..., :shift]), axis=-1) % Q
    return A2.reshape((len(A2), N))


def center(x):
    return ((x + Q // 2) % Q) - Q // 2


def residual_std(A, b, guess):
    return center((A @ guess - b) % Q).std()


def em_fit(X, y, s_init, n_iters=40, c=6.0):
    """Mod-aware EM, same algorithm as n=80 (em_wrap/main.py), values unrestricted.
    E-step: unwrap y using current estimate. M-step: Tukey-weighted least squares."""
    s_hat = s_init.astype(float).copy()
    for _ in range(n_iters):
        r = center((y - X @ s_hat) % Q)          # wrapped residuals
        y_unwrapped = X @ s_hat + r               # unwrap onto the real line
        scale = np.median(np.abs(r)) / 0.6745 + 1e-9
        w = (1 - (r / (c * scale)) ** 2) ** 2
        w[np.abs(r) > c * scale] = 0.0
        Xw = X * w[:, None]
        s_hat = np.linalg.solve(Xw.T @ X + 1e-6 * np.eye(N), Xw.T @ y_unwrapped)
    return s_hat


def candidates(est):
    """Binomial secrets: round each coordinate to nearest integer in [-2, 2]."""
    yield np.clip(np.round(est), -2, 2).astype(np.int64)


def main():
    h, seed = int(sys.argv[1]), int(sys.argv[2])
    X = transform(np.load(f"{GRID}/train_A.npy").astype(np.int64)).astype(float)
    tb = np.load(f"{GRID}/binomial_secrets_h6_10/train_b_{h}_{seed}.npy")
    y = tb[:, ~0].astype(float)                   # b column paired with shift=0
    vA = transform(np.load(f"{GRID}/orig_A.npy").astype(np.int64))
    vb = np.load(f"{GRID}/binomial_secrets_h6_10/orig_b_{h}_{seed}.npy")[:, ~0]
    secrets = np.load(f"{GRID}/binomial_secrets_h6_10/secret.npy").astype(np.int64)
    s_true = secrets[:, (h - 6) * 10 + seed].astype(float)

    eps = np.load(f"{GRID}/eps_profile_transformed.npy")
    dead = eps <= 0.1
    nz = s_true != 0
    print(f"h={h} seed={seed}: {int(nz.sum())} nonzeros, "
          f"{int((nz & dead).sum())} on dead columns")
    print(f"true-secret check: orig resid {residual_std(vA, vb, s_true):.2f}, "
          f"reduced resid {center((y - X @ s_true) % Q).std():.1f}")

    starts = {
        "(a) at truth": s_true.copy(),
        "(b) strong zeroed, dead kept": np.where(dead, s_true, 0.0),
        "(c) dead zeroed, strong kept": np.where(dead, 0.0, s_true),
        "(d) all zeros (cold)": np.zeros(N),
    }
    for name, s0 in starts.items():
        est = em_fit(X, y, s0)
        ok, best = False, 1e9
        for g in candidates(est):
            r = residual_std(vA, vb, g)
            best = min(best, r)
            if r < 2 * ERR_STD:
                ok = True
        print(f"{name}: recovered={ok}  best orig resid={best:.2f}")


if __name__ == "__main__":
    main()
