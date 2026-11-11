#!/usr/bin/env python3
"""
EM-over-wrap-counts attack on sparse-secret (R)LWE.

The idea in one paragraph
-------------------------
Every reduced LWE sample satisfies  b = a.s + e - k*q  where k is an unknown
integer (the "wrap count"). If we knew k for every sample, recovering the
secret s would be one ordinary least-squares solve. If we knew s, we could
compute every k directly. We know neither, so we alternate (this is EM):

    E-step: given the current estimate s_hat, guess each sample's wrap:
            k_hat = round((a.s_hat - b) / q)
            and "unwrap" the label:  y_unwrapped = b + k_hat*q
    M-step: refit s_hat by (weighted) least squares on the unwrapped labels.

Tukey biweight weights make the refit ignore samples whose current residual
is large (those are the samples whose wrap guess is probably wrong).

Verification never reads the true secret: a candidate secret is accepted iff
the residual std on the ORIGINAL (unreduced) samples is < 2*sigma, the same
rule as SecretCheck.match_secret in src/salsa.

Usage (toy grid example):
    python3 src/em_wrap/main.py \
        --secret_dir ../testn80_grid/binary_secrets_h6_12 \
        --hamming 6 --secret_seed 0
"""

import argparse
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def mlwei_transform(A, n, q, shift):
    """Decompress the repo's compressed-circulant RLWE rows (task mlwe-i).

    The repo stores one row per RLWE sample; the actual (a, b) pair used for
    training is a fixed negacyclic rotation of that row. This mirrors
    MLWEiDataset.transform in src/salsa/train/envs/datasets.py (k=1 case):
    flip the coefficient order, then rotate left by `shift` negating the
    wrapped-around entries.
    """
    A_flipped = np.flip(A.reshape(len(A), 1, n), axis=2)
    A_rotated = np.concatenate(
        (A_flipped[..., shift:], -A_flipped[..., :shift]), axis=-1
    ) % q
    return A_rotated.reshape(len(A), n)


def center(x, q):
    """Map values from [0, q) to the symmetric range (-q/2, q/2].

    Reduced entries are small SIGNED numbers stored mod q, so e.g. q-1 really
    means -1. All regression happens on centered values.
    """
    x = x % q
    return np.where(x > q // 2, x - q, x).astype(np.float64)


def load_data(secret_dir, h, secret_seed, n, q, shift):
    """Load (train A, train b, orig A, orig b) for one (h, secret_seed) pair.

    Layout produced by generate_A_b.py:
      <parent>/train_A.npy        reduced samples, shared by all secrets
      <parent>/orig_A.npy         original (unreduced) samples, for verification
      <secret_dir>/train_b_{h}_{seed}.npy   b per secret (one column per rotation)
      <secret_dir>/orig_b_{h}_{seed}.npy
    """
    parent = os.path.dirname(os.path.normpath(secret_dir))
    train_A = np.load(os.path.join(parent, "train_A.npy"))
    orig_A = np.load(os.path.join(parent, "orig_A.npy"))
    train_b = np.load(os.path.join(secret_dir, f"train_b_{h}_{secret_seed}.npy"))
    orig_b = np.load(os.path.join(secret_dir, f"orig_b_{h}_{secret_seed}.npy"))

    # The b files store one column per circulant rotation; column ~shift
    # (= -(shift+1)) is the one that pairs with the transformed A rows.
    # Same convention as MLWEiDataset.transform.
    X = center(mlwei_transform(train_A, n, q, shift), q)
    y = center(train_b[:, ~shift], q)
    verify_A = mlwei_transform(orig_A, n, q, shift)
    verify_b = orig_b[:, ~shift]
    return X, y, verify_A, verify_b


# ---------------------------------------------------------------------------
# Verification (never reads the true secret)
# ---------------------------------------------------------------------------

def residual_std(A, b, guess, q):
    """Std of (A@guess - b) mod q, centered.

    For the true secret this is ~sigma (the LWE noise, ~3 here);
    for a wrong secret it is ~q/sqrt(12) (~32 here, uniform residuals).
    """
    r = (A @ guess - b) % q
    r = np.where(r > q // 2, r - q, r)
    return r.std()


def passes_check(A, b, guess, q, sigma):
    """Accept iff residual std < 2*sigma (SecretCheck.match_secret rule)."""
    return residual_std(A, b, guess, q) < 2 * sigma


# ---------------------------------------------------------------------------
# The EM loop
# ---------------------------------------------------------------------------

def em_fit(X, y, q, n_iters=60, tukey_c=4.685, init=None, rng=None):
    """Run the EM / mod-aware IRLS loop. Returns the real-valued estimate.

    X: centered reduced A matrix, shape (num_samples, n)
    y: centered reduced b vector, shape (num_samples,)
    init: starting estimate of the secret (real vector). None = plain OLS.
    """
    if init is None:
        # Cold start: ordinary least squares, ignoring wraps entirely.
        # Garbage in absolute terms, but sometimes close enough to the truth
        # that the first E-step guesses enough wraps correctly.
        s_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        s_hat = init.astype(np.float64).copy()

    identity_reg = 1e-9 * np.eye(X.shape[1])  # tiny ridge for numerical safety

    for _ in range(n_iters):
        # ----- E-step -----
        # Residual on the circle: how far is each sample from the current
        # fit, AFTER removing whole multiples of q? Centered to (-q/2, q/2].
        r = (y - X @ s_hat) % q
        r = np.where(r > q // 2, r - q, r)

        # Unwrap: shift each label onto the line implied by the current fit.
        # Equivalent to y + k_hat*q with k_hat = round((X@s_hat - y)/q).
        y_unwrapped = X @ s_hat + r

        # ----- Tukey biweight weights -----
        # Scale = robust spread of the residuals (median absolute deviation).
        # Samples with |residual| > c*scale get weight 0 (their wrap guess is
        # not trusted); the rest get smoothly decreasing weights.
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1.0)
        u = r / (tukey_c * scale)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)

        # ----- M-step -----
        # Weighted least squares on the unwrapped labels.
        Xw = X * w[:, None]
        s_hat = np.linalg.solve(Xw.T @ X + identity_reg, Xw.T @ y_unwrapped)

    return s_hat


# ---------------------------------------------------------------------------
# Turning the real-valued estimate into binary secret candidates
# ---------------------------------------------------------------------------

def binary_candidates(s_hat, max_ones=20):
    """Round the real estimate to binary guesses, several ways.

    Yields: a 0.5-threshold guess, then "top-m coordinates = 1" for
    m = 1..max_ones. Each candidate is checked by the caller; checking is
    cheap, so we don't try to be clever about which rounding is right.
    """
    n = len(s_hat)
    yield (s_hat > 0.5).astype(np.int64)
    order = np.argsort(-s_hat)  # coordinates sorted by estimated value, descending
    for m in range(1, max_ones + 1):
        g = np.zeros(n, dtype=np.int64)
        g[order[:m]] = 1
        yield g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--secret_dir", required=True,
                   help="directory with train_b/orig_b/secret files "
                        "(train_A.npy/orig_A.npy live in its parent)")
    p.add_argument("--hamming", type=int, required=True, help="secret hamming weight h")
    p.add_argument("--secret_seed", type=int, required=True)
    p.add_argument("--N", type=int, default=80, help="LWE dimension n")
    p.add_argument("--Q", type=int, default=113, help="modulus q")
    p.add_argument("--A_shift", type=int, default=42,
                   help="circulant rotation used by the mlwe-i pairing")
    p.add_argument("--sigma", type=float, default=3.0, help="LWE noise std")
    p.add_argument("--n_iters", type=int, default=60, help="EM iterations")
    args = p.parse_args()

    t0 = time.time()
    X, y, verify_A, verify_b = load_data(
        args.secret_dir, args.hamming, args.secret_seed, args.N, args.Q, args.A_shift
    )

    # Fit, then test every rounding of the result against the verifier.
    s_hat = em_fit(X, y, args.Q, n_iters=args.n_iters)

    matched, best_std, best_guess = False, float("inf"), None
    for guess in binary_candidates(s_hat):
        std = residual_std(verify_A, verify_b, guess, args.Q)
        if std < best_std:
            best_std, best_guess = std, guess
        if std < 2 * args.sigma:
            matched = True
            break

    wall = time.time() - t0
    result = {
        "matched": bool(matched),
        "best_resid_std": round(float(best_std), 2),
        "wall_sec": round(wall, 1),
        "h": args.hamming,
        "secret_seed": args.secret_seed,
        "recovered_support": sorted(np.where(best_guess == 1)[0].tolist()) if matched else None,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
