#!/usr/bin/env python3
"""
Initialization strategies for the EM-over-wraps attack.

The core EM loop (main.em_fit) converges to the truth only if it starts in
the right "basin". A cold OLS start lands there for some secrets and not for
others. This file implements four ways to find a better starting point.
Checking whether a run worked is nearly free (residual std on the original
samples), so strategies are allowed to produce MANY candidate fits.

Strategies:
  1. anneal      - one run; trust everything at first, tighten gradually
  2. cool_start  - initialize from the low-variance ("cool") columns only
  3. greedy      - build the support one coordinate at a time by circular fit
  4. restarts    - many EM runs from perturbed / subsampled starting points
"""

import numpy as np

from main import em_fit, binary_candidates, residual_std


def centered_mod(r, q):
    """Reduce mod q into the symmetric range (-q/2, q/2]."""
    r = r % q
    return np.where(r > q // 2, r - q, r)


# ---------------------------------------------------------------------------
# Strategy 1: annealing (graduated non-convexity)
# ---------------------------------------------------------------------------

def em_fit_annealed(X, y, q, n_iters=80, tukey_c=4.685):
    """Like em_fit, but the Tukey scale starts huge and shrinks geometrically.

    A huge scale means every sample gets weight ~1, so the early iterations
    are plain least squares with unwrapping. As the scale shrinks, the fit
    gradually stops listening to samples that disagree with it. The hope is
    that the fit is pulled toward the truth before it becomes picky.
    The floor is the usual robust (MAD-based) scale.
    """
    s_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    identity_reg = 1e-9 * np.eye(X.shape[1])

    # Scale schedule: from q (everything trusted) down to the data-driven
    # robust scale, shrinking by a constant factor per iteration.
    scale_start = float(q)
    decay = 0.93  # reaches ~ q*0.93^60 ≈ q/75 ≈ 1.5 by iteration 60

    for it in range(n_iters):
        r = centered_mod(y - X @ s_hat, q)
        y_unwrapped = X @ s_hat + r

        mad_scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, 1.0)
        scale = max(scale_start * decay**it, mad_scale)

        u = r / (tukey_c * scale)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        Xw = X * w[:, None]
        s_hat = np.linalg.solve(Xw.T @ X + identity_reg, Xw.T @ y_unwrapped)

    return s_hat


# ---------------------------------------------------------------------------
# Strategy 2: cool-column warm start
# ---------------------------------------------------------------------------

def cool_start(X, y, q, cool_fraction=0.4):
    """Initialize from the "cool" columns only, then run full EM.

    After lattice reduction, column standard deviations are very uneven
    (the cool&cruel phenomenon / Z-shape). The low-std ("cool") columns
    barely contribute to wraps, so an OLS fit restricted to them is much
    less wrap-contaminated. Cruel coordinates start at 0.
    """
    col_std = X.std(axis=0)
    n = X.shape[1]
    n_cool = max(int(n * cool_fraction), 1)
    cool_idx = np.argsort(col_std)[:n_cool]  # lowest-variance columns

    init = np.zeros(n)
    init[cool_idx] = np.linalg.lstsq(X[:, cool_idx], y, rcond=None)[0]
    return em_fit(X, y, q, init=init)


# ---------------------------------------------------------------------------
# Strategy 3: greedy forward pursuit on circular concentration
# ---------------------------------------------------------------------------

def greedy_pursuit(X, y, q, max_ones=20):
    """Build a binary secret one coordinate at a time.

    Score of a candidate secret: how concentrated the residuals
    (y - X@s) mod q are around 0, measured by mean cos(2*pi*r/q).
    1.0 = perfectly concentrated (correct secret, no noise);
    0.0 = uniform residuals (wrong secret). The score sees THROUGH wraps
    because it lives on the circle.

    At each step, try adding each unused coordinate, keep the one that
    raises concentration most. Yields the estimate after every step, so the
    caller can EM-refine and verify each prefix (h is not known exactly).
    """
    n = X.shape[1]
    s = np.zeros(n)
    residual = y.copy()  # = y - X@s, maintained incrementally
    used = np.zeros(n, dtype=bool)

    for _ in range(max_ones):
        best_j, best_score = -1, -np.inf
        for j in range(n):
            if used[j]:
                continue
            r_j = centered_mod(residual - X[:, j], q)
            score = np.mean(np.cos(2 * np.pi * r_j / q))
            if score > best_score:
                best_score, best_j = score, j
        s[best_j] = 1.0
        used[best_j] = True
        residual = residual - X[:, best_j]
        yield s.copy()


# ---------------------------------------------------------------------------
# Strategy 4: multi-restart from perturbed / subsampled fits
# ---------------------------------------------------------------------------

def restart_inits(X, y, q, n_restarts, seed=0):
    """Yield many different starting points for EM.

    Half: the OLS solution plus Gaussian noise (explores around the cold
    start). Half: OLS fit on a random 20% of the rows (different subsets are
    contaminated by different wrapped samples, so the fits scatter widely).
    """
    rng = np.random.RandomState(seed)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    n_rows = X.shape[0]

    for i in range(n_restarts):
        if i % 2 == 0:
            yield ols + rng.normal(0, 0.5, size=len(ols))
        else:
            rows = rng.choice(n_rows, size=n_rows // 5, replace=False)
            yield np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
