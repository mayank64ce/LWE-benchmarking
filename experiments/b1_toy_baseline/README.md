# Baseline kill-test: toy setting (n=80, q=113, binary secrets, RLWE)

Pre-registered protocol: see `ideas/baseline-protocol.md` in the research notes.
Grid: h ∈ {6, 8, 10} × 5 fresh secrets × 3 model seeds (B1), 20-min budget per run,
single RTX 3090. Data: benchmark toy preprocessed set, reduction quality ρ = 0.783.

## Files

- `results.csv` — B1: published SALSA recipe (encoder-only, angular embedding,
  dx-distinguisher). Columns: h, secret_seed, model_seed, matched, wall_sec, success_epoch.
- `b1_baseline.png` — recovery rate per h with per-secret spread; time-to-recovery.
- `b2_results.csv` — B2: Tukey-biweight IRLS on centered reduced data (real-line
  residuals). Deterministic, one fit per (h, secret).
- `results_modaware_em.csv` — exploratory: same IRLS but residuals computed mod q
  (centered) with unwrap-refit each iteration. Not a registered baseline.

## Results

| method | h=6 | h=8 | h=10 | cost/secret |
|---|---|---|---|---|
| B1 transformer | 3/15 (only secret 0, all 3 seeds) | 0/15 | 0/15 | ~7–20 min GPU |
| B2 Tukey (real line) | 0/5 | 0/5 | 0/5 | ~1 s CPU |
| mod-aware IRLS | 1/5 (secret 0) | 0/5 | 0/5 | ~2 s CPU |

Verification for all methods: residual std on original (unreduced) samples < 2σ,
never reading the true secret (same rule as `SecretCheck.match_secret`).

Notes: B1 and the mod-aware fit succeed and fail on exactly the same (h, secret)
pairs, suggesting per-secret recoverability is a property of the reduced data
(clean-sample structure), not the attack. B2 fails everywhere because wrapped
labels cost ±q on the real line, leaving ~12% usable inliers in 80 dimensions.
