# Dead-core-seeded EM: a cheap attack on reduced LWE, and a rule for when it works

Mayank Kumar (UCF) — November 2026. Status: research note for discussion, toy scale.
All code and raw results in this branch (`em-wrap-attack`); every number below is
reproducible from the scripts referenced inline.

## 1. Setting

Toy benchmark setting from the LWE-benchmarking suite: n = 80, q = 113, RLWE,
binary secrets, Hamming weight h ∈ {6, 8, 10, 12}, 5 secrets per h (the suite's own
provided secrets), σ = 3, preprocessed (BKZ-reduced) data as shipped with the suite,
reduction quality ρ = 0.783, ~50k reduced samples. We do not touch preprocessing.

**Verification.** A guess ŝ is accepted iff the centered residual std of
(A·ŝ − b) mod q on the *original* (unreduced) samples is < 2σ. Calibration: the true
secret scores ≈ 3.1; the best of 3000 wrong candidates, including 1-bit near misses,
scores 29.6. Zero false positives observed; an accepted guess was bit-for-bit exact
in every case we audited (all successes below). Original samples are never seen by
any fit; note they are derived from the same LWE instance by reduction, not
independently drawn.

## 2. Scoreboard (same 20 secrets, 20-minute budget per run unless noted)

| method | h=6 | h=8 | h=10 | h=12 | total | cost per secret |
|---|---|---|---|---|---|---|
| B1: SALSA transformer (published recipe) | 3/15* | 0/15 | 0/15 | — | 1 secret | 7–20 min GPU |
| B2: robust regression, real line (Tukey IRLS) | 0/5 | 0/5 | 0/5 | 0/5 | 0/20 | ~1 s CPU |
| B3: Cool&Cruel (repo implementation) | 5/5 | 4/5 | 4/5 | 5/5 | 18/20 | 1 s–10 min GPU |
| ours: dead-core-seeded EM, weight ≤ 2 | 4/5 | 1/5 | 1/5 | 0/5 | 6/20 | 8 s–47 min CPU |
| **B3 ∪ ours** | 5/5 | 4/5 | 5/5 | 5/5 | **19/20** | — |

*B1: 15 runs = 5 secrets × 3 model seeds; only secret 0 recovered (by all 3 seeds).

The point is not the 6/20. It is that (i) the two attacks fail on *different*
secrets, (ii) which secrets each will crack is predictable in advance from public
data, and (iii) our failures are budget failures of a 466-seed cap, with a
measured per-secret price for extending it. CC failed h8_s3 and h10_s2 within its
budget; we recover h10_s2 in 7 EM runs / 8 seconds on one CPU core. Only h8_s3
resists both attacks.

## 3. The attack

**ε-profile.** For each column j of the reduced matrix A, compute
ε_j = |mean_i exp(2πi·A_ij/q)| — from A alone, no secret involved. ε_j measures how
concentrated column j's values are relative to q (1 = tiny values, 0 = uniform).
On this dataset the 80 columns split into ~38 strong (ε ≈ 0.68), ~12 lukewarm
(0.02 < ε ≤ 0.5), ~30 dead (ε ≤ 0.02, the sampling noise floor).

**Why ε matters.** A wrong secret bit on a high-ε column leaves a statistical
trace that a solver can find and fix. A wrong bit on a dead column makes the
residual indistinguishable from uniform noise — no statistic points to the
culprit, so that bit must be guessed.

**Algorithm** (`src/em_wrap/deadcore_em.py`): enumerate patterns of ≤ 2 ones over
the ~30 dead columns in ascending weight (ε-ordered); from each seed run a
mod-aware EM — E-step: unwrap b using the current estimate's residuals mod q;
M-step: Tukey-weighted least squares — then round and verify. At most 466 EM runs
per secret, ~8 s per run, one CPU core, no GPU, no training.

Per-secret results: `deadcore_full_grid.jsonl`. Successes: h6_s0 (1 run), h6_s1
(331), h6_s2 (277), h6_s3 (63), h8_s2 (422), h10_s2 (7).

## 4. Why it works: measured basin structure

- **Flip basin** (`basin_full_grid.txt`, all 20 secrets × flips 0–8 × 20 trials):
  starting EM from the true secret with k random bits flipped, recovery is ~100%
  at k ≤ 2, ~70% at 3, ~50% at 4, ~0 by 7–8 — uniform across all 20 secrets.
- **Asymmetry** (`basin_matrix_h8s3_h6s1.csv`): false ones added hurt more than
  true ones missing at equal Hamming distance.
- **What predicts failure is not distance but blur.** EM recovered one secret from
  Hamming distance 9 (all distance on strong columns) and failed others at
  distance 2 (on dead columns).
- **Hardness score.** Across the 20 grid secrets, the best single predictor of
  success we found is Σ(1 − ε_j) over the secret's support (total "blur mass" the
  solver must absorb beyond the seed): a single threshold separates 19/20
  outcomes. Caveat: this score and threshold were selected after seeing the
  outcomes — treat as a fitted hypothesis pending out-of-sample validation.

## 5. Does it scale? Kyber-setting tests (n=256, k=2 — secret dimension 512)

Scripts: `src/em_wrap/n256_*.py`; data: the suite's own preprocessed Kyber releases.

**log q = 12 (q = 3329, ρ = 0.88, binomial secrets h 6–10, 99k samples):**
- The ε-split exists and is sharp: 128 strong (ε ≈ 0.88) vs 384 dead.
- Mechanism transfers: given a seed correct on the dead columns, EM heals all 128
  strong coordinates (one stall fixed by a cheap verified one-coordinate sweep) —
  exact secret recovered in 14 s. Control: seeding strong instead of dead fails.
- But slack is zero: one wrong dead entry is fatal (tested at 14.5k AND 99k
  samples — more data does not help; the wall is informational).
- Secrets carry 5–9 entries on the 384 dead columns → enumeration infeasible.
- Mitigation measured: the MLWE rotation gives 256 equivalent views; dead support
  varies across views (3–6 for the secret tested). Picking the best view cuts the
  exponent but C(384,3)·4³ ≈ 6·10⁸ runs still needs cluster-scale parallelism.

**log q = 28 (q ≈ 1.8·10⁸, ρ = 0.67, h 13–19):**
- Better global reduction, but *no* strong band: max ε = 0.25. With everything
  only mildly concentrated relative to the huge modulus, healing dies entirely —
  even a single missing entry never heals (EM at truth does stay put).

**Feasibility rule** (both halves computable from A alone, before attacking):
the attack works iff (1) a high-ε band exists — that is what EM can heal — and
(2) the secret's dead support is small enough to enumerate. The toy setting has
both; each Kyber setting has exactly one. Note ρ does not capture (1): log q=28
has better ρ but a worse ε-spectrum. The per-column ε-spectrum, which current
benchmarks do not track, appears to be the right lens.

## 6. Negative results (all documented in this branch)

- Plain real-line Tukey IRLS: 0/20 (wraps leave ~12% usable inliers).
- Five generic EM initializations (cold, annealed, cool-column OLS start, greedy
  pursuit, 200 random restarts): fail on 19/20 secrets (`race_results.csv`).
- Probing SALSA models' positional embeddings for secret leakage: the one
  successful model shows real signal (AUC 1.0, permutation p = 0.003–0.005);
  failed models show nothing significant (all p ≥ 0.036 before a 45-test
  correction) — consistent with learning here being a phase transition rather
  than gradual accumulation.
- Shortest-rows initialization: correlation ≈ 0.

## 7. Caveats

- Toy scale (n = 80, q = 113) for the headline result; Kyber scale is
  mechanism-verified only (seeded diagnostics, not an end-to-end attack).
- All comparisons are budget-conditioned (CC: 20 min; ours: ≤ 466 runs). CC with
  a larger budget or more samples may crack its two failures.
- CC is not the strongest known attack at every setting: classical lattice
  attacks (Karenin et al. 2025) beat it at several benchmark points. We have not
  yet run that classical baseline (B4) and make no claims against it.
- `best_resid_std` figures in race results are selected minima over candidates,
  not typical values.
- The hardness score in §4 is post-hoc; validation on fresh secrets pending.

## 8. Questions we would ask

1. Does the per-column ε-spectrum of your larger preprocessed datasets (n = 256+
   at various BKZ budgets) show a high-ε band, and how does its width trade
   against reduction effort? That trade-off decides whether this attack class has
   a regime at scale.
2. Your preprocessing choices directly shape the ε-spectrum. Would tracking ε
   during reduction (rather than only the global ρ) change which benchmark cells
   are considered attackable?
3. Is the zero-slack phenomenon (a single wrong dead entry being unrecoverable)
   consistent with what you see in transformer training — i.e., models only
   learning secrets whose dead support is empty?
