# EM attack findings: toy win, Kyber-scale theory (2026-11-11)

All terms defined inline. Everything below is measured, not speculated.
Code: fork branch `em-wrap-attack`; n=256 scripts in `code/n256_*.py`.

## Vocabulary (used throughout)

- **ε (epsilon) of a column**: ε_j = |mean over samples of exp(2πi·a_j/q)|,
  computed from A alone (no secret). It measures how concentrated the
  column's values are relative to the modulus q. ε≈1: values tiny compared
  to q. ε≈0: values look uniform mod q.
- **Cruel column** (terminology of Nolte et al.; our scripts call it "dead"): ε ≤ 0.1. If the guess is wrong on a cruel column, the
  error looks like uniform noise — no statistic can point to the culprit.
- **Cool column**: high ε. Errors there leave a visible trace, so the
  solver can find and fix them.
- **Cruel support**: how many of the secret's nonzero entries sit on cruel
  columns. These are the entries that must be guessed by enumeration.
- **Healing**: EM started with some entries wrong converges to the truth
  anyway. **Slack**: how many *cruel* entries the seed may get wrong and
  still recover (via healing + repair sweep).
- **Repair sweep**: after EM, try changing each single coordinate to each
  possible value and accept if the exact verifier (residual test on
  original samples) passes. Cheap: ~2000 checks.

## The three measured settings

| | n=80 toy | Kyber n=256 k=2 log q=12 | Kyber n=256 k=2 log q=28 |
|---|---|---|---|
| q | 113 | 3329 | 2^28 (179M) |
| secret dim | 80 (binary) | 512 (binomial) | 512 (binomial) |
| reduction ρ | 0.783 | 0.88 | 0.67 |
| samples used | 50k | 99k | 99k |
| cool band | 38 cols, ε≈0.68 | 128 cols, ε≈0.88 | **none** (max ε=0.25) |
| cruel columns | 30/80 (37%) | 384/512 (75%) | 219/512 (43%) |
| cruel support | 0–2 | 5–9 (shift-dependent, min≈3) | 3–11 |
| EM healing | works | works (one stall, repair fixes) | **fails** (even 1 missing entry) |
| slack on cruel entries | 0 (same as below) | 0 (tested at 14.5k AND 99k samples) | n/a |
| attack verdict | **WINS** (≤466 runs/secret) | enumeration infeasible (C(384,5)·4^5) | mechanism cruel |

Key Kyber-scale positives, both at log q=12:
- Exact secret recovery in 14 s given a correct cruel seed (EM healed all
  128 cool coordinates; one stalled coordinate fixed by repair sweep).
- Control test: seeding cool instead of cruel fails — the cruel/cool
  asymmetry is real at scale, not a toy artifact.

## The feasibility rule (the actual finding)

The attack works iff BOTH hold, and both are computable from A alone,
before attacking:

1. **A high-ε band exists.** EM can only heal columns whose values are
   small relative to q. Global reduction quality ρ does NOT capture this:
   log q=28 has better ρ (0.67) but zero high-ε columns, because its huge
   modulus leaves even "reduced" values at ~0.2q. The per-column ε
   spectrum is the right lens, and nobody in the benchmark papers tracks it.
2. **The cruel support is enumerable.** Cost = C(#cruel cols, cruel support)
   × (values per entry). Toy: C(30,≤2) fine. Kyber log q=12: C(384,5+)
   hopeless.

Corollary: "more data" does not help — slack on cruel entries stayed zero
going from 14.5k to 99k samples. The wall is informational, not statistical.

## Tweaks measured or assessed

- **Shift search (real, measured)**: the MLWE rotation gives 256
  equivalent views; the cruel set rotates while the secret stays fixed, so
  cruel support varies per view — measured 3–6 for one h=6 secret. Same
  trick CC uses. Reduces the exponent (5→3 here) but C(384,3)·4³ ≈ 6e8 EM
  runs still infeasible on one machine.
- **Repair sweep (real, measured)**: fixes any single wrong coordinate
  after EM for ~free. Extends reach by exactly 1 coordinate, no more
  (2-missing tests all failed).
- **ε-ordered enumeration (used at n=80)**: enumerate likelier cruel
  patterns first. Helps constants, not the exponent.
- **GPU-batched EM (not done)**: EM is dense linear algebra; thousands of
  seeds could run in parallel. Shaves maybe 3 orders of magnitude off
  wall-clock; 6e8 runs would still need a cluster.
- **Better reduction (not ours to run)**: shrinking the cruel set directly
  attacks the exponent. The trade "BKZ effort vs enumeration cost" is the
  interesting open curve; the SALSA authors have the preprocessing
  infrastructure to explore it.

## What to tell the SALSA team

1. n=80: cruel-seeded EM recovers secrets CC fails on, in minutes on
   one CPU (full grid run in progress; 4/4 at time of writing).
2. The ε-spectrum feasibility rule, with the three-setting table above.
3. The constructive question: their preprocessing choices directly shape
   the ε-spectrum. Does tracking ε during reduction (instead of only ρ)
   change which (n, q, h) cells of their benchmark are attackable?

## Caveats that must ship with any claim

- Toy scale for the win; Kyber scale only mechanism-verified (seeded).
- CC comparison is budget-conditioned (CC had 20 min; our runs minutes to
  ~47 min).
- CC is not SOTA overall: classical attacks (Karenin et al.) beat it at
  several settings — B4 baseline still deferred.
- best_resid_std in race results is a selected minimum, not typical.
- Verify data is derived-by-reduction (never seen by the fit), not
  independently drawn.
