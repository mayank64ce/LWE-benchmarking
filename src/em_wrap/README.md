# em_wrap: dead-core-seeded EM attack on reduced LWE

A CPU-only solver for BKZ-reduced LWE data: enumerate guesses on the columns
the reduction failed to shrink (the "dead" / cruel columns), then recover the
rest of the secret with a mod-aware EM. Full description, scoreboard, and
caveats: [`experiments/em_wrap/WRITEUP.md`](../../experiments/em_wrap/WRITEUP.md).
Raw results live in the same directory.

Terms: ε_j = |mean exp(2πi·a_j/q)| per column of reduced A, computed with no
secret. High ε = column values small relative to q, errors there leave a trace
EM can fix ("strong"). ε at the sampling noise floor = no trace, must guess
("dead"). This is a continuous version of the cruel/cool split of Nolte et al.

## Files

Attack and core machinery:

- `main.py` — shared machinery: data loading (`load_data`), the MLWE
  transform, centering, `em_fit` (E-step: unwrap b mod q against the current
  estimate; M-step: Tukey-weighted least squares), residual verification.
- `deadcore_em.py` — **the attack**. Computes the ε-profile, enumerates seed
  patterns of ≤ `--max_weight` ones over the dead columns (ε-ordered), runs EM
  from each seed, verifies on original (unreduced) samples. ~8 s per EM run,
  one CPU core.

Measurements at n=80 (toy setting):

- `basin_test.py` / `basin_matrix.py` — EM basin of attraction: start from the
  true secret with k bits flipped, measure recovery rate vs k, and the
  false-one vs missing-one asymmetry.
- `run_race.py` + `strategies.py` — race of generic EM initializations (cold,
  annealed, OLS start, greedy pursuit, random restarts). All fail on 19/20
  secrets; this is the negative control for the dead-core seeding.
- `seeded_em.py` — single seeded-EM run for ad-hoc tests.
- `probe_pe.py` — probes trained SALSA transformers' positional embeddings for
  secret leakage (permutation test).

Diagnostics at Kyber scale (n=256, k=2, secret dimension 512):

- `n256_em_test.py` — four-way seeding test at log q=12: EM from truth / dead
  part only / strong part only / cold. Shows the mechanism transfers (dead
  seed heals all strong coordinates) and is directional.
- `n256_slack_test.py` — slack test at log q=12: drop k entries from a correct
  dead seed, count recoveries. Result: zero slack — one wrong dead entry is
  unrecoverable, at 14.5k and 99k samples alike.
- `n256_28_slack_test.py` — same at log q=28. Result: no high-ε band exists
  there, healing fails entirely.

## Running the attack (n=80)

Data layout is what `src/generate/generate_A_b.py` produces: a grid directory
with `train_A.npy` / `orig_A.npy` shared across secrets, and per-secret
`train_b_{h}_{seed}.npy` / `orig_b_{h}_{seed}.npy` under a secrets
subdirectory.

```bash
python3 deadcore_em.py --hamming 10 --secret_seed 2 \
    --secret_dir /path/to/testn80_grid/binary_secrets_h6_12
```

Key flags: `--max_weight` (seed enumeration cap, default 2),
`--include_lukewarm` (enumerate over ε ≤ 0.5 instead of ε ≤ 0.02),
`--dead_eps` / `--lukewarm_eps` (band thresholds).

A guess is accepted only if the centered residual std on the *original*
samples is below 2σ; calibration in the write-up (true secret ≈ 3.1, best of
3000 wrong candidates ≈ 29.6, zero false positives observed).

## What to expect

On the suite's n=80 grid (binary secrets, h 6–12, 5 seeds each): 6/20 secrets
recovered within the 466-run cap, including one Cool&Cruel fails on
(h10_s2: 7 runs, 8 s). Per-secret results:
`experiments/em_wrap/deadcore_full_grid.jsonl`. Whether a given secret will
fall is predictable in advance from the ε-profile — see the feasibility rule
and hardness score in the write-up, including their caveats.
