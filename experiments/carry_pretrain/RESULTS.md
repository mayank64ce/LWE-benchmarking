# Carry-pretraining results (toy setting: n=80, q=113, real reduced rows)

Two experiments. Logs for both are in this directory; code is in `src/carry_pretrain/`.

## Experiment 1: mechanism grid (single secret, wrap labels available)

Question: if the model is told the wrap count k = floor((A.s + e)/q) during
training, do secrets that are unlearnable from b alone become learnable?

Setup: one invented h=8 secret per run, labels computed on the victim's real
reduced rows, 20k steps, eval on held-out rows. Metric: within-2sigma accuracy
(prediction within +-6 of true b). Chance = 11.5%, noise ceiling ~ 97%.

| arm | labels | s0 | s1 | s2 |
|---|---|---|---|---|
| A (plain) | b only | 11.0 | 94.3 | 10.6 |
| B (carry head, ours) | b + k classification | 91.8 | 86.7 | 84.8 |
| C (aux modulus, Kikuchi) | label switch b mod q / b mod 4q | 95.3 | 93.1 | 92.7 |
| Cp (ablation) | b only, arm-C architecture | 12.5 | 94.4 | 9.7 |
| D (hybrid B+C) | label switch + carry head | 86.4 | 80.4 | 87.1 |

Findings: wrap supervision (B, C, D) makes all three secrets learnable; plain
labels (A, Cp) learn only secret 1, the one with the smallest cruel support.
Cp at chance on s0/s2 shows the effect comes from the labels, not the dual
architecture. The hybrid is not additive.

## Experiment 2: transfer to a real victim (Fresca-style pretrain + finetune)

Question: does wrap-supervised pretraining on invented secrets transfer to a
REAL h=8 secret where no wrap labels exist?

Setup: pretrain one model per arm on 32 invented h=8 secrets with secret-ID
tokens (Fresca ePrint 2024/150 Sec 5.2), 40k steps, rows [0, 25000). Fine-tune
on each real victim (train_b_8_{0,1,2}.npy) with b-only labels, a fresh victim
token, rows [25000, end), 20k steps, eval on last 2000 rows.

Pretraining sanity (within-2sigma on invented secret 0): A 12% (chance),
B 62% (still climbing), C 91%.

Fine-tune on real victims (within-2sigma, chance 11.5%, +-1.4% noise band):

| victim | cold | preA (Fresca) | preB (carry) | preC (aux mod) |
|---|---|---|---|---|
| h8 s0 | 11.5 | 11.4 | 9.7 | 12.4 |
| h8 s1 | 11.6 | 11.7 | 10.9 | 11.6 |
| h8 s2 | 11.8 | 11.3 | 10.5 | 10.8 |

Finding: NULL. Every arm is at chance on every victim. Wrap-supervised
pretraining does not make an unlearnable real secret learnable at this scale.

Caveats: 32 secrets / ~10M examples vs Fresca's 150 secrets / 528M examples;
shared reduced rows (row-split) instead of a separately generated pretraining
set. Also note Fresca's own claim is narrower than what we tested: they show
pretraining cuts the samples needed for secrets that are ALREADY recoverable
from random init (1.7M -> 409K), and their metric averages only over secrets
recovered by all checkpoints. They never claim pretraining rescues infeasible
secrets, which is what this experiment asked.

Untested follow-up: the speedup version — fine-tune on a LEARNABLE held-out
secret and compare steps-to-recovery across arms.
