# Aux-modulus recipe, pinned to the paper

Source: "Learning Large-Scale Modular Addition with an Auxiliary Modulus"
(arXiv 2605.07648, local copy `papers/aux_modulus_2605.07648.pdf`).
Every claim below cites the paper so the implementation can be audited
line-by-line against it. Where our setting forces a deviation, it is listed
in the Deviations table — nothing is silently changed.

## Their method, exactly

1. **Label switch, not a second head** (Eq. 2, Section 4): for each training
   sample, with probability r the target label is f_Kq(x) (the task answer
   mod Kq), and with probability 1−r it is f_q(x) (the normal answer mod q).
   One output head. Inputs are NEVER changed — they stay in {0,...,q−1} —
   which is the paper's whole point (no covariate shift).

2. **Two implementations** (Section 4.3):
   - *Approach 1, token embedding*: extend the output vocabulary from q to
     Kq tokens. Cross-entropy on whichever label was selected. At inference,
     use only the first q logits.
   - *Approach 2, dual angular embedding*: embed each scalar on TWO circles,
     (cos 2πx/q, sin 2πx/q, cos 2πx/Kq, sin 2πx/Kq); the read-out layer
     produces 4 dims; MSE angular loss; at inference use only the first two
     dims (the mod-q circle).

3. **Aux label construction** (Section 5.1, Dataset): they reconstruct the
   un-wrapped sum as c·q + f_q(x), where c is the known quotient (= wrap
   count), then reduce mod Kq. Their quotient c is exactly our wrap count k,
   so our precomputed bK = (u mod Kq) in make_synthetic_labels.py is the
   same quantity, computed once instead of dynamically.

4. **Hyperparameters** (Section 5.1 + appendix):
   - Practical grid: K ∈ {4..9}, r ∈ {0.1..0.4}. High-accuracy region in
     their broader sweep: K ∈ {5..7}, r ∈ {0.4, 0.5}.
   - Selection heuristic: pick (K, r) so the expected-wraps ratio of Eq. (4)
     is ≈ 1.0.
   - Their training setup (for reference, not binding on us): encoder-only
     transformer, 4 layers, 4 heads, emb 256, FFN 2048, no dropout, no
     positional embeddings, AdamW (0.9, 0.999), weight decay 0.1, lr 3e-5,
     linear decay, warmup ratio 0.05, 10 epochs, batch 250.

5. **Evaluation** (Section 5.1): match accuracy; for angular output, decode
   angle → scale by q/2π → round to nearest integer.

## Deviations we must make (each one deliberate and visible)

| # | Theirs | Ours | Why |
|---|--------|------|-----|
| 1 | Task: modular addition of N input scalars | LWE: b = a·s + e, secret unknown at attack time | That is the experiment: does the trick transfer to LWE? (The paper has no LWE experiments.) |
| 2 | Noiseless | LWE error e, sigma = 3 | Inherent to the task. |
| 3 | Their 4-layer encoder | The repo's SALSA encoder, unchanged except embedding/readout | We test the trick inside the attack pipeline we compare against, or the comparison means nothing. |
| 4 | Aux labels generated dynamically per batch | Precomputed in the .npz (same formula, c·q + f_q then mod Kq) | Simplicity; mathematically identical, verified by the k sanity check in make_synthetic_labels.py. |
| 5 | Uniform random inputs | The victim's real BKZ-reduced rows | Deliberate: pretraining must match the attack input distribution. |

## What arm C therefore is (implementation contract)

- SALSA's angular-embedding path, extended to the dual circle (Approach 2),
  because the repo's recipe already uses angular embeddings; readout 4-dim.
- Per-sample Bernoulli(r) label switch between b and bK, done in the data
  loader. No second head, no second loss term.
- Start at K = 4 (in their practical grid; our labels are generated with
  K = 4) and r = 0.3; sweep toward K ∈ {5..7}, r ∈ {0.4, 0.5} only if the
  first signal is positive.
- Inference/recovery identical to arm A: only the mod-q circle is read.
