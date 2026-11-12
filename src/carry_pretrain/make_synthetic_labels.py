"""Step 1 of the carry-pretraining experiment: synthetic secrets + labels.

The idea (ideas/representation.md, idea 2 + the aux-modulus paper):
we cannot know the wrap count k on the victim's data, but we can invent
secrets ourselves. For an invented secret we know everything, so we can
compute every label type a pretraining arm needs:

  arm A (baseline):    b  = (X @ s + e) mod q        (what SALSA already uses)
  arm B (carry head):  k  = how many times X @ s + e wrapped around q
  arm C (aux modulus): bK = (X @ s + e) mod (K*q)    (Kikuchi et al. recipe)

Crucially the INPUTS are the victim's real reduced rows (train_A from the
toy grid), so the pretraining input distribution matches the attack exactly.
Only the secrets (and therefore the labels) are invented.

X is the transformed matrix the SALSA trainer actually sees. We copy the
exact transform from MLWEiDataset.transform (src/salsa/train/envs/datasets.py)
so there is no convention mismatch: flip each block, negacyclic-shift by
A_shift, reduce mod q. The b column that pairs with this X is b[:, ~A_shift];
here we just produce y for that one shift directly.

Output: one .npz per run in --out_dir with
  secrets  (n, num_secrets)  invented binary secrets, one per column
  b        (num_secrets, num_samples)  labels for arm A
  k        (num_secrets, num_samples)  labels for arm B
  bK       (num_secrets, num_samples)  labels for arm C
Row order of the labels matches train_A row order, so a loader only needs
train_A plus this file.

Run:
  python3 make_synthetic_labels.py --grid_dir ../../../testn80_grid \
      --out_dir ../../../carry_pretrain_data --num_secrets 64 --hamming 8
"""

import argparse
import os

import numpy as np


def transform_A(A, n, q, shift):
    """Exact copy of the MLWEiDataset.transform convention for the A side
    (rlwe k=1): flip the columns, then negacyclic-shift left by `shift`
    (wrapped elements get negated), then reduce mod q."""
    A = np.flip(A.reshape((len(A), 1, n)), axis=2)
    A = np.concatenate((A[..., shift:], -A[..., :shift]), axis=-1) % q
    return A.reshape((len(A), n)).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", required=True,
                   help="directory with the victim grid's train_A.npy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_secrets", type=int, default=64,
                   help="how many secrets to invent")
    p.add_argument("--hamming", type=int, default=8,
                   help="Hamming weight of each invented binary secret")
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--Q", type=int, default=113)
    p.add_argument("--sigma", type=float, default=3.0,
                   help="LWE error std, same as the benchmark setting")
    p.add_argument("--A_shift", type=int, default=42,
                   help="the shift the SALSA trainer uses (its default)")
    p.add_argument("--K", type=int, default=4,
                   help="aux modulus factor: arm C labels are mod K*q")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    A = np.load(os.path.join(args.grid_dir, "train_A.npy"))
    assert A.shape[1] == args.N
    X = transform_A(A, args.N, args.Q, args.A_shift)

    # Invent binary secrets: exactly `hamming` ones in random positions.
    # Stored one secret per column, same layout as the repo's secret.npy.
    secrets = np.zeros((args.N, args.num_secrets), dtype=np.int64)
    for j in range(args.num_secrets):
        ones = rng.choice(args.N, size=args.hamming, replace=False)
        secrets[ones, j] = 1

    num_samples = len(X)
    b = np.zeros((args.num_secrets, num_samples), dtype=np.int64)
    k = np.zeros((args.num_secrets, num_samples), dtype=np.int64)
    bK = np.zeros((args.num_secrets, num_samples), dtype=np.int64)

    for j in range(args.num_secrets):
        # One error vector per secret, fresh noise, same sigma as the victim.
        e = np.rint(rng.normal(0.0, args.sigma, size=num_samples)).astype(np.int64)
        u = X @ secrets[:, j] + e          # the unreduced value: this is what
                                           # we can only know for invented secrets
        b[j] = u % args.Q                  # arm A label (ordinary LWE b)
        k[j] = (u - b[j]) // args.Q        # arm B label (exact wrap count)
        bK[j] = u % (args.K * args.Q)      # arm C label (aux modulus)

        # Sanity check: k must be recoverable from (X, s, b) alone by rounding,
        # because |e| << q. If this ever fails the labels are inconsistent.
        k_check = np.rint((X @ secrets[:, j] - b[j]) / args.Q).astype(np.int64)
        assert np.array_equal(k_check, k[j]), f"k mismatch for secret {j}"

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(
        args.out_dir,
        f"synth_h{args.hamming}_m{args.num_secrets}_K{args.K}_seed{args.seed}.npz")
    np.savez_compressed(out, secrets=secrets, b=b, k=k, bK=bK,
                        A_shift=args.A_shift, Q=args.Q, K=args.K,
                        grid_dir=args.grid_dir)

    # Report the k distribution: this tells us how big the carry-head
    # classification problem is (how many distinct wrap counts exist).
    kmin, kmax = int(k.min()), int(k.max())
    print(f"wrote {out}")
    print(f"samples={num_samples} secrets={args.num_secrets} h={args.hamming}")
    print(f"wrap count k range: [{kmin}, {kmax}] "
          f"({kmax - kmin + 1} classes for the carry head)")
    vals, counts = np.unique(k, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  k={v}: {100.0 * c / k.size:.1f}%")


if __name__ == "__main__":
    main()
