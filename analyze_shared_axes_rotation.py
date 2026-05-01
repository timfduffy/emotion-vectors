"""
Is there a single shared low-dim subspace that captures the inter-layer
rotations of emotion vectors? Or does each layer pair need its own
custom rotation axes?

For a chosen set of layer pairs we plot, as a function of k:
    - Per-pair subspace-restricted Procrustes:
        rotation lives in the top-k principal directions of that pair's
        stacked [X_i; X_j]  (different per pair)
    - Shared subspace-restricted Procrustes:
        rotation lives in the top-k principal directions of the FULL stack
        [X_0; X_1; ...; X_{L-1}]  (same for all pairs)

Both schemes have k(k-1)/2 free angles. If the shared curve catches up
to per-pair at small k, the inter-layer rotations live in a universal
low-d subspace. If shared lags far behind, each pair uses its own
custom rotation directions.

Also shows a shuffled-label null per k as a baseline.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_emotion_vectors(vectors_dir: Path):
    with open(vectors_dir / "metadata.json") as f:
        meta = json.load(f)
    emotions = meta["emotions"]
    layers = meta["layers"]
    out = {}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        out[layer] = np.stack([data[e] for e in emotions]).astype(np.float64)
    return out, emotions, layers


def safe_svd(M):
    try:
        return np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        Mp = M + 1e-12 * np.random.default_rng(0).standard_normal(M.shape)
        return np.linalg.svd(Mp, full_matrices=False)


def subspace_restricted_aligned_cos(X_i, X_j, B_k):
    """Restricted Procrustes: rotation acts only within span(B_k),
    identity on the orthogonal complement.

    B_k: (k, d) orthonormal basis defining the subspace.
    Returns: per-emotion cosine similarity of (aligned X_i) vs X_j.
    """
    Xi_proj = X_i @ B_k.T   # (n, k)
    Xj_proj = X_j @ B_k.T   # (n, k)
    M = Xi_proj.T @ Xj_proj # (k, k)
    Us, _, Vt = safe_svd(M)
    R_k = Us @ Vt           # (k, k) orthogonal rotation in subspace

    # X_i_aligned = X_i + Xi_proj @ (R_k @ B_k - B_k)
    #             = (X_i - Xi_proj @ B_k) + Xi_proj @ R_k @ B_k
    X_i_aligned = X_i + Xi_proj @ (R_k @ B_k - B_k)

    a_norms = np.linalg.norm(X_i_aligned, axis=1).clip(1e-12)
    b_norms = np.linalg.norm(X_j, axis=1).clip(1e-12)
    return (X_i_aligned * X_j).sum(axis=1) / (a_norms * b_norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="emotion_vectors_denoised")
    parser.add_argument("--output-dir", default="analysis_output/rotation")
    parser.add_argument(
        "--pairs", default="17-18,13-14,23-24,0-34",
        help="Comma-separated 'a-b' layer pairs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shuffles", type=int, default=3)
    parser.add_argument(
        "--ks", default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,171",
        help="Comma-separated k values to evaluate.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vectors from {args.vectors_dir}...")
    vectors_by_layer, emotions, layers = load_emotion_vectors(Path(args.vectors_dir))
    n, d = vectors_by_layer[layers[0]].shape
    print(f"  {n} concepts in {d}-d, {len(layers)} layers")

    pairs = []
    for spec in args.pairs.split(","):
        a, b = spec.split("-")
        pairs.append((int(a), int(b)))

    ks = [int(k) for k in args.ks.split(",")]
    if max(ks) > n:
        raise SystemExit(f"max k ({max(ks)}) exceeds n ({n})")

    rng = np.random.default_rng(args.seed)

    # ---- Shared basis: top-n right-singular vectors of stacked X ----
    print("\nComputing shared subspace basis (SVD of stacked emotion vectors)...")
    t0 = time.perf_counter()
    A_all = np.vstack([vectors_by_layer[layer] for layer in layers])  # (n*L, d)
    # We only need the right singular vectors (V). Compute via the (d, d) gram matrix
    # since d=1536 < n*L=5985, that's actually slower; np.linalg.svd handles it.
    _, _, Vt_all = np.linalg.svd(A_all, full_matrices=False)
    B_full = Vt_all[: max(ks), :]  # (k_max, d), orthonormal rows
    print(f"  shared basis ready in {time.perf_counter() - t0:.2f}s")

    # ---- Loop over pairs ----
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5), squeeze=False)
    axes = axes[0]

    print("\nRunning subspace-restricted Procrustes per pair...")
    for ax, (la, lb) in zip(axes, pairs):
        if la not in layers or lb not in layers:
            ax.set_title(f"L{la} <-> L{lb} not in metadata"); continue
        X_i = vectors_by_layer[la]; X_j = vectors_by_layer[lb]

        # Pair-specific basis: top-k_max right SVs of stacked [X_i; X_j]
        A_pair = np.vstack([X_i, X_j])
        _, _, Vt_pair = np.linalg.svd(A_pair, full_matrices=False)
        B_pair_full = Vt_pair[: max(ks), :]

        # Raw cosine baseline
        a_n = np.linalg.norm(X_i, axis=1).clip(1e-12)
        b_n = np.linalg.norm(X_j, axis=1).clip(1e-12)
        raw_cos = float(np.median((X_i * X_j).sum(axis=1) / (a_n * b_n)))

        per_pair_curve = []
        shared_curve = []
        shuffled_curve = []
        for k in ks:
            B_pair_k = B_pair_full[:k, :]
            B_shared_k = B_full[:k, :]
            per_pair_curve.append(float(np.median(
                subspace_restricted_aligned_cos(X_i, X_j, B_pair_k))))
            shared_curve.append(float(np.median(
                subspace_restricted_aligned_cos(X_i, X_j, B_shared_k))))

            # Shuffled control: same shared basis, but permute Y's emotion labels
            sh = []
            for _ in range(args.n_shuffles):
                perm = rng.permutation(n)
                X_j_perm = X_j[perm]
                sh.append(float(np.median(
                    subspace_restricted_aligned_cos(X_i, X_j_perm, B_shared_k))))
            shuffled_curve.append(float(np.mean(sh)))

        ax.axhline(raw_cos, color="tab:gray", linestyle=":", label=f"Raw cos = {raw_cos:+.3f}")
        ax.plot(ks, per_pair_curve, marker="o", color="tab:blue",
                label="Per-pair k-d subspace\n(custom axes per pair)")
        ax.plot(ks, shared_curve,   marker="s", color="tab:green",
                label="Shared k-d subspace\n(same axes for all pairs)")
        ax.plot(ks, shuffled_curve, marker="x", color="tab:red", linestyle="--",
                label="Shared + shuffled labels\n(overfitting null)")
        ax.set_xscale("log")
        ax.set_xlabel("k = subspace dimension")
        ax.set_ylabel("Median per-emotion cosine sim")
        ax.set_title(f"Layer {la} -> Layer {lb}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
        print(f"  pair ({la}, {lb}) done")

    fig.suptitle("Per-pair vs shared rotation subspace — is there a universal low-d rotation?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "shared_axes_rotation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'shared_axes_rotation.png'}")


if __name__ == "__main__":
    main()
