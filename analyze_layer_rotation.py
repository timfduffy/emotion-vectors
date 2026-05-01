"""
How much of the cross-layer change in emotion-vector geometry is a rigid
rotation, vs something more structural?

For every pair of layers (L_i, L_j) we fit:
    1. Identity (no transform) - baseline raw cosine
    2. Orthogonal Procrustes rotation R that minimizes ||X @ R - Y||
    3. Same as (2) but with a uniform scalar scale s (R, s minimize
       ||s * X @ R - Y||)

We measure success as the median per-emotion cosine similarity:
    sim_k = cos((transformed X)[k], Y[k]).

Critically, we also run the SAME Procrustes fit with Y's rows randomly
permuted, breaking the emotion-identity correspondence. If a rotation can
align random-correspondence data just as well, the alignment is overfitting
on the high dimensionality. If it can't, then preserving emotion identity
matters and the rotational story is real.

Outputs:
  - Heatmaps of rotation-aligned similarity vs raw similarity
  - Heatmap of "rotation lift" = aligned - raw
  - Heatmap of shuffled-label baseline (the overfitting null)
  - Heatmap of rotation lift CORRECTED for the shuffled baseline
  - Summary plots of alignment vs layer distance
"""

import argparse
import json
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


def per_row_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    b = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return (a * b).sum(axis=1)


def precompute_layer(X: np.ndarray):
    """QR of X.T plus per-row norms. Cached once per layer."""
    _, R = np.linalg.qr(X.T)         # X.T = Q R; we never need Q for the diag trick
    norms = np.linalg.norm(X, axis=1)
    return R, norms


def aligned_diag_cosines(R1, norms1, R2, norms2):
    """Per-row cosine similarity of (X @ R) and Y, where R is the orthogonal
    Procrustes rotation X -> Y, using only the (n, n) factors.

    Derivation:
      X.T = Q1 R1, Y.T = Q2 R2 (rank-n QR; X, Y are (n, d), n << d).
      Procrustes SVD: R1 R2.T = Us S Vt   (in (n, n))
      Then  (X @ R) @ Y.T  =  R1.T @ Us @ Vt @ R2     (n, n)
      Diagonal of that gives the per-row dot products.
      Norms are preserved by orthogonal R, so ||X @ R[k]|| = ||X[k]||.
    """
    M = R1 @ R2.T
    try:
        Us, _, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        Mp = M + 1e-12 * np.random.default_rng(0).standard_normal(M.shape)
        Us, _, Vt = np.linalg.svd(Mp, full_matrices=False)
    inner = R1.T @ Us @ Vt @ R2          # (n, n)
    diag = np.diag(inner)
    return diag / (norms1 * norms2 + 1e-12)


def median_aligned_cached(R1, norms1, R2, norms2) -> float:
    return float(np.median(aligned_diag_cosines(R1, norms1, R2, norms2)))


def median_raw(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.median(per_row_cosine(X, Y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="emotion_vectors_denoised")
    parser.add_argument("--output-dir", default="analysis_output/rotation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for label shuffles")
    parser.add_argument("--n-shuffles", type=int, default=5,
                        help="How many label shuffles to average for the overfitting null")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vectors from {args.vectors_dir}...")
    vectors_by_layer, emotions, layers = load_emotion_vectors(Path(args.vectors_dir))
    n, d = vectors_by_layer[layers[0]].shape
    print(f"  {n} concepts in {d}-d, {len(layers)} layers")

    L = len(layers)
    raw = np.zeros((L, L))
    rot = np.zeros((L, L))
    shuf = np.zeros((L, L))   # mean over n_shuffles
    rng = np.random.default_rng(args.seed)

    # Pre-compute QR + norms once per layer (35 QRs total, ~0.2s)
    print("\nPre-computing QR per layer...")
    R_per_layer = {}
    norms_per_layer = {}
    for layer in layers:
        R_per_layer[layer], norms_per_layer[layer] = precompute_layer(vectors_by_layer[layer])

    print(f"\nRunning Procrustes for all {L*(L+1)//2} layer pairs "
          f"(plus {args.n_shuffles} shuffled controls each), all in (n, n) subspace...")
    import time
    t0 = time.perf_counter()
    for i, li in enumerate(layers):
        R1 = R_per_layer[li]; n1 = norms_per_layer[li]
        Xi = vectors_by_layer[li]
        for j, lj in enumerate(layers):
            if j < i:
                raw[i, j] = raw[j, i]; rot[i, j] = rot[j, i]; shuf[i, j] = shuf[j, i]
                continue
            Xj = vectors_by_layer[lj]
            R2 = R_per_layer[lj]; n2 = norms_per_layer[lj]

            raw[i, j] = median_raw(Xi, Xj)
            rot[i, j] = median_aligned_cached(R1, n1, R2, n2)

            # Shuffled-label control: permute Y's rows, refit Procrustes,
            # measure cos vs the (still-permuted) Y. We can stay in (n, n)
            # space: permuting Y's rows = permuting columns of R2 by same perm.
            shuffles = []
            for _ in range(args.n_shuffles):
                perm = rng.permutation(n)
                R2_perm = R2[:, perm]
                n2_perm = n2[perm]
                shuffles.append(median_aligned_cached(R1, n1, R2_perm, n2_perm))
            shuf[i, j] = float(np.mean(shuffles))
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Lift = improvement gained from the rotation
    rot_lift = rot - raw
    # Corrected lift = how much more rotation buys vs the overfitting null
    corrected_lift = rot - shuf

    # ---- Save numerical data ----
    np.savez(
        out_dir / "rotation_metrics.npz",
        layers=np.array(layers),
        raw=raw, rot=rot, shuf=shuf,
    )

    # ---- Heatmaps ----
    def plot_grid(matrices: list[tuple[str, np.ndarray, str, float, float]], out_path: Path):
        cols = len(matrices)
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5.5), squeeze=False)
        axes = axes[0]
        for ax, (title, M, cmap, vmin, vmax) in zip(axes, matrices):
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Layer"); ax.set_ylabel("Layer")
            step = max(1, L // 12)
            ticks = list(range(0, L, step))
            ax.set_xticks(ticks); ax.set_xticklabels([layers[t] for t in ticks])
            ax.set_yticks(ticks); ax.set_yticklabels([layers[t] for t in ticks])
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    plot_grid(
        [
            ("Raw median per-emotion cosine\n(no transform)",  raw, "RdBu_r", -1, 1),
            ("After orthogonal Procrustes rotation\n(median per-emotion cosine)", rot, "RdBu_r", -1, 1),
            ("Shuffled-label control\n(rotation overfitting null)", shuf, "RdBu_r", -1, 1),
        ],
        out_dir / "rotation_alignment_heatmaps.png",
    )

    plot_grid(
        [
            ("Rotation lift = aligned - raw\n(gain from finding optimal rotation)", rot_lift, "viridis", 0, 1),
            ("Overfitting null (shuffled labels)\n(should be near zero if rotation is real)", shuf, "viridis", 0, max(0.01, shuf.max())),
            ("Corrected lift = aligned - shuffled\n(gain BEYOND chance rotation)", corrected_lift, "viridis", 0, 1),
        ],
        out_dir / "rotation_lift_breakdown.png",
    )

    # ---- Summary: alignment vs layer distance ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    distances = list(range(1, L))
    raw_by_dist = []; rot_by_dist = []; shuf_by_dist = []
    for k in distances:
        ridx = np.arange(L - k)
        raw_by_dist.append(np.median(raw[ridx, ridx + k]))
        rot_by_dist.append(np.median(rot[ridx, ridx + k]))
        shuf_by_dist.append(np.median(shuf[ridx, ridx + k]))
    ax.plot(distances, raw_by_dist,  marker="o", label="Raw (no transform)",   color="tab:gray")
    ax.plot(distances, rot_by_dist,  marker="o", label="After rotation",        color="tab:blue")
    ax.plot(distances, shuf_by_dist, marker="o", label="Shuffled-label rotation\n(overfitting null)", color="tab:red", linestyle="--")
    ax.set_xlabel("Layer distance |i - j|")
    ax.set_ylabel("Median per-emotion cosine similarity")
    ax.set_title("Alignment vs layer distance — does pure rotation explain the change?")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_vs_layer_distance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'alignment_vs_layer_distance.png'}")

    # ---- Console summary ----
    print("\n=== Summary ===")
    print(f"Effective Procrustes DoF: ~n*(n-1)/2 = {n*(n-1)//2:,} "
          f"(constraints: n*d = {n*d:,}; ratio constraint/DoF = {n*d / (n*(n-1)/2):.1f}x)")
    print()
    iu = np.triu_indices(L, k=1)
    print(f"Across all {len(iu[0])} off-diagonal pairs:")
    print(f"  raw cosine        : median={np.median(raw[iu]):+.3f}  "
          f"min={raw[iu].min():+.3f}  max={raw[iu].max():+.3f}")
    print(f"  rotation-aligned  : median={np.median(rot[iu]):+.3f}  "
          f"min={rot[iu].min():+.3f}  max={rot[iu].max():+.3f}")
    print(f"  shuffled control  : median={np.median(shuf[iu]):+.3f}  "
          f"min={shuf[iu].min():+.3f}  max={shuf[iu].max():+.3f}")
    print()
    pct_explained_by_rotation = (rot - raw) / (1.0 - raw + 1e-9)
    print(f"Fraction of the gap to perfect alignment closed by rotation:")
    print(f"  median over pairs: {float(np.median(pct_explained_by_rotation[iu])):.3f}")
    print(f"  (subtracting shuffled baseline first):")
    pct_corrected = (rot - shuf) / (1.0 - shuf + 1e-9)
    print(f"  median over pairs: {float(np.median(pct_corrected[iu])):.3f}")


if __name__ == "__main__":
    main()
