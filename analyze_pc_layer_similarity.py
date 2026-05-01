"""
Cross-layer cosine similarity of steering direction vectors.

For each concept (PC, raw emotion, ...), build an (n_layers, hidden_dim)
matrix and compute the (n_layers, n_layers) cosine similarity matrix.
Visualize as a grid of heatmaps.

Supports any vectors_dir with a metadata.json. Common uses:

  # PC1 + PC2 only
  python analyze_pc_layer_similarity.py --vectors-dir pca_vectors \
      --concepts pc1_valence,pc2_arousal --output-suffix _pc12

  # Random 12 emotions
  python analyze_pc_layer_similarity.py --vectors-dir emotion_vectors_denoised \
      --sample-n 12 --seed 42 --output-suffix _sample12

  # Median (across all concepts) layer-similarity heatmap
  python analyze_pc_layer_similarity.py --vectors-dir emotion_vectors_denoised \
      --median --no-grid --output-suffix _emotions_median
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_pca_vectors(vectors_dir: Path):
    with open(vectors_dir / "metadata.json") as f:
        metadata = json.load(f)
    pc_names = metadata["emotions"]
    layers = metadata["layers"]

    # vectors_by_pc[name] = (n_layers, hidden_dim) array
    vectors_by_pc = {name: [] for name in pc_names}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        for name in pc_names:
            vectors_by_pc[name].append(data[name].astype(np.float32))
    for name in pc_names:
        vectors_by_pc[name] = np.stack(vectors_by_pc[name])  # (L, D)
    return vectors_by_pc, pc_names, layers


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of rows of X: (N, D) -> (N, N)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = X / norms
    return Xn @ Xn.T


def plot_pc_layer_similarity(vectors_by_pc, pc_names, layers, out_path: Path):
    """Grid of (n_pcs) heatmaps, each (n_layers, n_layers)."""
    n = len(pc_names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), squeeze=False)

    for i, name in enumerate(pc_names):
        ax = axes[i // cols][i % cols]
        sim = cosine_sim_matrix(vectors_by_pc[name])
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{name}: layer x layer cosine similarity", fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

        # Tick every few layers for readability
        step = max(1, len(layers) // 12)
        ticks = list(range(0, len(layers), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([layers[t] for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([layers[t] for t in ticks])
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Cross-layer cosine similarity for each PC (sign-aligned vectors)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cross_pc_at_layer(vectors_by_pc, pc_names, layer_idx_in_array: int,
                           layers, out_path: Path):
    """Cross-PC similarity at a single representative layer."""
    layer = layers[layer_idx_in_array]
    rows = []
    for name in pc_names:
        rows.append(vectors_by_pc[name][layer_idx_in_array])
    M = np.stack(rows)
    sim = cosine_sim_matrix(M)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pc_names))); ax.set_xticklabels(pc_names, rotation=30, ha="right")
    ax.set_yticks(range(len(pc_names))); ax.set_yticklabels(pc_names)
    for i in range(len(pc_names)):
        for j in range(len(pc_names)):
            ax.text(j, i, f"{sim[i,j]:+.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(sim[i,j]) < 0.6 else "white")
    ax.set_title(f"Cross-PC cosine similarity at layer {layer}\n"
                 f"(should be ~identity since PCs are orthogonal at fit time)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_median_layer_similarity(vectors_by_concept, concept_names, layers, out_path: Path):
    """Median (across concepts) cosine similarity layer x layer.

    Each cell (i, j) = median over concepts of cos_sim(concept at L_i, concept at L_j).
    Right panel shows the 25-75 IQR as a measure of how much concepts disagree
    on the cross-layer similarity at each (i, j).
    """
    n_concepts = len(concept_names)
    L = len(layers)
    stack = np.zeros((n_concepts, L, L), dtype=np.float32)
    for k, name in enumerate(concept_names):
        stack[k] = cosine_sim_matrix(vectors_by_concept[name])
    median_sim = np.median(stack, axis=0)
    iqr = np.percentile(stack, 75, axis=0) - np.percentile(stack, 25, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(median_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[0].set_title(f"Median cross-layer cosine similarity\n"
                      f"(over {n_concepts} concepts)", fontsize=11)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Layer")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(iqr, cmap="viridis", aspect="equal")
    axes[1].set_title("IQR (75th - 25th pct) across concepts\n"
                      "= disagreement between concepts", fontsize=11)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Layer")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    step = max(1, L // 12)
    ticks = list(range(0, L, step))
    for ax in axes:
        ax.set_xticks(ticks); ax.set_xticklabels([layers[t] for t in ticks])
        ax.set_yticks(ticks); ax.set_yticklabels([layers[t] for t in ticks])

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary(vectors_by_pc, pc_names, layers, out_path: Path):
    """Summary line plot: similarity of each PC at each layer with the
    same PC at a 'central' reference layer (default layer 25 if present).

    Makes it easy to see how stable each PC is around the mid-late zone.
    """
    ref_layer = 25 if 25 in layers else layers[len(layers) // 2]
    ref_idx = layers.index(ref_layer)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in pc_names:
        M = vectors_by_pc[name]
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        Mn = M / norms
        ref = Mn[ref_idx]
        sims = Mn @ ref
        ax.plot(layers, sims, marker="o", label=name)

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(ref_layer, color="gray", lw=0.5, ls="--", alpha=0.5,
               label=f"reference (L{ref_layer})")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Cosine similarity vs same PC at layer {ref_layer}")
    ax.set_title("Per-PC stability across layers (vs. mid-late reference layer)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="pca_vectors")
    parser.add_argument("--output-dir", default="analysis_output/pca")
    parser.add_argument("--reference-layer", type=int, default=25,
                        help="Reference layer for cross-concept sanity heatmap and stability plot")
    parser.add_argument("--concepts", default=None,
                        help="Comma-separated concept names to plot. Default: all.")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="If set, randomly sample N concepts from the source.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for --sample-n random sampling.")
    parser.add_argument("--median", action="store_true",
                        help="Also produce a median-across-all-concepts layer-similarity heatmap.")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip the per-concept grid heatmap.")
    parser.add_argument("--no-stability", action="store_true",
                        help="Skip the stability line plot.")
    parser.add_argument("--no-cross", action="store_true",
                        help="Skip the cross-concept similarity heatmap at reference layer.")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to output filenames.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix

    print(f"Loading vectors from {args.vectors_dir}...")
    vectors_by_concept, all_names, layers = load_pca_vectors(Path(args.vectors_dir))
    print(f"  {len(all_names)} concepts, {len(layers)} layers ({layers[0]}..{layers[-1]})")

    # Decide which concepts to use for per-concept plots
    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
        missing = [c for c in concepts if c not in vectors_by_concept]
        if missing:
            raise SystemExit(f"Unknown concepts: {missing}")
    elif args.sample_n is not None:
        if args.sample_n > len(all_names):
            raise SystemExit(f"--sample-n ({args.sample_n}) > {len(all_names)} concepts available")
        rng = random.Random(args.seed)
        concepts = sorted(rng.sample(all_names, args.sample_n))
        print(f"  Sampled {args.sample_n} concepts (seed {args.seed}): {concepts}")
    else:
        concepts = all_names

    subset = {c: vectors_by_concept[c] for c in concepts}

    # Per-concept layer x layer heatmap grid
    if not args.no_grid and len(concepts) > 0:
        plot_pc_layer_similarity(
            subset, concepts, layers,
            out_dir / f"pc_layer_similarity_grid{suffix}.png",
        )

    # Median across all concepts
    if args.median:
        plot_median_layer_similarity(
            vectors_by_concept, all_names, layers,
            out_dir / f"layer_similarity_median{suffix}.png",
        )

    # Cross-concept at reference layer
    if not args.no_cross and len(concepts) >= 2:
        ref_idx = layers.index(args.reference_layer) if args.reference_layer in layers else len(layers) // 2
        plot_cross_pc_at_layer(
            subset, concepts, ref_idx, layers,
            out_dir / f"pc_cross_similarity_layer_{layers[ref_idx]}{suffix}.png",
        )

    # Stability line plot
    if not args.no_stability and len(concepts) >= 1:
        plot_summary(
            subset, concepts, layers,
            out_dir / f"pc_stability_vs_reference_layer{suffix}.png",
        )

    # Print quick text summary
    if concepts:
        print("\nQuick layer-stability summary (cosine sim of consecutive layers):")
        for name in concepts:
            M = subset[name]
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            Mn = M / norms
            consec = [float(Mn[i] @ Mn[i + 1]) for i in range(len(layers) - 1)]
            print(f"  {name:25s}  median={np.median(consec):+.3f}  "
                  f"min={min(consec):+.3f}  max={max(consec):+.3f}")


if __name__ == "__main__":
    main()
