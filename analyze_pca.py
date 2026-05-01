"""
PCA analysis of emotion vectors.

- Layer sweep 17-27 with mean-centered PCA
- Layer 25: comparison of three preprocessing modes (raw / mean_center / l2_then_center)
- Correlates PC scores with NRC VAD valence and arousal to test whether
  the leading PCs correspond to the classic valence/arousal axes.

Outputs go to analysis_output/pca/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


# Fallback word forms for emotions not directly in NRC VAD.
# Picked so the substitute is the closest single-word approximation.
VAD_FALLBACKS = {
    "at ease": "ease",
    "dumbstruck": "speechless",
    "empathetic": "empathic",
    "energized": "energetic",
    "exuberant": "enthusiastic",
    "grief-stricken": "grief",
    "insulted": "insult",
    "invigorated": "invigorate",
    "on edge": "edgy",
    "self-confident": "confident",
    "self-conscious": "conscious",
    "self-critical": "critical",
    "stimulated": "stimulate",
    "unnerved": "nervous",
    "worn out": "weary",
}


def load_emotion_vectors(vectors_dir: Path):
    with open(vectors_dir / "metadata.json") as f:
        metadata = json.load(f)
    emotions = metadata["emotions"]
    layers = metadata["layers"]
    vectors_by_layer = {}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        vectors_by_layer[layer] = np.stack([data[e] for e in emotions]).astype(np.float32)
    return vectors_by_layer, emotions, layers


def load_vad(vad_path: Path, emotions: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (valence, arousal, info) aligned to `emotions`."""
    vad = {}
    with open(vad_path, encoding="utf-8") as f:
        for line in f:
            w, v, a, d = line.rstrip("\n").split("\t")
            vad[w.lower()] = (float(v), float(a), float(d))

    valence = np.zeros(len(emotions))
    arousal = np.zeros(len(emotions))
    used_key = []
    for i, emo in enumerate(emotions):
        key = emo.lower()
        if key in vad:
            v, a, _ = vad[key]
            used_key.append(key)
        elif emo in VAD_FALLBACKS and VAD_FALLBACKS[emo] in vad:
            v, a, _ = vad[VAD_FALLBACKS[emo]]
            used_key.append(VAD_FALLBACKS[emo])
        else:
            raise KeyError(f"No VAD entry or fallback for {emo!r}")
        valence[i] = v
        arousal[i] = a
    info = {
        "n_direct": sum(1 for e, k in zip(emotions, used_key) if e.lower() == k),
        "n_fallback": sum(1 for e, k in zip(emotions, used_key) if e.lower() != k),
        "fallbacks": {e: k for e, k in zip(emotions, used_key) if e.lower() != k},
    }
    return valence, arousal, info


def preprocess(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return X
    if mode == "mean_center":
        return X - X.mean(axis=0, keepdims=True)
    if mode == "l2_center":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        Xn = X / norms
        return Xn - Xn.mean(axis=0, keepdims=True)
    raise ValueError(mode)


def fit_pca(X: np.ndarray, n_components: int = 10) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(X)
    return scores, pca


def align_sign(scores: np.ndarray, valence: np.ndarray, arousal: np.ndarray) -> np.ndarray:
    """Flip each PC so the correlation with its dominant external axis is positive.

    For each PC we pick whichever of (valence, arousal) it correlates more
    strongly with (by absolute value), and flip the sign so that correlation
    is positive. This keeps signs consistent across layers when, e.g., PC2
    is the arousal axis but happens to have near-zero valence correlation
    (which would otherwise leave its sign essentially random).
    """
    out = scores.copy()
    for k in range(scores.shape[1]):
        rv = np.corrcoef(scores[:, k], valence)[0, 1]
        ra = np.corrcoef(scores[:, k], arousal)[0, 1]
        ref_r = rv if abs(rv) >= abs(ra) else ra
        if ref_r < 0:
            out[:, k] = -scores[:, k]
    return out


def _place_labels_non_overlapping(ax, x, y, labels, fontsize, priority=None,
                                  pad_px: float = 1.0):
    """Greedy non-overlap label placement.

    Sort labels by `priority` (default: distance from origin, highest first)
    and add them one at a time, skipping any whose bounding box would
    overlap a previously placed label. Returns the number of labels actually
    placed.
    """
    if priority is None:
        priority = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
    order = np.argsort(-np.asarray(priority))

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed_bboxes = []
    placed = 0
    for idx in order:
        txt = ax.annotate(labels[idx], (x[idx], y[idx]), fontsize=fontsize,
                          alpha=0.95, ha="center", va="bottom",
                          xytext=(0, 3), textcoords="offset points")
        bbox = txt.get_window_extent(renderer=renderer).expanded(
            1.0 + pad_px / 100, 1.0 + pad_px / 100
        )
        if any(bbox.overlaps(b) for b in placed_bboxes):
            txt.remove()
        else:
            placed_bboxes.append(bbox)
            placed += 1
    return placed


def scatter_labeled(
    ax,
    x,
    y,
    labels,
    color_values=None,
    color_label="",
    title="",
    cmap="RdBu_r",
    marker_size: int = 22,
    label_fontsize: int = 6,
    title_fontsize: int = 11,
    colorbar_fontsize: int = 9,
    prune_overlapping_labels: bool = False,
):
    if color_values is not None:
        norm = TwoSlopeNorm(vmin=float(np.min(color_values)),
                            vcenter=float(np.median(color_values)),
                            vmax=float(np.max(color_values)))
        sc = ax.scatter(x, y, c=color_values, cmap=cmap, norm=norm,
                        s=marker_size, alpha=0.85, edgecolors="none")
        cb = ax.figure.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(color_label, fontsize=colorbar_fontsize)
        cb.ax.tick_params(labelsize=colorbar_fontsize)
    else:
        ax.scatter(x, y, s=marker_size, alpha=0.7, c="steelblue")

    ax.set_title(title, fontsize=title_fontsize)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xticks([])
    ax.set_yticks([])

    if prune_overlapping_labels:
        n_placed = _place_labels_non_overlapping(ax, x, y, labels, label_fontsize)
        return n_placed
    else:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), fontsize=label_fontsize, alpha=0.9,
                        ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
        return len(labels)


def top_loadings(scores: np.ndarray, emotions: list[str], k: int = 10):
    """For each PC return the top-k positive and bottom-k negative emotions."""
    out = []
    for pc in range(scores.shape[1]):
        order = np.argsort(scores[:, pc])
        bot = [(emotions[i], float(scores[i, pc])) for i in order[:k]]
        top = [(emotions[i], float(scores[i, pc])) for i in order[-k:][::-1]]
        out.append({"pc": pc + 1, "top": top, "bottom": bot})
    return out


def correlate_pcs(scores, valence, arousal, n_pcs=5):
    rows = []
    for pc in range(n_pcs):
        rv_p, _ = pearsonr(scores[:, pc], valence)
        ra_p, _ = pearsonr(scores[:, pc], arousal)
        rv_s, _ = spearmanr(scores[:, pc], valence)
        ra_s, _ = spearmanr(scores[:, pc], arousal)
        rows.append({
            "pc": pc + 1,
            "valence_pearson": float(rv_p),
            "arousal_pearson": float(ra_p),
            "valence_spearman": float(rv_s),
            "arousal_spearman": float(ra_s),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="emotion_vectors_denoised")
    parser.add_argument("--vad-path", default="data/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt")
    parser.add_argument("--output-dir", default="analysis_output/pca")
    parser.add_argument("--layer-start", type=int, default=None,
                        help="Lowest layer to analyze. Default: all layers in metadata.")
    parser.add_argument("--layer-end", type=int, default=None,
                        help="Highest layer to analyze. Default: all layers in metadata.")
    parser.add_argument("--compare-layer", type=int, default=25)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scatters").mkdir(exist_ok=True)

    print("Loading vectors…")
    vectors_by_layer, emotions, all_layers = load_emotion_vectors(Path(args.vectors_dir))
    print(f"  {len(emotions)} emotions, layers {all_layers[0]}-{all_layers[-1]}")

    print("Loading NRC VAD…")
    valence, arousal, vad_info = load_vad(Path(args.vad_path), emotions)
    print(f"  Direct: {vad_info['n_direct']}, Fallback: {vad_info['n_fallback']}")

    lo = args.layer_start if args.layer_start is not None else min(all_layers)
    hi = args.layer_end if args.layer_end is not None else max(all_layers)
    layers = [l for l in all_layers if lo <= l <= hi]
    print(f"  Will analyze {len(layers)} layers: {layers[0]}..{layers[-1]}")

    # ---- Layer sweep, mean-centered ----
    sweep_results = {}
    for layer in layers:
        print(f"\nLayer {layer} (mean_center)…")
        X = preprocess(vectors_by_layer[layer], "mean_center")
        scores, pca = fit_pca(X, args.n_components)
        scores = align_sign(scores, valence, arousal)  # PC1 oriented + with valence

        corrs = correlate_pcs(scores, valence, arousal, n_pcs=5)
        loadings = top_loadings(scores, emotions, k=args.top_k)
        sweep_results[layer] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "correlations": corrs,
            "top_loadings": loadings,
        }

        # PC1 vs PC2 colored by valence and by arousal (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(28, 14))
        scatter_labeled(
            axes[0], scores[:, 0], scores[:, 1], emotions,
            color_values=valence, color_label="Valence (NRC VAD)",
            title=f"Layer {layer} - PC1 vs PC2 (color=valence)\n"
                  f"r(PC1,V)={corrs[0]['valence_pearson']:+.2f}  "
                  f"r(PC2,V)={corrs[1]['valence_pearson']:+.2f}",
        )
        scatter_labeled(
            axes[1], scores[:, 0], scores[:, 1], emotions,
            color_values=arousal, color_label="Arousal (NRC VAD)",
            title=f"Layer {layer} - PC1 vs PC2 (color=arousal)\n"
                  f"r(PC1,A)={corrs[0]['arousal_pearson']:+.2f}  "
                  f"r(PC2,A)={corrs[1]['arousal_pearson']:+.2f}",
            cmap="viridis",
        )
        for ax in axes:
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        fig.suptitle(f"Emotion PCA - Layer {layer} (denoised, mean-centered)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "scatters" / f"pca_pc1_pc2_layer_{layer}.png",
                    dpi=140, bbox_inches="tight")
        plt.close(fig)

        # PC1 vs PC3 (valence-colored)
        fig, ax = plt.subplots(figsize=(14, 14))
        scatter_labeled(
            ax, scores[:, 0], scores[:, 2], emotions,
            color_values=valence, color_label="Valence",
            title=f"Layer {layer} - PC1 vs PC3 (color=valence)\n"
                  f"r(PC3,V)={corrs[2]['valence_pearson']:+.2f}  "
                  f"r(PC3,A)={corrs[2]['arousal_pearson']:+.2f}",
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        fig.tight_layout()
        fig.savefig(out_dir / "scatters" / f"pca_pc1_pc3_layer_{layer}.png",
                    dpi=140, bbox_inches="tight")
        plt.close(fig)

        # Standalone valence and arousal scatters for the comparison layer.
        # Larger fonts/markers; non-overlapping greedy label placement
        # (most-extreme points labeled first, others left as dots).
        if layer == args.compare_layer:
            for color_vals, color_name, cmap_name in [
                (valence, "valence", "RdBu_r"),
                (arousal, "arousal", "viridis"),
            ]:
                fig, ax = plt.subplots(figsize=(11, 11))
                n_placed = scatter_labeled(
                    ax, scores[:, 0], scores[:, 1], emotions,
                    color_values=color_vals,
                    color_label=f"{color_name.capitalize()} (NRC VAD)",
                    title=f"Layer {layer} - PC1 vs PC2 (color={color_name})\n"
                          f"r(PC1,{color_name[0].upper()})="
                          f"{corrs[0][f'{color_name}_pearson']:+.2f}  "
                          f"r(PC2,{color_name[0].upper()})="
                          f"{corrs[1][f'{color_name}_pearson']:+.2f}",
                    cmap=cmap_name,
                    marker_size=70,
                    label_fontsize=11,
                    title_fontsize=18,
                    colorbar_fontsize=14,
                    prune_overlapping_labels=True,
                )
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                              fontsize=14)
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                              fontsize=14)
                print(f"  [{color_name}] placed {n_placed}/{len(emotions)} labels")
                fig.tight_layout()
                fig.savefig(out_dir / "scatters"
                            / f"pca_pc1_pc2_layer_{layer}_{color_name}.png",
                            dpi=140, bbox_inches="tight")
                plt.close(fig)

    # ---- Cross-layer summary plots ----

    # Explained variance per PC across layers
    fig, ax = plt.subplots(figsize=(11, 6))
    n_show = min(args.n_components, 10)
    for pc in range(n_show):
        ys = [sweep_results[l]["explained_variance_ratio"][pc] for l in layers]
        ax.plot(layers, ys, marker="o", label=f"PC{pc+1}")
    ax.set_xlabel("Layer"); ax.set_ylabel("Explained variance ratio")
    ax.set_title("Per-PC explained variance across layers (mean-centered, denoised)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "explained_variance_per_pc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Cumulative variance
    fig, ax = plt.subplots(figsize=(11, 6))
    for layer in layers:
        ax.plot(range(1, args.n_components + 1),
                sweep_results[layer]["cumulative_variance"],
                marker="o", label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("# PCs"); ax.set_ylabel("Cumulative variance")
    ax.set_title("Cumulative variance vs. PCs (mean-centered, denoised)")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_variance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Heatmap: |corr(PC, valence)| and |corr(PC, arousal)| over layers x PCs
    n_pcs_corr = 5
    val_mat = np.zeros((len(layers), n_pcs_corr))
    aro_mat = np.zeros((len(layers), n_pcs_corr))
    for i, layer in enumerate(layers):
        for j, c in enumerate(sweep_results[layer]["correlations"][:n_pcs_corr]):
            val_mat[i, j] = c["valence_pearson"]
            aro_mat[i, j] = c["arousal_pearson"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, name in zip(axes, [val_mat, aro_mat], ["Valence", "Arousal"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_pcs_corr))
        ax.set_xticklabels([f"PC{k+1}" for k in range(n_pcs_corr)])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("PC"); ax.set_ylabel("Layer")
        ax.set_title(f"Pearson r(PC, {name})")
        for i in range(len(layers)):
            for j in range(n_pcs_corr):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(mat[i, j]) < 0.6 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("PC correlations with NRC VAD valence/arousal (sign-aligned to valence)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "pc_vad_correlation_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Focused line plot: PC1 and PC2 correlations with V and A across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, pc_idx, pc_name in [(axes[0], 0, "PC1"), (axes[1], 1, "PC2")]:
        rv = [sweep_results[l]["correlations"][pc_idx]["valence_pearson"] for l in layers]
        ra = [sweep_results[l]["correlations"][pc_idx]["arousal_pearson"] for l in layers]
        ax.plot(layers, rv, marker="o", label="r(.,Valence)", color="tab:red")
        ax.plot(layers, ra, marker="o", label="r(.,Arousal)", color="tab:blue")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Layer")
        ax.set_title(f"{pc_name} correlation with VAD")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Pearson r")
    fig.suptitle("How PC1 and PC2 align with valence/arousal at each layer",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "pc12_vad_correlation_lines.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Layer 25: preprocessing comparison ----
    cmp_layer = args.compare_layer
    print(f"\nPreprocessing comparison on layer {cmp_layer}…")
    X_raw = vectors_by_layer[cmp_layer]
    modes = ["raw", "mean_center", "l2_center"]
    cmp_results = {}

    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    for col, mode in enumerate(modes):
        Xp = preprocess(X_raw, mode)
        scores, pca = fit_pca(Xp, args.n_components)
        scores = align_sign(scores, valence, arousal)
        corrs = correlate_pcs(scores, valence, arousal, n_pcs=5)
        cmp_results[mode] = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "correlations": corrs,
            "top_loadings": top_loadings(scores, emotions, k=args.top_k),
        }

        # Top row: color by valence
        scatter_labeled(
            axes[0, col], scores[:, 0], scores[:, 1], emotions,
            color_values=valence, color_label="Valence",
            title=f"{mode}\nr(PC1,V)={corrs[0]['valence_pearson']:+.2f}  "
                  f"r(PC2,V)={corrs[1]['valence_pearson']:+.2f}",
        )
        axes[0, col].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[0, col].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        # Bottom row: color by arousal
        scatter_labeled(
            axes[1, col], scores[:, 0], scores[:, 1], emotions,
            color_values=arousal, color_label="Arousal",
            title=f"r(PC1,A)={corrs[0]['arousal_pearson']:+.2f}  "
                  f"r(PC2,A)={corrs[1]['arousal_pearson']:+.2f}",
            cmap="viridis",
        )
        axes[1, col].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[1, col].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.suptitle(f"Layer {cmp_layer} - PCA preprocessing comparison (denoised)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"preprocessing_comparison_layer_{cmp_layer}.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Save summary JSON + console table ----
    summary = {
        "vectors_dir": str(args.vectors_dir),
        "vad_info": vad_info,
        "layers": layers,
        "n_components": args.n_components,
        "sweep_mean_center": sweep_results,
        "preprocessing_comparison": {
            "layer": cmp_layer,
            "modes": cmp_results,
        },
    }
    with open(out_dir / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console table
    print("\n" + "=" * 78)
    print(f"{'Layer':>5}  {'PC1 var':>7}  {'r(PC1,V)':>9}  {'r(PC1,A)':>9}  "
          f"{'r(PC2,V)':>9}  {'r(PC2,A)':>9}  {'r(PC3,V)':>9}  {'r(PC3,A)':>9}")
    print("-" * 78)
    for layer in layers:
        r = sweep_results[layer]["correlations"]
        ev = sweep_results[layer]["explained_variance_ratio"][0]
        print(f"{layer:>5}  {ev:>7.1%}  "
              f"{r[0]['valence_pearson']:>+9.2f}  {r[0]['arousal_pearson']:>+9.2f}  "
              f"{r[1]['valence_pearson']:>+9.2f}  {r[1]['arousal_pearson']:>+9.2f}  "
              f"{r[2]['valence_pearson']:>+9.2f}  {r[2]['arousal_pearson']:>+9.2f}")

    print(f"\nLayer {cmp_layer} preprocessing comparison:")
    for mode, info in cmp_results.items():
        r = info["correlations"]
        ev1, ev2 = info["explained_variance_ratio"][:2]
        print(f"  {mode:>12s}: PC1 {ev1:.1%} (rV={r[0]['valence_pearson']:+.2f},"
              f" rA={r[0]['arousal_pearson']:+.2f})   "
              f"PC2 {ev2:.1%} (rV={r[1]['valence_pearson']:+.2f},"
              f" rA={r[1]['arousal_pearson']:+.2f})")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
