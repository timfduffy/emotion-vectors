"""
Analyze pairwise cosine similarity between emotion vectors.

Compares original and denoised vectors across layers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import argparse


def load_emotion_vectors(vectors_dir: str) -> tuple[dict, list, list]:
    """Load emotion vectors and metadata."""
    vectors_dir = Path(vectors_dir)

    with open(vectors_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    emotions = metadata["emotions"]
    layers = metadata["layers"]

    # Load vectors for each layer
    vectors_by_layer = {}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        # Stack all emotions into a matrix (n_emotions, hidden_dim)
        vectors_by_layer[layer] = np.stack([data[e] for e in emotions])

    return vectors_by_layer, emotions, layers


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = vectors / norms

    # Compute similarity matrix
    similarity = normalized @ normalized.T
    return similarity


def plot_similarity_heatmap(
    similarity: np.ndarray,
    emotions: list,
    title: str,
    ax: plt.Axes,
    vmin: float = -1,
    vmax: float = 1,
):
    """Plot a similarity heatmap."""
    im = ax.imshow(similarity, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_title(title, fontsize=10)

    # Only show a subset of labels if there are many emotions
    n_emotions = len(emotions)
    if n_emotions > 30:
        # Show every nth label
        step = max(1, n_emotions // 20)
        tick_positions = list(range(0, n_emotions, step))
        tick_labels = [emotions[i] for i in tick_positions]
    else:
        tick_positions = range(n_emotions)
        tick_labels = emotions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=6)

    return im


def plot_comparison(
    orig_sim: np.ndarray,
    denoised_sim: np.ndarray,
    emotions: list,
    layer: int,
    output_path: str,
):
    """Plot side-by-side comparison of original and denoised similarity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    im1 = plot_similarity_heatmap(
        orig_sim, emotions, f"Original (Layer {layer})", axes[0]
    )

    # Denoised
    im2 = plot_similarity_heatmap(
        denoised_sim, emotions, f"Denoised (Layer {layer})", axes[1]
    )

    # Difference
    diff = denoised_sim - orig_sim
    im3 = plot_similarity_heatmap(
        diff, emotions, f"Difference (Denoised - Original)", axes[2],
        vmin=-0.5, vmax=0.5
    )

    # Colorbars
    fig.colorbar(im1, ax=axes[0], shrink=0.8, label='Cosine Similarity')
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label='Cosine Similarity')
    fig.colorbar(im3, ax=axes[2], shrink=0.8, label='Difference')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_comparison(
    orig_vectors: dict,
    denoised_vectors: dict,
    emotions: list,
    layers: list,
    output_dir: str,
):
    """Plot similarity matrices for multiple layers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select representative layers (early, middle, late)
    n_layers = len(layers)
    selected_layers = [
        layers[0],                    # First
        layers[n_layers // 4],        # 1/4
        layers[n_layers // 2],        # Middle
        layers[3 * n_layers // 4],    # 3/4
        layers[-1],                   # Last
    ]

    for layer in selected_layers:
        orig_sim = compute_similarity_matrix(orig_vectors[layer])
        denoised_sim = compute_similarity_matrix(denoised_vectors[layer])

        plot_comparison(
            orig_sim, denoised_sim, emotions, layer,
            output_dir / f"similarity_layer_{layer}.png"
        )

    # Also create a summary plot showing all layers
    plot_layer_summary(orig_vectors, denoised_vectors, emotions, layers, output_dir)


def plot_layer_summary(
    orig_vectors: dict,
    denoised_vectors: dict,
    emotions: list,
    layers: list,
    output_dir: Path,
):
    """Plot summary statistics across layers."""

    # Compute statistics for each layer
    orig_means = []
    orig_stds = []
    denoised_means = []
    denoised_stds = []

    for layer in layers:
        orig_sim = compute_similarity_matrix(orig_vectors[layer])
        denoised_sim = compute_similarity_matrix(denoised_vectors[layer])

        # Get off-diagonal elements (exclude self-similarity)
        mask = ~np.eye(len(emotions), dtype=bool)
        orig_off = orig_sim[mask]
        denoised_off = denoised_sim[mask]

        orig_means.append(np.mean(orig_off))
        orig_stds.append(np.std(orig_off))
        denoised_means.append(np.mean(denoised_off))
        denoised_stds.append(np.std(denoised_off))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean similarity across layers
    axes[0].plot(layers, orig_means, 'b-', label='Original', linewidth=2)
    axes[0].plot(layers, denoised_means, 'r-', label='Denoised', linewidth=2)
    axes[0].fill_between(layers,
                         np.array(orig_means) - np.array(orig_stds),
                         np.array(orig_means) + np.array(orig_stds),
                         alpha=0.2, color='blue')
    axes[0].fill_between(layers,
                         np.array(denoised_means) - np.array(denoised_stds),
                         np.array(denoised_means) + np.array(denoised_stds),
                         alpha=0.2, color='red')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Mean Pairwise Similarity Across Layers')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Std similarity across layers
    axes[1].plot(layers, orig_stds, 'b-', label='Original', linewidth=2)
    axes[1].plot(layers, denoised_stds, 'r-', label='Denoised', linewidth=2)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Std of Cosine Similarity')
    axes[1].set_title('Similarity Spread Across Layers')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "similarity_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'similarity_summary.png'}")


def print_most_similar_pairs(
    vectors: np.ndarray,
    emotions: list,
    n_pairs: int = 20,
    title: str = "",
):
    """Print the most similar emotion pairs."""
    similarity = compute_similarity_matrix(vectors)

    # Get upper triangle indices (excluding diagonal)
    n = len(emotions)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((emotions[i], emotions[j], similarity[i, j]))

    # Sort by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Emotion 1':<20} {'Emotion 2':<20} {'Similarity':>10}")
    print("-" * 60)
    for e1, e2, sim in pairs[:n_pairs]:
        print(f"{e1:<20} {e2:<20} {sim:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze emotion vector similarity")
    parser.add_argument(
        "--original-dir",
        type=str,
        default="emotion_vectors",
        help="Directory with original vectors",
    )
    parser.add_argument(
        "--denoised-dir",
        type=str,
        default="emotion_vectors_denoised",
        help="Directory with denoised vectors",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to analyze (default: multiple representative layers)",
    )

    args = parser.parse_args()

    print("Loading original vectors...")
    orig_vectors, emotions, layers = load_emotion_vectors(args.original_dir)
    print(f"  {len(emotions)} emotions, {len(layers)} layers")

    print("Loading denoised vectors...")
    denoised_vectors, _, _ = load_emotion_vectors(args.denoised_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.layer is not None:
        # Analyze specific layer
        layer = args.layer
        print(f"\nAnalyzing layer {layer}...")

        orig_sim = compute_similarity_matrix(orig_vectors[layer])
        denoised_sim = compute_similarity_matrix(denoised_vectors[layer])

        plot_comparison(
            orig_sim, denoised_sim, emotions, layer,
            output_dir / f"similarity_layer_{layer}.png"
        )

        print_most_similar_pairs(
            orig_vectors[layer], emotions, 20,
            f"Most Similar Pairs - Original (Layer {layer})"
        )

        print_most_similar_pairs(
            denoised_vectors[layer], emotions, 20,
            f"Most Similar Pairs - Denoised (Layer {layer})"
        )
    else:
        # Analyze multiple layers
        print("\nAnalyzing multiple layers...")
        plot_layer_comparison(
            orig_vectors, denoised_vectors, emotions, layers, output_dir
        )

        # Print similarity for middle layer
        mid_layer = layers[len(layers) // 2]
        print_most_similar_pairs(
            orig_vectors[mid_layer], emotions, 15,
            f"Most Similar Pairs - Original (Layer {mid_layer})"
        )

        print_most_similar_pairs(
            denoised_vectors[mid_layer], emotions, 15,
            f"Most Similar Pairs - Denoised (Layer {mid_layer})"
        )

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
