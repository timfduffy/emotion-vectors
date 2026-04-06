"""
Representational Similarity Analysis (RSA) across layers.

Computes pairwise cosine similarity of similarity matrices across layers
to measure how stable the emotion representation structure is.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse


def load_emotion_vectors(vectors_dir: str) -> tuple[dict, list, list]:
    """Load emotion vectors and metadata."""
    vectors_dir = Path(vectors_dir)

    with open(vectors_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    emotions = metadata["emotions"]
    layers = metadata["layers"]

    vectors_by_layer = {}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        vectors_by_layer[layer] = np.stack([data[e] for e in emotions])

    return vectors_by_layer, emotions, layers


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms
    similarity = normalized @ normalized.T
    return similarity


def get_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Get upper triangle of matrix as flat vector (excluding diagonal)."""
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def compute_rsa_matrix(vectors_by_layer: dict, layers: list) -> np.ndarray:
    """Compute RSA matrix - cosine similarity between similarity matrices."""
    n_layers = len(layers)

    # First compute similarity matrices for each layer
    sim_vectors = {}
    for layer in layers:
        sim_matrix = compute_similarity_matrix(vectors_by_layer[layer])
        # Flatten upper triangle to vector
        sim_vectors[layer] = get_upper_triangle(sim_matrix)

    # Now compute cosine similarity between these vectors
    rsa_matrix = np.zeros((n_layers, n_layers))

    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            vec_i = sim_vectors[layer_i]
            vec_j = sim_vectors[layer_j]

            # Cosine similarity
            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)
            if norm_i > 0 and norm_j > 0:
                rsa_matrix[i, j] = np.dot(vec_i, vec_j) / (norm_i * norm_j)
            else:
                rsa_matrix[i, j] = 0

    return rsa_matrix


def plot_rsa_matrix(
    rsa_matrix: np.ndarray,
    layers: list,
    mid_late_layer: int,
    output_path: str,
):
    """Plot RSA matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use actual min value for color scale
    vmin = rsa_matrix.min()
    im = ax.imshow(rsa_matrix, cmap='viridis', vmin=vmin, vmax=1, aspect='equal')

    ax.set_title('Representational Similarity Across Layers\n(Cosine similarity of emotion similarity matrices)',
                 fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=9)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)

    # Mark mid-late layer
    if mid_late_layer in layers:
        idx = layers.index(mid_late_layer)
        ax.axhline(y=idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(len(layers) + 0.5, idx, f'← Layer {mid_late_layer} (mid-late)',
                va='center', fontsize=10, color='red')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RSA (Cosine Similarity)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Representational Similarity Analysis")
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

    args = parser.parse_args()

    print("Loading denoised vectors...")
    vectors, emotions, all_layers = load_emotion_vectors(args.denoised_dir)
    print(f"  {len(emotions)} emotions, {len(all_layers)} layers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select layers from 25% to 75%
    n_layers = len(all_layers)
    start_idx = int(n_layers * 0.25)
    end_idx = int(n_layers * 0.75)
    selected_layers = all_layers[start_idx:end_idx + 1]

    # Mid-late layer (about 2/3 through model)
    mid_late_idx = int(n_layers * 0.67)
    mid_late_layer = all_layers[mid_late_idx]

    print(f"\nAnalyzing layers {selected_layers[0]} to {selected_layers[-1]} ({len(selected_layers)} layers)")
    print(f"Mid-late layer (2/3 through): {mid_late_layer}")

    # Compute RSA matrix
    print("\nComputing RSA matrix...")
    rsa_matrix = compute_rsa_matrix(vectors, selected_layers)

    # Plot for selected layers (25%-75%)
    plot_rsa_matrix(
        rsa_matrix,
        selected_layers,
        mid_late_layer,
        output_dir / "rsa_across_layers.png",
    )

    # Also compute and plot for all layers
    print("\nComputing RSA matrix for all layers...")
    rsa_matrix_all = compute_rsa_matrix(vectors, all_layers)
    plot_rsa_matrix(
        rsa_matrix_all,
        all_layers,
        mid_late_layer,
        output_dir / "rsa_all_layers.png",
    )

    # Print some statistics
    print("\nRSA Statistics:")
    print(f"  Min similarity: {rsa_matrix.min():.4f}")
    print(f"  Max similarity (off-diagonal): {rsa_matrix[~np.eye(len(selected_layers), dtype=bool)].max():.4f}")
    print(f"  Mean similarity (off-diagonal): {rsa_matrix[~np.eye(len(selected_layers), dtype=bool)].mean():.4f}")

    # Find most/least similar layer pairs
    n = len(selected_layers)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((selected_layers[i], selected_layers[j], rsa_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost similar layer pairs:")
    for l1, l2, sim in pairs[:5]:
        print(f"  Layer {l1} <-> Layer {l2}: {sim:.4f}")

    print("\nLeast similar layer pairs:")
    for l1, l2, sim in pairs[-5:]:
        print(f"  Layer {l1} <-> Layer {l2}: {sim:.4f}")

    print(f"\nPlot saved to: {output_dir}")


if __name__ == "__main__":
    main()
