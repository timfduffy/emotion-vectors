"""
Hierarchical clustering heatmaps and UMAP visualizations for emotion vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import umap
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


def plot_clustered_heatmap(
    similarity: np.ndarray,
    emotions: list,
    title: str,
    output_path: str,
):
    """Plot similarity heatmap with hierarchical clustering."""
    # Convert similarity to distance (1 - similarity)
    # Clip to ensure valid distance values
    distance_matrix = np.clip(1 - similarity, 0, 2)

    # Get condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Get the order of leaves
    order = leaves_list(linkage_matrix)

    # Reorder similarity matrix and labels
    reordered_sim = similarity[order][:, order]
    reordered_emotions = [emotions[i] for i in order]

    # Create figure - make it large enough for all labels
    fig_size = max(20, len(emotions) * 0.12)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Plot heatmap
    im = ax.imshow(reordered_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Show all labels
    ax.set_xticks(range(len(reordered_emotions)))
    ax.set_xticklabels(reordered_emotions, rotation=90, fontsize=7)
    ax.set_yticks(range(len(reordered_emotions)))
    ax.set_yticklabels(reordered_emotions, fontsize=7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tsne(
    vectors: np.ndarray,
    emotions: list,
    title: str,
    output_path: str,
    flip_x: bool = False,
    flip_y: bool = False,
):
    """Create t-SNE visualization with all points labeled."""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms

    # Fit t-SNE
    print(f"  Computing t-SNE for {title}...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric='cosine',
        random_state=42,
        init='pca',
    )
    embedding = tsne.fit_transform(normalized)

    # Flip axes if requested
    if flip_x:
        embedding[:, 0] = -embedding[:, 0]
    if flip_y:
        embedding[:, 1] = -embedding[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot points
    ax.scatter(embedding[:, 0], embedding[:, 1], s=50, alpha=0.7, c='steelblue')

    # Add labels for all points
    for i, emotion in enumerate(emotions):
        ax.annotate(
            emotion,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=7,
            alpha=0.9,
            ha='center',
            va='bottom',
            xytext=(0, 3),
            textcoords='offset points',
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_umap(
    vectors: np.ndarray,
    emotions: list,
    title: str,
    output_path: str,
    flip_x: bool = False,
    flip_y: bool = False,
):
    """Create UMAP visualization with all points labeled."""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms

    # Fit UMAP
    print(f"  Computing UMAP for {title}...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
    )
    embedding = reducer.fit_transform(normalized)

    # Flip axes if requested
    if flip_x:
        embedding[:, 0] = -embedding[:, 0]
    if flip_y:
        embedding[:, 1] = -embedding[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot points
    ax.scatter(embedding[:, 0], embedding[:, 1], s=50, alpha=0.7, c='steelblue')

    # Add labels for all points
    for i, emotion in enumerate(emotions):
        ax.annotate(
            emotion,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=7,
            alpha=0.9,
            ha='center',
            va='bottom',
            xytext=(0, 3),
            textcoords='offset points',
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clustering and UMAP analysis")
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
    vectors, emotions, layers = load_emotion_vectors(args.denoised_dir)
    print(f"  {len(emotions)} emotions, {len(layers)} layers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hierarchical clustering heatmaps for layers 17 and 26
    print("\nGenerating hierarchical clustering heatmaps...")
    for layer in [17, 26]:
        similarity = compute_similarity_matrix(vectors[layer])
        plot_clustered_heatmap(
            similarity,
            emotions,
            f"Emotion Similarity - Layer {layer} (Hierarchical Clustering)",
            output_dir / f"clustered_heatmap_layer_{layer}.png",
        )

    # t-SNE plots for layers 20-27
    print("\nGenerating t-SNE plots for layers 20-27...")
    for layer in range(20, 28):
        plot_tsne(
            vectors[layer],
            emotions,
            f"Emotion t-SNE - Layer {layer} (Denoised)",
            output_dir / f"tsne_layer_{layer}.png",
            flip_x=True,
            flip_y=True,
        )

    # UMAP plots for layers 20-27
    print("\nGenerating UMAP plots for layers 20-27...")
    for layer in range(20, 28):
        plot_umap(
            vectors[layer],
            emotions,
            f"Emotion UMAP - Layer {layer} (Denoised)",
            output_dir / f"umap_layer_{layer}.png",
            flip_x=False,
            flip_y=False,
        )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
