"""
Extract per-layer PCA directions as full hidden-dim vectors saved in the
same .npz layout as emotion_vectors_denoised, so they can be loaded by
the emotion viewer and self-steering apps as drop-in "emotions".

For each layer:
  - PCA on the (n_emotions, hidden_dim) denoised emotion-vector matrix
    (mean-centered).
  - Sign-align each PC so its dominant correlation with NRC VAD valence
    or arousal is positive (PC1 -> +valence, PC2 -> +arousal).
  - Scale unit-norm components by the median emotion-vector norm at that
    layer, so a steering coefficient of 1.0 yields a magnitude comparable
    to steering by a single emotion vector.

Saves to `pca_vectors/` with metadata listing the components as
"emotions": ["pc1_valence", "pc2_arousal", "pc3", "pc4", "pc5"].
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from analyze_pca import VAD_FALLBACKS, load_emotion_vectors, load_vad


COMPONENT_NAMES = ["pc1_valence", "pc2_arousal", "pc3", "pc4", "pc5"]


def fit_aligned_pca(
    X: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
    n_components: int,
):
    """Mean-centered PCA, signs aligned by VAD."""
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(Xc)

    components = pca.components_.copy()  # (n_components, hidden_dim), unit-norm rows
    flips = np.ones(n_components)
    for k in range(n_components):
        rv = np.corrcoef(scores[:, k], valence)[0, 1]
        ra = np.corrcoef(scores[:, k], arousal)[0, 1]
        ref = rv if abs(rv) >= abs(ra) else ra
        if ref < 0:
            components[k] = -components[k]
            scores[:, k] = -scores[:, k]
            flips[k] = -1.0

    corrs = []
    for k in range(n_components):
        corrs.append({
            "valence_pearson": float(np.corrcoef(scores[:, k], valence)[0, 1]),
            "arousal_pearson": float(np.corrcoef(scores[:, k], arousal)[0, 1]),
        })
    return components, scores, pca.explained_variance_ratio_, flips, corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="emotion_vectors_denoised")
    parser.add_argument("--vad-path", default="data/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt")
    parser.add_argument("--output-dir", default="pca_vectors")
    parser.add_argument("--n-components", type=int, default=5)
    args = parser.parse_args()

    if args.n_components > len(COMPONENT_NAMES):
        raise ValueError(
            f"n-components={args.n_components} exceeds known names {COMPONENT_NAMES}"
        )
    component_names = COMPONENT_NAMES[: args.n_components]

    vectors_dir = Path(args.vectors_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vectors from {vectors_dir}…")
    vectors_by_layer, emotions, layers = load_emotion_vectors(vectors_dir)
    print(f"  {len(emotions)} emotions, {len(layers)} layers ({layers[0]}..{layers[-1]})")

    print(f"Loading NRC VAD…")
    valence, arousal, vad_info = load_vad(Path(args.vad_path), emotions)
    print(f"  Direct: {vad_info['n_direct']}, Fallback: {vad_info['n_fallback']}")

    per_layer_info = {}
    print(f"\nFitting PCA per layer (mean-centered, sign-aligned)…")
    for layer in layers:
        X = vectors_by_layer[layer]
        components, scores, evr, flips, corrs = fit_aligned_pca(
            X, valence, arousal, args.n_components
        )

        # Magnitude calibration: scale unit-norm components to typical emotion-vector norm
        emo_norms = np.linalg.norm(X, axis=1)
        scale = float(np.median(emo_norms))

        layer_data = {}
        for name, comp in zip(component_names, components):
            layer_data[name] = (comp * scale).astype(np.float32)

        np.savez(output_dir / f"emotion_vectors_layer_{layer}.npz", **layer_data)

        per_layer_info[layer] = {
            "explained_variance_ratio": [float(x) for x in evr],
            "scale": scale,
            "median_emotion_norm": scale,
            "component_correlations": corrs,
            "sign_flipped": [bool(f < 0) for f in flips],
        }

    # Pick a representative layer for the headline summary
    rep_layer = 25 if 25 in layers else layers[len(layers) // 2]
    rep = per_layer_info[rep_layer]
    print(f"\nLayer {rep_layer} summary:")
    for name, evr_pc, c in zip(component_names,
                               rep["explained_variance_ratio"],
                               rep["component_correlations"]):
        print(f"  {name:14s} var={evr_pc:.1%}  "
              f"r(V)={c['valence_pearson']:+.2f}  r(A)={c['arousal_pearson']:+.2f}")
    print(f"  scale (median emotion norm) = {rep['scale']:.2f}")

    metadata = {
        "emotions": component_names,
        "layers": layers,
        "timestamp": datetime.now().isoformat(),
        "source_vectors_dir": str(args.vectors_dir),
        "preprocessing": "mean_center",
        "sign_alignment": "by_dominant_vad_axis",
        "scale_method": "median_emotion_vector_norm_per_layer",
        "vad_lexicon": "NRC-VAD-Lexicon",
        "vad_info": vad_info,
        "per_layer": {str(layer): info for layer, info in per_layer_info.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(layers)} layers to {output_dir}")
    print(f"Components per layer: {component_names}")
    print(f"\nTo use in apps, point them at this directory, e.g.:")
    print(f"  python emotion_viewer/app.py --vectors-dir {output_dir}")
    print(f"  python self_steering/app.py --vectors-dir {output_dir}")


if __name__ == "__main__":
    main()
