"""
Three follow-up rotation analyses:

  1. Adjacent-layer alignment: raw vs Procrustes-aligned vs shuffled
     for every adjacent pair (L_i, L_{i+1}).

  2. Composition test: starting from X_0, apply the chain
        V_i = V_{i-1} @ R_{i-1, i}
     where R_{i-1, i} is the optimal Procrustes rotation between
     adjacent layers. If the per-step rotations compose smoothly,
     V_i should align well with X_i for any i. Compare to the direct
     single-step Procrustes(X_0, X_i).

     If composed alignment ~= direct alignment for all i, the model's
     cross-layer rotation behaves as a smooth chain of small rotations.
     If composed alignment falls below direct, then per-step rotations
     don't chain coherently (each is locally optimal but they don't agree
     globally).

  3. Limited rotation: for selected layer pairs, how many SVD axes
     does the rotation actually need? We truncate the small-space SVD
     to the top-k components and report alignment as k grows. The
     knee of the curve = effective number of "rotation planes" needed.

All in the small (n, n) subspace via QR caching, so it should run in <1 minute.
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


def per_row_cosine(A, B):
    a = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    b = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return (a * b).sum(axis=1)


def precompute_layer(X):
    """QR(X.T) -> (Q (d, n), R (n, n)) plus per-row norms of X."""
    Q, R = np.linalg.qr(X.T)
    norms = np.linalg.norm(X, axis=1)
    return Q, R, norms


def safe_svd(M):
    try:
        return np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        Mp = M + 1e-12 * np.random.default_rng(0).standard_normal(M.shape)
        return np.linalg.svd(Mp, full_matrices=False)


def aligned_cosines(R1, n1, R2, n2):
    """Per-row cosines of (X_i @ R) vs X_j, all in (n, n) subspace.

    Derivation as in analyze_layer_rotation.py:
    (X_i @ R) @ X_j.T  =  R1.T @ Us @ Vt @ R2     where M = R1 R2.T = Us S Vt.
    """
    Us, _, Vt = safe_svd(R1 @ R2.T)
    inner = (R1.T @ Us @ Vt @ R2).diagonal()
    return inner / (n1 * n2 + 1e-12)


def aligned_cosines_topk_subspace(X_i, X_j, B_k):
    """Subspace-restricted Procrustes: rotation acts only within span(B_k),
    identity on the orthogonal complement.

    B_k: (k, d) orthonormal basis defining the rotation subspace.

    This is the cleanly-interpretable version of "limited rotation":
    at k=0 we do nothing (alignment = raw cos), and as k grows we can
    only improve, since identity rotation is always feasible.

    (The earlier rank-k SVD truncation - aligned_cosines_topk - is a
    different beast: it projects to top-k subspace and zeroes the
    complement, which can give alignments BELOW raw cos at small k.)
    """
    Xi_proj = X_i @ B_k.T
    Xj_proj = X_j @ B_k.T
    M = Xi_proj.T @ Xj_proj
    Us, _, Vt = safe_svd(M)
    R_k = Us @ Vt
    X_i_aligned = X_i + Xi_proj @ (R_k @ B_k - B_k)
    a_norms = np.linalg.norm(X_i_aligned, axis=1).clip(1e-12)
    b_norms = np.linalg.norm(X_j, axis=1).clip(1e-12)
    return (X_i_aligned * X_j).sum(axis=1) / (a_norms * b_norms)


def chain_compose_alignment(vectors_by_layer, layers, Q_list, R_list):
    """Apply the chain R_0 ∘ R_1 ∘ ... iteratively from V_0 = X_0.

    Returns (composed_alignment, direct_alignment) where each is a
    list of length len(layers): per-row cosine median of V_i vs X_i,
    where V_0 = X_0 and V_i = V_{i-1} @ R_{i-1, i}.

    To apply R_{i-1, i} to V_{i-1}, we use:
       V @ R = (V @ Q1) @ Us @ Vt @ Q2.T
    where (Q1, R1) = QR(X_{i-1}.T) and (Q2, R2) = QR(X_i.T).
    """
    n_layers = len(layers)
    V = vectors_by_layer[layers[0]].copy()  # (n, d)
    X0 = vectors_by_layer[layers[0]]
    composed = [1.0]
    direct = [1.0]

    n0 = np.linalg.norm(X0, axis=1)

    for i in range(1, n_layers):
        prev = layers[i - 1]; cur = layers[i]
        Q1, R1 = Q_list[prev], R_list[prev]
        Q2, R2 = Q_list[cur],  R_list[cur]

        Us, _, Vt = safe_svd(R1 @ R2.T)
        # V = V @ Q1 @ Us @ Vt @ Q2.T   -- all small ops
        V_proj = V @ Q1                  # (n, n)
        V_rot = V_proj @ Us @ Vt         # (n, n)
        V = V_rot @ Q2.T                 # (n, d)

        Xi = vectors_by_layer[cur]
        composed.append(float(np.median(per_row_cosine(V, Xi))))

        # Direct: aligned X0 -> Xi single-shot Procrustes
        ni = np.linalg.norm(Xi, axis=1)
        direct.append(float(np.median(aligned_cosines(R_list[layers[0]], n0, R2, ni))))

    return composed, direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", default="emotion_vectors_denoised")
    parser.add_argument("--output-dir", default="analysis_output/rotation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument(
        "--limited-pairs", default="17-18,13-14,23-24,0-34",
        help="Comma-separated 'a-b' layer pairs for the top-k rotation analysis.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vectors from {args.vectors_dir}...")
    vectors_by_layer, emotions, layers = load_emotion_vectors(Path(args.vectors_dir))
    n, d = vectors_by_layer[layers[0]].shape
    print(f"  {n} concepts in {d}-d, {len(layers)} layers")

    print("Pre-computing QR per layer...")
    Q_list = {}; R_list = {}; norms_list = {}
    for layer in layers:
        Q, R, nm = precompute_layer(vectors_by_layer[layer])
        Q_list[layer] = Q; R_list[layer] = R; norms_list[layer] = nm

    rng = np.random.default_rng(args.seed)

    # ----- 1. Adjacent-pair analysis -----
    print("\n[1] Adjacent-layer alignment...")
    t0 = time.perf_counter()
    adj_layers = layers[:-1]
    raw_adj = []; rot_adj = []; shuf_adj = []
    for li, lj in zip(layers, layers[1:]):
        Xi = vectors_by_layer[li]; Xj = vectors_by_layer[lj]
        R1 = R_list[li]; R2 = R_list[lj]
        n1 = norms_list[li]; n2 = norms_list[lj]
        raw_adj.append(float(np.median(per_row_cosine(Xi, Xj))))
        rot_adj.append(float(np.median(aligned_cosines(R1, n1, R2, n2))))
        sh = []
        for _ in range(args.n_shuffles):
            perm = rng.permutation(n)
            sh.append(float(np.median(aligned_cosines(R1, n1, R2[:, perm], n2[perm]))))
        shuf_adj.append(float(np.mean(sh)))
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(adj_layers, raw_adj,  marker="o", label="Raw cos (no transform)",       color="tab:gray")
    ax.plot(adj_layers, rot_adj,  marker="o", label="After rotation",                color="tab:blue")
    ax.plot(adj_layers, shuf_adj, marker="o", label="Shuffled-label rotation\n(overfitting null)", color="tab:red", linestyle="--")
    ax.set_xlabel("Layer i (pair = layers i and i+1)")
    ax.set_ylabel("Median per-emotion cosine similarity")
    ax.set_title("Adjacent-layer alignment — does each layer's geometry rotate cleanly into the next?")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "adjacent_layer_alignment.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved adjacent_layer_alignment.png")

    # ----- 2. Composition test -----
    print("\n[2] Composition test...")
    t0 = time.perf_counter()
    composed_from_0, direct_from_0 = chain_compose_alignment(
        vectors_by_layer, layers, Q_list, R_list,
    )

    # Also a shuffled-label control: chain rotations fit on shuffled adjacent labels.
    # If even shuffled rotations chain into a high-alignment endpoint, composition
    # is artifactual; if not, the true chain is doing real geometric work.
    V_shuf = vectors_by_layer[layers[0]].copy()
    X0 = vectors_by_layer[layers[0]]
    composed_shuf = [1.0]
    for i in range(1, len(layers)):
        prev = layers[i - 1]; cur = layers[i]
        Q1 = Q_list[prev]; R1 = R_list[prev]
        Q2 = Q_list[cur];  R2 = R_list[cur]
        perm = rng.permutation(n)
        R2_perm = R2[:, perm]
        Q2_perm = Q2  # column space of Y_perm.T == column space of Y.T

        Us, _, Vt = safe_svd(R1 @ R2_perm.T)
        V_proj = V_shuf @ Q1
        V_rot = V_proj @ Us @ Vt
        V_shuf = V_rot @ Q2_perm.T  # land in the *original* Y subspace
        Xi = vectors_by_layer[cur]
        composed_shuf.append(float(np.median(per_row_cosine(V_shuf, Xi))))
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, direct_from_0,    marker="o", label="Direct Procrustes (X_0 -> X_i)",     color="tab:blue")
    ax.plot(layers, composed_from_0,  marker="o", label="Composed adjacent rotations\n(R_0 ∘ R_1 ∘ ... ∘ R_{i-1})", color="tab:green")
    ax.plot(layers, composed_shuf,    marker="o", label="Composed shuffled-label rotations\n(overfitting null)", color="tab:red", linestyle="--")
    ax.set_xlabel("Target layer i")
    ax.set_ylabel("Median per-emotion cosine similarity vs X_i")
    ax.set_title("Do adjacent rotations compose into the cross-layer rotation?")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "composition_vs_direct.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved composition_vs_direct.png")

    # ----- 3. Limited (subspace-restricted) rotation -----
    print("\n[3] Limited rotation (subspace-restricted, k axes)...")
    t0 = time.perf_counter()
    pairs = []
    for spec in args.limited_pairs.split(","):
        a, b = spec.split("-")
        pairs.append((int(a), int(b)))

    # Coarse k grid -- enough points to see the curve, log-spaced
    ks = sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, n]))

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5), squeeze=False)
    axes = axes[0]
    for ax, (la, lb) in zip(axes, pairs):
        if la not in layers or lb not in layers:
            ax.set_title(f"Layer {la}->{lb} not in metadata")
            continue
        Xa = vectors_by_layer[la]; Xb = vectors_by_layer[lb]
        raw_c = float(np.median(per_row_cosine(Xa, Xb)))

        # Per-pair subspace basis: top right singular vectors of [X_i; X_j]
        _, _, Vt_pair = np.linalg.svd(np.vstack([Xa, Xb]), full_matrices=False)
        B_pair_full = Vt_pair[: max(ks), :]

        rot_k = []
        shuf_k = []
        for k in ks:
            B_k = B_pair_full[:k, :]
            rot_k.append(float(np.median(aligned_cosines_topk_subspace(Xa, Xb, B_k))))
            sh = []
            for _ in range(args.n_shuffles):
                perm = rng.permutation(n)
                sh.append(float(np.median(aligned_cosines_topk_subspace(Xa, Xb[perm], B_k))))
            shuf_k.append(float(np.mean(sh)))

        ax.axhline(raw_c, color="tab:gray", linestyle=":", label=f"Raw cos = {raw_c:+.3f}")
        ax.plot(ks, rot_k,  marker="o", color="tab:blue",
                label="Aligned (rotation in top-k subspace)")
        ax.plot(ks, shuf_k, marker="x", color="tab:red", linestyle="--",
                label="Shuffled null")
        ax.set_xlabel("k = subspace dimension (rotation 'planes')")
        ax.set_ylabel("Median per-emotion cosine sim")
        ax.set_title(f"Layer {la} -> Layer {lb}")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("How many rotation 'planes' does each pair actually need?\n"
                 "(subspace-restricted Procrustes; identity outside the k-dim subspace)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "limited_rotation_topk.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  done in {time.perf_counter() - t0:.2f}s")
    print(f"  saved limited_rotation_topk.png")

    # ----- Console summary -----
    print("\n=== Summary ===")
    print(f"\nAdjacent pairs (n={len(adj_layers)}):")
    print(f"  raw cos median across pairs:        {np.median(raw_adj):+.3f}  "
          f"(min {min(raw_adj):+.3f}, max {max(raw_adj):+.3f})")
    print(f"  rotation-aligned cos median:        {np.median(rot_adj):+.3f}  "
          f"(min {min(rot_adj):+.3f}, max {max(rot_adj):+.3f})")
    print(f"  shuffled control median:            {np.median(shuf_adj):+.3f}")

    print(f"\nComposition test (chain from L0):")
    print(f"  direct alignment at L34:            {direct_from_0[-1]:+.3f}")
    print(f"  composed alignment at L34:          {composed_from_0[-1]:+.3f}")
    print(f"  composed shuffled-control at L34:   {composed_shuf[-1]:+.3f}")
    print(f"  gap (direct - composed):            {direct_from_0[-1] - composed_from_0[-1]:+.3f}")
    print()
    print("  Interpretation:")
    if abs(direct_from_0[-1] - composed_from_0[-1]) < 0.05:
        print("    Composed and direct alignments match -> rotations chain smoothly.")
    elif composed_from_0[-1] < direct_from_0[-1]:
        print("    Composed < direct -> per-step rotations don't fully compose; some")
        print("    error accumulates. The cross-layer geometry is rotation-like in")
        print("    aggregate but not via a coherent chain of small rotations.")
    else:
        print("    Composed > direct -> unusual; investigate.")


if __name__ == "__main__":
    main()
