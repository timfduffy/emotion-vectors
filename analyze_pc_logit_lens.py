"""
Logit-lens analysis of steering direction vectors.

For each (layer, concept) pair we run:
    logits = lm_head(final_norm(vector))
and report the top-k positive and top-k negative tokens. This shows
which output tokens each direction most aligns with at each layer -
a direct way to see what semantic content the direction encodes.

Works on any steering vector source (PCA components, raw emotion vectors,
etc.) -- just point --vectors-dir at any folder with a metadata.json.

Outputs (filenames prefixed by the basename of the vectors-dir):
  - {prefix}_logit_lens_{with,no}_norm.json  -- full numeric data
  - {prefix}_logit_lens_{with,no}_norm.md    -- per-concept readable report
  - {prefix}_logit_lens_{with,no}_norm.csv   -- compact 10-column table

Notes on rigor:
  - Applies the model's final RMSNorm (logit lens convention) by default
    so the projection is in the same coordinate system the model uses
    for output. Use --no-final-norm to skip it.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pca_vectors(vectors_dir: Path):
    with open(vectors_dir / "metadata.json") as f:
        metadata = json.load(f)
    pc_names = metadata["emotions"]
    layers = metadata["layers"]
    vectors_by_layer = {}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        vectors_by_layer[layer] = {n: data[n].astype(np.float32) for n in pc_names}
    return vectors_by_layer, pc_names, layers


def find_lm_head_and_norm(model):
    """Locate (final_norm_module, lm_head_weight) on a Gemma-style model.

    Tries multiple paths so this works whether the model is wrapped in a
    multimodal container or not.
    """
    # lm_head: usually accessible as model.lm_head; sometimes nested
    lm_head = None
    for getter in [
        lambda m: m.lm_head,
        lambda m: m.language_model.lm_head,
        lambda m: m.model.lm_head,
    ]:
        try:
            cand = getter(model)
            if cand is not None:
                lm_head = cand
                break
        except AttributeError:
            continue

    # final norm: usually m.model.norm or m.model.language_model.model.norm
    final_norm = None
    for getter in [
        lambda m: m.model.language_model.model.norm,
        lambda m: m.model.language_model.norm,
        lambda m: m.language_model.model.norm,
        lambda m: m.model.norm,
    ]:
        try:
            cand = getter(model)
            if cand is not None:
                final_norm = cand
                break
        except AttributeError:
            continue

    if lm_head is None or final_norm is None:
        raise RuntimeError(
            f"Could not locate lm_head ({'found' if lm_head is not None else 'missing'}) "
            f"or final_norm ({'found' if final_norm is not None else 'missing'}) on the model."
        )
    return final_norm, lm_head


def topk_tokens(logits: torch.Tensor, tokenizer, k: int):
    """Return (top_k tokens-and-scores, bottom_k tokens-and-scores)."""
    top_vals, top_idx = torch.topk(logits, k=k, largest=True)
    bot_vals, bot_idx = torch.topk(logits, k=k, largest=False)

    def fmt(idx, vals):
        out = []
        for i, v in zip(idx.tolist(), vals.tolist()):
            tok = tokenizer.decode([i])
            tok_repr = tok.replace("\n", "\\n").replace("\r", "\\r")
            out.append({"token_id": i, "token": tok_repr, "logit": float(v)})
        return out

    return fmt(top_idx, top_vals), fmt(bot_idx, bot_vals)


def compute_logit_lens_batched(
    vectors_by_layer: dict,
    concept_names: list[str],
    layers: list[int],
    final_norm,
    lm_head,
    device,
    apply_norm: bool,
    top_k: int,
    chunk_size: int = 256,
):
    """Batched matmul through the unembedding for every (layer, concept).

    Returns: dict[concept][layer] -> {"top": [...], "bottom": [...]}
    """
    # Build a flat list of (concept, layer) -> vector
    flat_keys: list[tuple[str, int]] = []
    flat_vecs: list[np.ndarray] = []
    for concept in concept_names:
        for layer in layers:
            flat_keys.append((concept, layer))
            flat_vecs.append(vectors_by_layer[layer][concept])

    # Stack and move to GPU
    X = torch.from_numpy(np.stack(flat_vecs)).to(device=device, dtype=torch.bfloat16)
    if apply_norm:
        X = final_norm(X)
    # Compute logits in fp32 in chunks to keep peak memory bounded
    W = lm_head.weight.float()  # (vocab, hidden)
    n = X.shape[0]
    results = {c: {} for c in concept_names}

    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size].float()
        logits = chunk @ W.T  # (chunk, vocab)
        top_vals, top_idx = torch.topk(logits, k=top_k, largest=True)
        bot_vals, bot_idx = torch.topk(logits, k=top_k, largest=False)
        top_vals = top_vals.cpu()
        top_idx = top_idx.cpu()
        bot_vals = bot_vals.cpu()
        bot_idx = bot_idx.cpu()

        for j in range(chunk.shape[0]):
            concept, layer = flat_keys[i + j]
            results[concept][layer] = {
                "top_idx": top_idx[j].tolist(),
                "top_logit": top_vals[j].tolist(),
                "bot_idx": bot_idx[j].tolist(),
                "bot_logit": bot_vals[j].tolist(),
            }
    return results


def decode_results(results: dict, tokenizer, top_k: int):
    """Convert raw token-id results into rich {token, logit} entries."""
    out = {}
    for concept, by_layer in results.items():
        out[concept] = {}
        for layer, raw in by_layer.items():
            top = [
                {"token_id": tid,
                 "token": tokenizer.decode([tid]).replace("\n", "\\n").replace("\r", "\\r"),
                 "logit": float(lg)}
                for tid, lg in zip(raw["top_idx"][:top_k], raw["top_logit"][:top_k])
            ]
            bot = [
                {"token_id": tid,
                 "token": tokenizer.decode([tid]).replace("\n", "\\n").replace("\r", "\\r"),
                 "logit": float(lg)}
                for tid, lg in zip(raw["bot_idx"][:top_k], raw["bot_logit"][:top_k])
            ]
            out[concept][layer] = {"top": top, "bottom": bot}
    return out


def write_csv(results: dict, concept_names: list[str], layers: list[int],
              out_path: Path, n_cols: int = 10):
    """Compact CSV: one row per (concept, layer, direction), top n_cols tokens."""
    header = (
        ["concept", "layer", "direction"]
        + [f"token_{i+1}" for i in range(n_cols)]
        + [f"logit_{i+1}" for i in range(n_cols)]
    )
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for concept in concept_names:
            for layer in layers:
                entry = results[concept][layer]
                for direction, tokens in [("top", entry["top"]), ("bottom", entry["bottom"])]:
                    tokens = tokens[:n_cols]
                    toks = [t["token"] for t in tokens] + [""] * (n_cols - len(tokens))
                    logs = [f"{t['logit']:.4f}" for t in tokens] + [""] * (n_cols - len(tokens))
                    w.writerow([concept, layer, direction] + toks + logs)


def write_markdown(results: dict, concept_names: list[str], layers: list[int],
                   model_path: str, vectors_dir: str, apply_norm: bool,
                   top_k: int, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Logit lens ({'with' if apply_norm else 'without'} final RMSNorm)\n\n")
        f.write(f"Model: `{model_path}`\n\n")
        f.write(f"Vectors: `{vectors_dir}`\n\n")
        f.write(f"Top {top_k} tokens per (layer, concept).\n\n")
        for concept in concept_names:
            f.write(f"## {concept}\n\n")
            for layer in layers:
                entry = results[concept][layer]
                f.write(f"### Layer {layer}\n\n")
                f.write("**Top (positive direction):** ")
                f.write(", ".join(f"`{t['token']}`" for t in entry["top"]))
                f.write("\n\n**Bottom (negative direction):** ")
                f.write(", ".join(f"`{t['token']}`" for t in entry["bottom"]))
                f.write("\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=r"H:\Models\huggingface\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
    )
    parser.add_argument(
        "--vectors-dir", default="pca_vectors",
        help="Directory of vectors (must contain metadata.json). Can be repeated by "
             "running the script once per source.",
    )
    parser.add_argument("--output-dir", default="analysis_output/logit_lens")
    parser.add_argument("--prefix", default=None,
                        help="Prefix for output filenames. Default: basename of --vectors-dir.")
    parser.add_argument("--layer-start", type=int, default=None,
                        help="Lowest layer to analyze. Default: all layers in metadata.")
    parser.add_argument("--layer-end", type=int, default=None,
                        help="Highest layer to analyze. Default: all layers in metadata.")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-k tokens to keep per (layer, concept, direction).")
    parser.add_argument("--csv-cols", type=int, default=10,
                        help="Number of token columns in the CSV output.")
    parser.add_argument("--no-final-norm", action="store_true",
                        help="Skip the final RMSNorm before unembedding.")
    parser.add_argument("--no-md", action="store_true",
                        help="Skip the Markdown report (useful for very large concept sets).")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="How many vectors to push through unembedding per matmul chunk.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_dir = Path(args.vectors_dir)
    prefix = args.prefix or vectors_dir.name

    print(f"Loading vectors from {vectors_dir}...")
    vectors_by_layer, concept_names, all_layers = load_pca_vectors(vectors_dir)
    if args.layer_start is None and args.layer_end is None:
        layers_to_use = list(all_layers)
    else:
        lo = args.layer_start if args.layer_start is not None else min(all_layers)
        hi = args.layer_end if args.layer_end is not None else max(all_layers)
        layers_to_use = [l for l in all_layers if lo <= l <= hi]
    if not layers_to_use:
        raise SystemExit("No layers in the requested range")
    if max(args.csv_cols, 0) > args.top_k:
        raise SystemExit(f"--csv-cols ({args.csv_cols}) > --top-k ({args.top_k})")

    print(f"  {len(concept_names)} concepts, {len(layers_to_use)} layers "
          f"({layers_to_use[0]}..{layers_to_use[-1]})")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    final_norm, lm_head = find_lm_head_and_norm(model)
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}. Vocab size: {lm_head.weight.shape[0]}")

    apply_norm = not args.no_final_norm
    suffix = "with_norm" if apply_norm else "no_norm"
    n_jobs = len(concept_names) * len(layers_to_use)
    print(f"\nComputing logit lens ({'with' if apply_norm else 'without'} final norm), "
          f"{n_jobs} (concept, layer) pairs in chunks of {args.chunk_size}...")

    raw_results = compute_logit_lens_batched(
        vectors_by_layer, concept_names, layers_to_use,
        final_norm, lm_head, device,
        apply_norm=apply_norm, top_k=args.top_k, chunk_size=args.chunk_size,
    )
    results = decode_results(raw_results, tokenizer, args.top_k)

    # ---- JSON ----
    json_path = out_dir / f"{prefix}_logit_lens_{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vectors_dir": str(vectors_dir),
                "model": args.model,
                "layers": layers_to_use,
                "concepts": concept_names,
                "top_k": args.top_k,
                "apply_final_norm": apply_norm,
                "results": {
                    c: {
                        str(layer): {
                            "top": [t["token"] for t in entry["top"]],
                            "bottom": [t["token"] for t in entry["bottom"]],
                            "top_full": entry["top"],
                            "bottom_full": entry["bottom"],
                        }
                        for layer, entry in by_layer.items()
                    }
                    for c, by_layer in results.items()
                },
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved: {json_path}")

    # ---- CSV ----
    csv_path = out_dir / f"{prefix}_logit_lens_{suffix}.csv"
    write_csv(results, concept_names, layers_to_use, csv_path, n_cols=args.csv_cols)
    print(f"Saved: {csv_path}")

    # ---- Markdown ----
    if not args.no_md:
        md_path = out_dir / f"{prefix}_logit_lens_{suffix}.md"
        write_markdown(results, concept_names, layers_to_use,
                       args.model, str(vectors_dir), apply_norm, args.top_k, md_path)
        print(f"Saved: {md_path}")

    # ---- Console preview at a representative layer ----
    def _safe(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")

    preview_layer = 25 if 25 in layers_to_use else layers_to_use[len(layers_to_use) // 2]
    print(f"\n=== Preview at layer {preview_layer} ({min(8, len(concept_names))} of "
          f"{len(concept_names)} concepts) ===")
    for concept in concept_names[:8]:
        entry = results[concept][preview_layer]
        top_str = ", ".join(t["token"] for t in entry["top"][:6])
        bot_str = ", ".join(t["token"] for t in entry["bottom"][:6])
        print(f"\n{concept}:")
        print(f"  + {_safe(top_str)}")
        print(f"  - {_safe(bot_str)}")


if __name__ == "__main__":
    main()
