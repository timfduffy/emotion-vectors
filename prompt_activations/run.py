"""
Generate responses to many prompts and collect per-token, per-layer projections
onto a configurable set of concept vectors (emotions and/or PCA components).

Per prompt we save an .npz with the response text + a (n_response_tokens,
n_layers, n_concepts) float16 activation tensor. A summary CSV with one row
per prompt records the response plus mean PC1/PC2 at a chosen layer.

See example_config.yaml for the input format.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = (
    r"H:\Models\huggingface\models--google--gemma-4-E2B-it"
    r"\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ["prompts_csv", "sources"]


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing required keys: {missing}")
    cfg.setdefault("system_prompt", "")
    cfg.setdefault("temperature", 0.7)
    cfg.setdefault("max_tokens", 200)
    cfg.setdefault("top_p", 1.0)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("seed", None)
    cfg.setdefault("limit", None)
    cfg.setdefault("summary_layer", 25)
    cfg.setdefault("dtype_storage", "float16")
    return cfg


# ---------------------------------------------------------------------------
# Vector loading and projection-matrix construction
# ---------------------------------------------------------------------------


def load_vectors(source_dir: Path, device, dtype=torch.float32):
    """Returns dict[layer_idx][concept_name] -> (hidden,) tensor on device."""
    with open(source_dir / "metadata.json") as f:
        meta = json.load(f)
    names = meta["emotions"]
    layers = meta["layers"]
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        data = np.load(source_dir / f"emotion_vectors_layer_{layer}.npz")
        out[layer] = {
            n: torch.from_numpy(data[n]).to(device=device, dtype=dtype)
            for n in names if n in data.files
        }
    return out, names, layers


def build_projection_matrix(
    vectors_by_source: dict, model_layers: list[int], hidden: int, device,
):
    """Concatenate every source's concepts into a single per-layer projection
    matrix of unit-normed vectors.

    Returns:
      M: (n_layers, hidden, n_concepts) tensor of unit-norm vectors per layer
      concept_index: list of {source, name} dicts in column order
      layer_index: list of layer indices in row order
    """
    concept_index = []
    for source_label, by_layer in vectors_by_source.items():
        any_layer = next(iter(by_layer.values()))
        for name in any_layer.keys():
            concept_index.append({"source": source_label, "name": name})

    n_layers = len(model_layers)
    n_concepts = len(concept_index)
    M = torch.zeros((n_layers, hidden, n_concepts), device=device, dtype=torch.float32)
    for ci, entry in enumerate(concept_index):
        src_layers = vectors_by_source[entry["source"]]
        for li, layer in enumerate(model_layers):
            if layer not in src_layers or entry["name"] not in src_layers[layer]:
                continue
            v = src_layers[layer][entry["name"]]
            M[li, :, ci] = v / v.norm().clamp(min=1e-12)
    return M, concept_index


def find_model_layers(model):
    for getter in [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
    ]:
        try:
            layers = getter(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Could not locate transformer layers on the model")


# ---------------------------------------------------------------------------
# Hook-based per-layer activation collector
# ---------------------------------------------------------------------------


class StepActivationCollector:
    """For each forward pass, captures the LAST-position hidden state at every
    layer. Cleared between steps; stays disabled during prompt prefill.
    """
    def __init__(self, layers):
        self.layers = layers
        self.last_step: dict[int, torch.Tensor] = {}
        self.enabled = False
        self.hooks = []

    def register(self):
        for L, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(self._make_hook(L)))

    def _make_hook(self, L):
        def hook(module, inp, out):
            if not self.enabled:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            self.last_step[L] = hidden[:, -1, :].detach()
            return out
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset(self):
        self.last_step = {}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_next_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Batched prompt generation + activation collection
# ---------------------------------------------------------------------------


def get_stop_token_ids(tokenizer) -> set[int]:
    """Collect every token id that should terminate a generation: eos plus
    any end-of-turn / end-of-conversation marker the tokenizer knows about."""
    stop = set()
    if tokenizer.eos_token_id is not None:
        stop.add(int(tokenizer.eos_token_id))
    # Standard tokenizer attributes that may exist on chat-model tokenizers
    for attr in ("eot_token_id", "end_of_turn_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            stop.add(int(tid))
    # Common end-of-turn token strings across model families. Gemma 4 uses
    # `<turn|>` (id 106); Gemma 2/3 use `<end_of_turn>`.
    for tok in ("<turn|>", "<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                stop.add(tid)
        except Exception:
            pass
    return stop


def generate_and_collect_batch(
    model, tokenizer, collector: StepActivationCollector, projection_matrix: torch.Tensor,
    prompts: list[str], system_prompt: str,
    *, max_tokens: int, temperature: float, top_p: float,
    storage_dtype: torch.dtype, stop_token_ids: set[int], pad_token_id: int,
):
    """Run one batched generation. Returns list of dicts (one per prompt):
        {prompt, response_text, response_token_ids, projections}
    where projections has shape (n_response_tokens, n_layers, n_concepts).
    """
    device = next(model.parameters()).device

    # Build chat-template prompts
    chat_texts = []
    for p in prompts:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p})
        chat_texts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    # Left-pad for batched generation
    enc = tokenizer(
        chat_texts, padding=True, return_tensors="pt", padding_side="left",
    )
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    B, prompt_len = input_ids.shape

    stop_tensor = torch.tensor(sorted(stop_token_ids), device=device)

    # ---- Prefill (collector OFF) ----
    collector.reset()
    collector.enabled = False
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_token = sample_next_tokens(out.logits[:, -1, :], temperature, top_p)

    # ---- Generation loop (collector ON) ----
    collector.enabled = True
    response_tokens: list[torch.Tensor] = []  # list of (B,) tensors
    per_step_proj: list[torch.Tensor] = []    # list of (B, n_layers, n_concepts) on CPU
    done = torch.zeros(B, dtype=torch.bool, device=device)

    n_layers = projection_matrix.shape[0]

    for step in range(max_tokens):
        if done.all():
            break
        # Append a 1 to attention mask
        attn = torch.cat([attn, torch.ones(B, 1, device=device, dtype=attn.dtype)], dim=1)
        with torch.no_grad():
            out = model(
                input_ids=next_token.unsqueeze(-1),
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values

        # Project this step's per-layer captures
        # Stack to (n_layers, B, hidden), then einsum with projection_matrix -> (B, n_layers, n_concepts)
        captures = torch.stack([collector.last_step[L] for L in range(n_layers)], dim=0)
        # Cast to fp32 for the matmul, then to storage dtype for CPU offload
        proj = torch.einsum("lbh,lhc->blc", captures.float(), projection_matrix)
        per_step_proj.append(proj.to(storage_dtype).cpu())

        # Record the token we just processed
        response_tokens.append(next_token.detach().cpu())

        # Sample next token
        nxt = sample_next_tokens(out.logits[:, -1, :], temperature, top_p)
        # Force already-done positions to keep emitting a stop token
        nxt = torch.where(done, torch.full_like(nxt, pad_token_id), nxt)
        # Update done flag
        is_stop = (nxt.unsqueeze(-1) == stop_tensor.unsqueeze(0)).any(dim=-1)
        # But we mark done ONLY after we've captured the stop token's predecessor;
        # actually, we DO want to capture activations for the stop token itself, which
        # would happen on the next step (when we feed it). To keep things simple we
        # treat stop as "this prompt is done; no more capture".
        next_token = nxt
        done = done | is_stop

    collector.enabled = False

    # ---- Stack and trim per prompt ----
    if not response_tokens:
        # Shouldn't happen since max_tokens >= 1, but handle gracefully
        return [{"prompt": p, "response_text": "", "response_token_ids": np.array([], dtype=np.int64),
                 "projections": np.zeros((0, n_layers, projection_matrix.shape[2]),
                                          dtype=np.float16 if storage_dtype == torch.float16 else np.float32)}
                for p in prompts]

    tokens_tensor = torch.stack(response_tokens, dim=1)            # (B, n_steps)
    proj_tensor   = torch.stack(per_step_proj, dim=1)              # (B, n_steps, n_layers, n_concepts)

    results = []
    for b in range(B):
        seq = tokens_tensor[b].tolist()
        # Trim at the first stop token (and don't include it in the response)
        cut = len(seq)
        for i, t in enumerate(seq):
            if t in stop_token_ids:
                cut = i
                break
        toks = seq[:cut]
        proj = proj_tensor[b, :cut].numpy()
        text = tokenizer.decode(toks, skip_special_tokens=True)
        results.append({
            "prompt": prompts[b],
            "response_text": text,
            "response_token_ids": np.array(toks, dtype=np.int64),
            "projections": proj,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_prompts(csv_path: Path, limit) -> list[str]:
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "prompt" not in reader.fieldnames:
            raise SystemExit(f"prompts_csv {csv_path} must have a 'prompt' column "
                             f"(found {reader.fieldnames})")
        prompts = [row["prompt"] for row in reader]
    if limit is not None:
        prompts = prompts[: int(limit)]
    return prompts


def find_pc_concept_indices(concept_index, names=("pc1_valence", "pc2_arousal")):
    """Return dict: concept_name -> column index in projection matrix."""
    out = {}
    for ci, entry in enumerate(concept_index):
        if entry["name"] in names:
            out[entry["name"]] = ci
    return out


def write_summary_csv(out_path: Path, per_prompt_dir: Path, summary_layer: int,
                      concept_index: list, layer_index: list[int]):
    """Scan all per-prompt npz files and write one CSV row per file."""
    pc_cols = find_pc_concept_indices(concept_index, ("pc1_valence", "pc2_arousal"))
    if "pc1_valence" not in pc_cols or "pc2_arousal" not in pc_cols:
        print("  WARNING: pc1_valence and/or pc2_arousal not in concepts - "
              "summary mean columns will be empty.")
    if summary_layer not in layer_index:
        raise SystemExit(f"summary_layer={summary_layer} not in available layers {layer_index}")
    li = layer_index.index(summary_layer)

    files = sorted(per_prompt_dir.glob("*.npz"))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["prompt_id", "n_response_tokens", "mean_pc1", "mean_pc2",
                    "prompt", "response"])
        for fp in files:
            data = np.load(fp, allow_pickle=True)
            prompt_id = int(fp.stem)
            proj = data["projections"]            # (n, n_layers, n_concepts)
            n = proj.shape[0]
            row_pc1 = ""
            row_pc2 = ""
            if n > 0:
                if "pc1_valence" in pc_cols:
                    row_pc1 = f"{proj[:, li, pc_cols['pc1_valence']].mean():.4f}"
                if "pc2_arousal" in pc_cols:
                    row_pc2 = f"{proj[:, li, pc_cols['pc2_arousal']].mean():.4f}"
            w.writerow([
                prompt_id, n, row_pc1, row_pc2,
                str(data["prompt_text"]), str(data["response_text"]),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: ./runs/<timestamp>/)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-resume", action="store_true",
                        help="Always process every prompt, even if its .npz exists")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (Path(__file__).parent / "runs"
                   / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_dir = out_dir / "per_prompt"
    per_prompt_dir.mkdir(exist_ok=True)
    print(f"Outputs -> {out_dir}")

    prompts_csv = Path(cfg["prompts_csv"])
    if not prompts_csv.is_absolute():
        prompts_csv = (config_path.parent / prompts_csv).resolve()
    prompts = load_prompts(prompts_csv, cfg["limit"])
    print(f"Loaded {len(prompts)} prompts from {prompts_csv}")

    # Save settings
    settings = {
        "config_path": str(config_path),
        "config": cfg,
        "model": args.model,
        "n_prompts": len(prompts),
        "started_at": datetime.now().isoformat(),
    }
    with open(out_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, default=str, ensure_ascii=False)

    # ---- Load model ----
    print(f"\nLoading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    device = next(model.parameters()).device

    # Hidden size
    cfg_obj = model.config
    if hasattr(cfg_obj, "text_config"):
        cfg_obj = cfg_obj.text_config
    hidden = getattr(cfg_obj, "hidden_size", None) or getattr(cfg_obj, "d_model", 1536)

    # ---- Load vector sources, build projection matrix ----
    print("\nLoading vector sources...")
    vectors_by_source = {}
    for label, path in cfg["sources"].items():
        p = Path(path)
        if not p.is_absolute():
            p = (config_path.parent / p).resolve()
        if not (p / "metadata.json").exists():
            raise SystemExit(f"Source '{label}' has no metadata.json at {p}")
        vecs, names, layers_in_src = load_vectors(p, device)
        vectors_by_source[label] = vecs
        print(f"  {label!r}: {len(names)} concepts, {len(layers_in_src)} layers ({p})")

    layers = find_model_layers(model)
    n_layers = len(layers)
    layer_index = list(range(n_layers))
    print(f"\nFound {n_layers} model layers")

    storage_dtype = torch.float16 if cfg["dtype_storage"] == "float16" else torch.float32
    np_storage_dtype = np.float16 if storage_dtype == torch.float16 else np.float32

    M, concept_index = build_projection_matrix(
        vectors_by_source, layer_index, hidden, device,
    )
    print(f"Projection matrix: {tuple(M.shape)} "
          f"({M.numel() * 4 / 1e6:.1f} MB fp32 on GPU), "
          f"{len(concept_index)} concepts total")

    # Save concept index
    with open(out_dir / "concept_index.json", "w", encoding="utf-8") as f:
        json.dump({"concepts": concept_index, "layers": layer_index}, f, indent=2)

    # ---- Activation collector ----
    collector = StepActivationCollector(layers)
    collector.register()

    # ---- Determine which prompts still need processing ----
    if args.no_resume:
        todo_ids = list(range(len(prompts)))
    else:
        existing = {int(fp.stem) for fp in per_prompt_dir.glob("*.npz")}
        todo_ids = [i for i in range(len(prompts)) if i not in existing]
        if existing:
            print(f"\nResuming: {len(existing)} prompts already done, "
                  f"{len(todo_ids)} remaining")

    if not todo_ids:
        print("Nothing to do.")
        write_summary_csv(out_dir / "summary.csv", per_prompt_dir,
                          int(cfg["summary_layer"]), concept_index, layer_index)
        print(f"Summary CSV: {out_dir / 'summary.csv'}")
        return

    if cfg["seed"] is not None:
        torch.manual_seed(int(cfg["seed"]))

    stop_ids = get_stop_token_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    print(f"Stop tokens: {sorted(stop_ids)}, pad: {pad_id}")

    batch_size = int(cfg["batch_size"])
    n_batches = (len(todo_ids) + batch_size - 1) // batch_size
    print(f"\nProcessing {len(todo_ids)} prompts in {n_batches} batches "
          f"(batch_size={batch_size}, max_tokens={cfg['max_tokens']}, "
          f"temperature={cfg['temperature']})...")

    total_t0 = time.time()
    for bi, start in enumerate(range(0, len(todo_ids), batch_size)):
        batch_ids = todo_ids[start: start + batch_size]
        batch_prompts = [prompts[i] for i in batch_ids]
        bt0 = time.time()
        results = generate_and_collect_batch(
            model, tokenizer, collector, M,
            batch_prompts, cfg["system_prompt"],
            max_tokens=int(cfg["max_tokens"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            storage_dtype=storage_dtype,
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
        )
        elapsed = time.time() - bt0
        per = elapsed / len(batch_ids)

        # Save per-prompt npz files
        for pid, res in zip(batch_ids, results):
            np.savez_compressed(
                per_prompt_dir / f"{pid:05d}.npz",
                prompt_text=res["prompt"],
                response_text=res["response_text"],
                response_token_ids=res["response_token_ids"],
                projections=res["projections"].astype(np_storage_dtype),
            )

        # Show a teeny preview of the first prompt of the batch
        first = results[0]
        ascii_resp = first["response_text"].encode("ascii", errors="replace").decode("ascii")
        preview = ascii_resp[:80].replace("\n", " ")
        print(f"  batch {bi+1}/{n_batches}: {len(batch_ids)} prompts in {elapsed:.1f}s "
              f"({per:.2f}s/prompt) | first: {preview!r}")

    total_elapsed = time.time() - total_t0
    print(f"\nDone in {total_elapsed:.1f}s "
          f"({total_elapsed / max(1, len(todo_ids)):.2f}s avg per prompt)")

    collector.remove()

    # ---- Write summary CSV from all .npz files ----
    print("\nWriting summary CSV...")
    write_summary_csv(out_dir / "summary.csv", per_prompt_dir,
                      int(cfg["summary_layer"]), concept_index, layer_index)
    print(f"Summary CSV: {out_dir / 'summary.csv'}")

    settings["finished_at"] = datetime.now().isoformat()
    settings["total_seconds"] = round(total_elapsed, 2)
    settings["seconds_per_prompt"] = round(total_elapsed / max(1, len(todo_ids)), 3)
    with open(out_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
