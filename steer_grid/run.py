"""
Grid-search steering: run one prompt through many (vector, strength, layer-range)
combinations in batched-parallel forward passes, dump completions to CSV.

See example_config.yaml for the config format.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = (
    r"H:\Models\huggingface\models--google--gemma-4-E2B-it"
    r"\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
)


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------


def load_vectors(source_dir: Path, device, dtype=torch.bfloat16):
    """Load all per-layer vectors from a source directory.

    Returns: dict[layer_idx][concept_name] -> torch.Tensor on device.
    Also returns the list of available layers.
    """
    with open(source_dir / "metadata.json") as f:
        meta = json.load(f)
    names = meta["emotions"]   # field is "emotions" for legacy reasons
    layers = meta["layers"]
    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        data = np.load(source_dir / f"emotion_vectors_layer_{layer}.npz")
        out[layer] = {
            n: torch.from_numpy(data[n]).to(device=device, dtype=dtype)
            for n in names
            if n in data.files
        }
    return out, names, layers


def find_model_layers(model):
    """Locate the transformer-decoder layer list, regardless of wrapping."""
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
# Batched per-position steering
# ---------------------------------------------------------------------------


class BatchedSteeringManager:
    """Applies a different steering vector to each batch position via hooks.

    Usage:
        m = BatchedSteeringManager(model, layers, vectors_by_source, hidden_size)
        m.set_batched(combos)        # combos: list of dicts with source/name/strength/layer_range
        m.disable()                  # turn steering off (e.g. during prefill)
        # ... run model.forward(batched_input) ...
        m.enable()
        # ... generate with steering on ...
        m.disable()
    """

    def __init__(self, model, layers, vectors_by_source, hidden_size, device):
        self.model = model
        self.layers = layers
        self.vectors_by_source = vectors_by_source
        self.hidden_size = hidden_size
        self.device = device
        self.hooks: list = []
        self.batch_size: int = 0
        self.deltas: dict[int, torch.Tensor] = {}  # layer_idx -> (B, D) tensor
        self.steering_enabled: bool = False

    # -- setup --------------------------------------------------------------

    def set_batched(self, combos: list[dict]):
        """Configure per-position steering for one batch.

        Each combo is a dict with keys: source, name, strength, layer_range.
        layer_range is (start, end) inclusive.
        """
        self.batch_size = len(combos)
        self.deltas = {}
        all_layers_used: set[int] = set()

        # Find every layer touched by any combo
        for c in combos:
            ls, le = c["layer_range"]
            for L in range(ls, le + 1):
                src_vecs = self.vectors_by_source.get(c["source"], {})
                if L in src_vecs and c["name"] in src_vecs[L]:
                    all_layers_used.add(L)

        # Build a (B, D) delta per touched layer
        for L in all_layers_used:
            delta = torch.zeros(
                (self.batch_size, self.hidden_size),
                device=self.device, dtype=torch.bfloat16,
            )
            for b, c in enumerate(combos):
                ls, le = c["layer_range"]
                if not (ls <= L <= le):
                    continue
                src_vecs = self.vectors_by_source.get(c["source"], {})
                if L not in src_vecs or c["name"] not in src_vecs[L]:
                    continue
                v = src_vecs[L][c["name"]]
                delta[b] = float(c["strength"]) * (v / v.norm())
            self.deltas[L] = delta

        self._register_hooks(all_layers_used)

    def _register_hooks(self, layer_indices):
        self.remove_hooks()
        for L in layer_indices:
            if L < len(self.layers):
                hook = self.layers[L].register_forward_hook(self._make_hook(L))
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, out):
            if not self.steering_enabled or layer_idx not in self.deltas:
                return out
            hidden = out[0] if isinstance(out, tuple) else out
            B, _, _ = hidden.shape
            delta = self.deltas[layer_idx]
            if delta.shape[0] != B:
                # batch size mismatch (e.g. final partial batch); skip safely
                return out
            modified = hidden + delta.unsqueeze(1).to(hidden.dtype)
            return (modified,) + out[1:] if isinstance(out, tuple) else modified
        return hook

    def enable(self): self.steering_enabled = True
    def disable(self): self.steering_enabled = False

    def __del__(self):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Sampling and batched generation
# ---------------------------------------------------------------------------


def sample_next_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample one token per row of logits. Supports temp + top-p (or greedy)."""
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
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_batch(
    model, tokenizer, manager: BatchedSteeringManager, prompt_ids: torch.Tensor,
    combos: list[dict],
    *, max_tokens: int, temperature: float, top_p: float, scope: str,
    eos_token_id: int,
):
    """Run one batched generation. Returns (completions, n_tokens)."""
    manager.set_batched(combos)
    B = len(combos)
    device = next(model.parameters()).device

    # Replicate the prompt for the batch
    input_ids = prompt_ids.unsqueeze(0).repeat(B, 1).to(device)

    # Prefill (steering off if assistant_only, on if all)
    if scope == "assistant_only":
        manager.disable()
    else:
        manager.enable()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_tokens = sample_next_tokens(out.logits[:, -1, :], temperature, top_p)

    # From here on, steering active for both modes
    manager.enable()

    generated = [next_tokens]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_tokens - 1):
        if done.all():
            break
        with torch.no_grad():
            out = model(next_tokens.unsqueeze(-1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = sample_next_tokens(out.logits[:, -1, :], temperature, top_p)
        # Force already-done positions to keep emitting EOS so the cache stays sane
        nxt = torch.where(done, torch.full_like(nxt, eos_token_id), nxt)
        generated.append(nxt)
        done |= (nxt == eos_token_id)
        next_tokens = nxt

    manager.disable()

    gen_ids = torch.stack(generated, dim=1).cpu().tolist()  # (B, T_gen)
    completions, n_tokens = [], []
    for seq in gen_ids:
        if eos_token_id in seq:
            seq = seq[:seq.index(eos_token_id)]
        n_tokens.append(len(seq))
        completions.append(tokenizer.decode(seq, skip_special_tokens=True))
    return completions, n_tokens


# ---------------------------------------------------------------------------
# Config / grid expansion
# ---------------------------------------------------------------------------


REQUIRED_KEYS = ["prompt", "vectors", "strengths", "layer_ranges", "sources"]


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing required keys: {missing}")
    # Defaults
    cfg.setdefault("system_prompt", "")
    cfg.setdefault("temperature", 0.7)
    cfg.setdefault("max_tokens", 200)
    cfg.setdefault("top_p", 1.0)
    cfg.setdefault("scope", "assistant_only")        # or "all"
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("seed", None)
    return cfg


def expand_grid(cfg: dict) -> list[dict]:
    combos = []
    for vec, strength, lrange in product(cfg["vectors"], cfg["strengths"], cfg["layer_ranges"]):
        if not isinstance(vec, dict) or "source" not in vec or "name" not in vec:
            raise SystemExit(f"Each vector must be {{source: ..., name: ...}}, got: {vec!r}")
        combos.append({
            "source": vec["source"],
            "name": vec["name"],
            "strength": float(strength),
            "layer_range": (int(lrange[0]), int(lrange[1])),
        })
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: ./runs/<timestamp>/)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    combos = expand_grid(cfg)
    print(f"Total combos: {len(combos)} = "
          f"{len(cfg['vectors'])} vectors x {len(cfg['strengths'])} strengths "
          f"x {len(cfg['layer_ranges'])} layer-ranges")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (Path(__file__).parent / "runs"
                   / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs -> {out_dir}")

    # Save settings (immediately, so even a partial run leaves the config behind)
    settings = {
        "config_path": str(config_path),
        "config": cfg,
        "model": args.model,
        "n_combos": len(combos),
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

    # Hidden dim
    cfg_obj = model.config
    if hasattr(cfg_obj, "text_config"):
        cfg_obj = cfg_obj.text_config
    hidden = getattr(cfg_obj, "hidden_size", None) or getattr(cfg_obj, "d_model", 1536)

    # ---- Load vector sources ----
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

    # Sanity: every (source, name) referenced exists somewhere
    for c in combos:
        src_vecs = vectors_by_source.get(c["source"])
        if src_vecs is None:
            raise SystemExit(f"Combo refers to unknown source {c['source']!r}")
        any_layer_has_it = any(c["name"] in layer_dict for layer_dict in src_vecs.values())
        if not any_layer_has_it:
            raise SystemExit(f"Source {c['source']!r} has no concept named "
                             f"{c['name']!r}")

    # ---- Steering manager ----
    layers = find_model_layers(model)
    print(f"\nFound {len(layers)} model layers")
    manager = BatchedSteeringManager(model, layers, vectors_by_source, hidden, device)

    # ---- Tokenize the prompt with chat template ----
    messages = []
    if cfg["system_prompt"]:
        messages.append({"role": "system", "content": cfg["system_prompt"]})
    messages.append({"role": "user", "content": cfg["prompt"]})
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
    print(f"Prompt: {len(prompt_ids)} tokens")

    # ---- Run batched generations ----
    if cfg["seed"] is not None:
        torch.manual_seed(int(cfg["seed"]))

    batch_size = int(cfg["batch_size"])
    eos = tokenizer.eos_token_id
    n_batches = (len(combos) + batch_size - 1) // batch_size
    print(f"\nGenerating {len(combos)} combos in {n_batches} batches "
          f"(batch_size={batch_size}, max_tokens={cfg['max_tokens']}, "
          f"temperature={cfg['temperature']}, scope={cfg['scope']})...")

    rows = []
    total_t0 = time.time()
    for bi, start in enumerate(range(0, len(combos), batch_size)):
        batch = combos[start:start + batch_size]
        bt0 = time.time()
        completions, n_tokens = generate_batch(
            model, tokenizer, manager, prompt_ids, batch,
            max_tokens=int(cfg["max_tokens"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            scope=cfg["scope"],
            eos_token_id=eos,
        )
        elapsed = time.time() - bt0
        per = elapsed / len(batch)
        print(f"  batch {bi+1}/{n_batches}: {len(batch)} combos in {elapsed:.1f}s "
              f"({per:.2f}s/combo)")

        for i, (c, comp, ntok) in enumerate(zip(batch, completions, n_tokens)):
            rows.append({
                "combo_id": start + i,
                "vector_source": c["source"],
                "vector_name": c["name"],
                "strength": c["strength"],
                "layer_start": c["layer_range"][0],
                "layer_end": c["layer_range"][1],
                "completion_tokens": ntok,
                "gen_seconds": round(per, 2),
                "completion": comp,
            })

    total_elapsed = time.time() - total_t0
    print(f"\nDone in {total_elapsed:.1f}s ({total_elapsed/len(combos):.2f}s avg per combo)")

    # ---- Write CSV ----
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "combo_id", "vector_source", "vector_name",
                "strength", "layer_start", "layer_end",
                "completion_tokens", "gen_seconds", "completion",
            ],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV written: {csv_path}")

    # Update settings with completion info
    settings["finished_at"] = datetime.now().isoformat()
    settings["total_seconds"] = round(total_elapsed, 2)
    settings["seconds_per_combo"] = round(total_elapsed / len(combos), 3)
    with open(out_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
