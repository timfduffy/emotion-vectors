"""
Apply PCA denoising to existing emotion vectors.

Uses neutral dialogues to identify confounding directions and projects
them out of the emotion vectors.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import argparse
from datetime import datetime
import gc


def load_neutral_dialogues(dialogues_dir: str) -> list[str]:
    """Load neutral dialogues from directory."""
    dialogues_dir = Path(dialogues_dir)

    # Try consolidated file first
    all_dialogues_file = dialogues_dir / "all_dialogues.json"
    if all_dialogues_file.exists():
        with open(all_dialogues_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        dialogues = []
        for topic_data in data["dialogues"]:
            dialogues.extend(topic_data.get("dialogues", []))
        return dialogues

    # Fall back to checkpoints
    checkpoint_dir = dialogues_dir / "checkpoints"
    if checkpoint_dir.exists():
        dialogues = []
        for cp in tqdm(list(checkpoint_dir.glob("*.json")), desc="Loading dialogues"):
            with open(cp, "r", encoding="utf-8") as f:
                data = json.load(f)
                dialogues.extend(data.get("dialogues", []))
        return dialogues

    raise FileNotFoundError(f"No dialogues found in {dialogues_dir}")


def load_emotion_vectors(vectors_dir: str) -> tuple[dict, dict]:
    """Load existing emotion vectors and metadata."""
    vectors_dir = Path(vectors_dir)

    with open(vectors_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    emotions = metadata["emotions"]
    layers = metadata["layers"]

    emotion_vectors = {e: {} for e in emotions}

    for layer in tqdm(layers, desc="Loading emotion vectors"):
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        for emotion in emotions:
            emotion_vectors[emotion][layer] = torch.from_numpy(data[emotion])

    return emotion_vectors, metadata


def get_model_config(model):
    """Get num_layers and hidden_size, handling different config formats."""
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'num_layers', 36)
    hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', 2048)
    return num_layers, hidden_size


def load_model(model_name: str, device: str = None):
    """Load model for activation extraction."""
    print(f"Loading model: {model_name}")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device != "cuda":
        model = model.to(device)

    model.eval()
    num_layers, hidden_size = get_model_config(model)
    print(f"Model loaded! Layers: {num_layers}, Hidden size: {hidden_size}")

    return model, tokenizer


def extract_activations_batch(
    model,
    tokenizer,
    texts: list[str],
    start_token: int = 50,
    layers: list[int] = None,
) -> dict[int, torch.Tensor]:
    """Extract residual stream activations for a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1

    if layers is None:
        layers = list(range(num_layers))

    layer_activations = {}

    for layer_idx in layers:
        layer_hidden = hidden_states[layer_idx + 1]
        batch_size, seq_len, hidden_dim = layer_hidden.shape

        positions = torch.arange(seq_len, device=model.device).unsqueeze(0)
        position_mask = positions >= start_token
        combined_mask = attention_mask * position_mask

        masked_hidden = layer_hidden * combined_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(dim=1)
        count = combined_mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_hidden = sum_hidden / count

        layer_activations[layer_idx] = avg_hidden.cpu()

    del inputs, outputs, hidden_states
    gc.collect()
    torch.cuda.empty_cache()

    return layer_activations


def extract_neutral_activations(
    model,
    tokenizer,
    dialogues: list[str],
    layers: list[int],
    batch_size: int = 16,
    start_token: int = 50,
    min_tokens: int = 100,
) -> dict[int, torch.Tensor]:
    """Extract activations from neutral dialogues."""

    # Filter dialogues by length
    filtered_dialogues = []
    for dialogue in dialogues:
        tokens = tokenizer.encode(dialogue, add_special_tokens=False)
        if len(tokens) >= min_tokens:
            filtered_dialogues.append(dialogue)

    print(f"Using {len(filtered_dialogues)}/{len(dialogues)} dialogues (>= {min_tokens} tokens)")

    layer_activations = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(filtered_dialogues), batch_size), desc="Extracting neutral activations"):
        batch = filtered_dialogues[i:i + batch_size]
        batch_acts = extract_activations_batch(model, tokenizer, batch, start_token, layers)

        for layer in layers:
            layer_activations[layer].append(batch_acts[layer])

        if (i // batch_size) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Concatenate
    for layer in layers:
        layer_activations[layer] = torch.cat(layer_activations[layer], dim=0)

    return layer_activations


def apply_pca_denoising(
    emotion_vectors: dict[str, dict[int, torch.Tensor]],
    neutral_activations: dict[int, torch.Tensor],
    variance_threshold: float = 0.5,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict]:
    """
    Project out top PCA components of neutral activations from emotion vectors.

    Returns denoised vectors and PCA info for each layer.
    """
    emotions = list(emotion_vectors.keys())
    layers = list(next(iter(emotion_vectors.values())).keys())

    denoised_vectors = {e: {} for e in emotions}
    pca_info = {}

    for layer in tqdm(layers, desc="Applying PCA denoising"):
        neutral_acts = neutral_activations[layer].float().numpy()

        # Fit PCA
        pca = PCA()
        pca.fit(neutral_acts)

        # Find components explaining variance_threshold
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)

        pca_info[layer] = {
            "n_components": n_components,
            "variance_explained": float(cumvar[n_components - 1]),
            "total_variance_ratio": pca.explained_variance_ratio_[:n_components].tolist(),
        }

        # Get projection matrix
        components = pca.components_[:n_components]

        # Project out from each emotion vector
        for emotion in emotions:
            vec = emotion_vectors[emotion][layer].float().numpy()

            # Project onto components and subtract
            projections = components @ vec
            projection_sum = components.T @ projections
            denoised = vec - projection_sum

            denoised_vectors[emotion][layer] = torch.from_numpy(denoised)

    return denoised_vectors, pca_info


def save_results(
    emotion_vectors: dict,
    output_path: str,
    metadata: dict = None,
):
    """Save emotion vectors to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    emotions = list(emotion_vectors.keys())
    layers = list(next(iter(emotion_vectors.values())).keys())

    for layer in layers:
        layer_data = {}
        for emotion in emotions:
            layer_data[emotion] = emotion_vectors[emotion][layer].float().numpy()

        np.savez(
            output_path / f"emotion_vectors_layer_{layer}.npz",
            **layer_data
        )

    meta = {
        "emotions": emotions,
        "layers": layers,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {}),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved emotion vectors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply PCA denoising to emotion vectors")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="emotion_vectors",
        help="Directory containing existing emotion vectors",
    )
    parser.add_argument(
        "--dialogues-dir",
        type=str,
        default="neutral_dialogues",
        help="Directory containing neutral dialogues",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="emotion_vectors_denoised",
        help="Output directory for denoised vectors",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.5,
        help="Variance threshold for PCA (default: 0.5 = 50%%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=50,
        help="Token position to start averaging from",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum token count for dialogue inclusion",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu)",
    )

    args = parser.parse_args()

    # Load existing emotion vectors
    print("Loading existing emotion vectors...")
    emotion_vectors, orig_metadata = load_emotion_vectors(args.vectors_dir)
    layers = orig_metadata["layers"]
    print(f"Loaded {len(emotion_vectors)} emotions, {len(layers)} layers")

    # Load neutral dialogues
    print(f"\nLoading neutral dialogues from {args.dialogues_dir}...")
    dialogues = load_neutral_dialogues(args.dialogues_dir)
    print(f"Loaded {len(dialogues)} dialogues")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Extract activations from neutral dialogues
    print("\nExtracting activations from neutral dialogues...")
    neutral_activations = extract_neutral_activations(
        model=model,
        tokenizer=tokenizer,
        dialogues=dialogues,
        layers=layers,
        batch_size=args.batch_size,
        start_token=args.start_token,
        min_tokens=args.min_tokens,
    )

    # Apply PCA denoising
    print(f"\nApplying PCA denoising (variance threshold: {args.variance_threshold})...")
    denoised_vectors, pca_info = apply_pca_denoising(
        emotion_vectors,
        neutral_activations,
        args.variance_threshold,
    )

    # Print PCA summary
    print("\nPCA components per layer:")
    sample_layers = [0, len(layers)//4, len(layers)//2, 3*len(layers)//4, len(layers)-1]
    for layer in sample_layers:
        if layer in pca_info:
            info = pca_info[layer]
            print(f"  Layer {layer}: {info['n_components']} components ({info['variance_explained']:.1%} variance)")

    # Save results
    metadata = {
        **orig_metadata,
        "pca_denoising": True,
        "pca_variance_threshold": args.variance_threshold,
        "pca_info": pca_info,
        "neutral_dialogues_dir": args.dialogues_dir,
        "n_neutral_dialogues": len(dialogues),
        "denoised_timestamp": datetime.now().isoformat(),
    }

    save_results(denoised_vectors, args.output_dir, metadata)

    print("\nDone!")
    print(f"Original vectors: {args.vectors_dir}")
    print(f"Denoised vectors: {args.output_dir}")


if __name__ == "__main__":
    main()
