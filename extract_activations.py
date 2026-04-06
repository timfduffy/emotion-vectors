"""
Activation Extraction Script for Emotion Vector Computation

Phase 2 of the Anthropic paper replication:
1. Load generated stories
2. Run through model, capturing residual stream activations
3. Average activations from token 50 onward
4. Compute emotion vectors (mean per emotion - global mean)
5. Optionally apply PCA denoising using neutral dialogues

This script requires HuggingFace transformers (not vLLM) since we need
access to intermediate hidden states.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime


def load_stories(stories_dir: str) -> dict:
    """Load generated stories from checkpoints directory."""
    checkpoint_dir = Path(stories_dir) / "checkpoints"

    if not checkpoint_dir.exists():
        # Try loading consolidated file
        all_stories_file = Path(stories_dir) / "all_stories.json"
        if all_stories_file.exists():
            with open(all_stories_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["stories"]
        raise FileNotFoundError(f"No stories found in {stories_dir}")

    stories = {}
    for checkpoint_file in tqdm(list(checkpoint_dir.glob("*.json")), desc="Loading stories"):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            emotion = data["emotion"]
            if emotion not in stories:
                stories[emotion] = {}
            stories[emotion][data["topic_idx"]] = data

    return stories


def get_model_config(model):
    """Get num_layers and hidden_size, handling different config formats."""
    config = model.config
    # Handle multimodal models with text_config
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

    # Ensure padding token exists
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
    """
    Extract residual stream activations for a batch of texts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text strings to process
        start_token: Token position to start averaging from (default: 50)
        layers: Which layers to extract (default: all)

    Returns:
        dict mapping layer index to tensor of shape (batch_size, hidden_dim)
    """
    import gc

    # Tokenize with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Get attention mask for proper averaging
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
    num_layers = len(hidden_states) - 1  # -1 because first is embeddings

    if layers is None:
        layers = list(range(num_layers))

    layer_activations = {}

    for layer_idx in layers:
        # +1 because hidden_states[0] is embeddings
        layer_hidden = hidden_states[layer_idx + 1]  # (batch, seq_len, hidden_dim)

        # Create mask for tokens from start_token onward
        batch_size, seq_len, hidden_dim = layer_hidden.shape

        # Build position mask
        positions = torch.arange(seq_len, device=model.device).unsqueeze(0)
        position_mask = positions >= start_token  # (1, seq_len)

        # Combine with attention mask
        combined_mask = attention_mask * position_mask  # (batch, seq_len)

        # Average activations where mask is 1
        masked_hidden = layer_hidden * combined_mask.unsqueeze(-1)  # (batch, seq_len, hidden_dim)
        sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
        count = combined_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        avg_hidden = sum_hidden / count  # (batch, hidden_dim)

        layer_activations[layer_idx] = avg_hidden.cpu()

    # Explicit cleanup
    del inputs, outputs, hidden_states
    gc.collect()
    torch.cuda.empty_cache()

    return layer_activations


def compute_emotion_vectors(
    activations_by_emotion: dict[str, dict[int, torch.Tensor]],
    layers: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Compute emotion vectors by subtracting mean across emotions.

    For each emotion and layer:
        emotion_vector = mean(activations for emotion) - mean(activations across all emotions)
    """
    emotions = list(activations_by_emotion.keys())

    # First compute mean per emotion per layer
    emotion_means = {}
    for emotion in emotions:
        emotion_means[emotion] = {}
        for layer in layers:
            # Stack all activations for this emotion and layer
            acts = activations_by_emotion[emotion][layer]  # (n_stories, hidden_dim)
            emotion_means[emotion][layer] = acts.mean(dim=0)  # (hidden_dim,)

    # Compute global mean per layer
    global_means = {}
    for layer in layers:
        all_means = torch.stack([emotion_means[e][layer] for e in emotions])
        global_means[layer] = all_means.mean(dim=0)

    # Compute emotion vectors (centered)
    emotion_vectors = {}
    for emotion in emotions:
        emotion_vectors[emotion] = {}
        for layer in layers:
            emotion_vectors[emotion][layer] = emotion_means[emotion][layer] - global_means[layer]

    return emotion_vectors


def filter_stories_by_length(
    stories: list[str],
    tokenizer,
    min_tokens: int = 100,
) -> list[str]:
    """Filter stories to only include those with at least min_tokens."""
    filtered = []
    for story in stories:
        tokens = tokenizer.encode(story, add_special_tokens=False)
        if len(tokens) >= min_tokens:
            filtered.append(story)
    return filtered


def extract_all_activations(
    model,
    tokenizer,
    stories: dict,
    batch_size: int = 32,
    start_token: int = 50,
    layers: list[int] = None,
    max_stories_per_emotion: int = None,
    min_tokens: int = 100,
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Extract activations for all stories, organized by emotion.

    Args:
        min_tokens: Minimum token count for a story to be included
                   (default: 100 = 50 start + 50 content)

    Returns:
        dict mapping emotion -> layer -> tensor of shape (n_stories, hidden_dim)
    """
    import gc

    num_layers, hidden_size = get_model_config(model)

    if layers is None:
        layers = list(range(num_layers))

    activations_by_emotion = {}
    total_stories = 0
    filtered_count = 0

    for emotion, topic_data in tqdm(stories.items(), desc="Processing emotions"):
        emotion_activations = {layer: [] for layer in layers}

        # Collect all stories for this emotion
        all_stories = []
        for topic_idx, data in topic_data.items():
            story_texts = data.get("stories", [])
            all_stories.extend(story_texts)

        original_count = len(all_stories)
        total_stories += original_count

        # Filter by minimum length
        all_stories = filter_stories_by_length(all_stories, tokenizer, min_tokens)
        filtered_count += (original_count - len(all_stories))

        if max_stories_per_emotion:
            all_stories = all_stories[:max_stories_per_emotion]

        if not all_stories:
            print(f"Warning: No stories found for emotion '{emotion}' after filtering")
            continue

        # Process in batches
        batch_count = 0
        for i in range(0, len(all_stories), batch_size):
            batch = all_stories[i:i + batch_size]
            batch_activations = extract_activations_batch(
                model, tokenizer, batch, start_token, layers
            )

            for layer in layers:
                emotion_activations[layer].append(batch_activations[layer])

            batch_count += 1

            # Memory cleanup every 10 batches
            if batch_count % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Concatenate all batches
        for layer in layers:
            if emotion_activations[layer]:
                emotion_activations[layer] = torch.cat(emotion_activations[layer], dim=0)
            else:
                emotion_activations[layer] = torch.zeros(0, hidden_size)

        activations_by_emotion[emotion] = emotion_activations

        # Memory cleanup after each emotion
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFiltered {filtered_count}/{total_stories} stories ({filtered_count/total_stories*100:.1f}%) for being < {min_tokens} tokens")

    return activations_by_emotion


def apply_pca_denoising(
    emotion_vectors: dict[str, dict[int, torch.Tensor]],
    neutral_activations: dict[int, torch.Tensor],
    variance_threshold: float = 0.5,
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Project out top PCA components of neutral activations from emotion vectors.

    This removes confounds unrelated to emotion (as described in the paper).
    """
    from sklearn.decomposition import PCA

    emotions = list(emotion_vectors.keys())
    layers = list(next(iter(emotion_vectors.values())).keys())

    denoised_vectors = {e: {} for e in emotions}

    for layer in tqdm(layers, desc="Applying PCA denoising"):
        neutral_acts = neutral_activations[layer].numpy()  # (n_neutral, hidden_dim)

        # Fit PCA to find components explaining variance_threshold of variance
        pca = PCA()
        pca.fit(neutral_acts)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumvar, variance_threshold) + 1

        # Get the projection matrix for these components
        components = pca.components_[:n_components]  # (n_components, hidden_dim)

        # Project out these components from each emotion vector
        for emotion in emotions:
            vec = emotion_vectors[emotion][layer].numpy()  # (hidden_dim,)

            # Project onto components and subtract
            projections = components @ vec  # (n_components,)
            projection_sum = (components.T @ projections)  # (hidden_dim,)
            denoised = vec - projection_sum

            denoised_vectors[emotion][layer] = torch.from_numpy(denoised)

    return denoised_vectors


def save_results(
    emotion_vectors: dict,
    output_path: str,
    metadata: dict = None,
):
    """Save emotion vectors to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays for each layer
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

    # Save metadata
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
    parser = argparse.ArgumentParser(
        description="Extract activations from generated stories"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--stories-dir",
        type=str,
        default="stories_output",
        help="Directory containing generated stories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="emotion_vectors",
        help="Output directory for emotion vectors"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (adjust for VRAM)"
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=50,
        help="Token position to start averaging from"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum token count for story inclusion (default: 100 = 50 start + 50 content)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to extract (default: all)"
    )
    parser.add_argument(
        "--max-stories-per-emotion",
        type=int,
        default=None,
        help="Limit stories per emotion (for testing)"
    )
    parser.add_argument(
        "--neutral-stories-dir",
        type=str,
        default=None,
        help="Directory with neutral dialogues for PCA denoising"
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.5,
        help="Variance threshold for PCA denoising (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--emotions-subset",
        type=str,
        nargs="+",
        default=None,
        help="Only process specific emotions (for testing)"
    )

    args = parser.parse_args()

    # Load stories
    print(f"Loading stories from {args.stories_dir}...")
    stories = load_stories(args.stories_dir)

    if args.emotions_subset:
        stories = {e: stories[e] for e in args.emotions_subset if e in stories}

    print(f"Found {len(stories)} emotions")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Determine layers
    num_layers, _ = get_model_config(model)
    if args.layers:
        layers = args.layers
    else:
        layers = list(range(num_layers))

    print(f"Extracting activations from {len(layers)} layers...")

    # Extract activations
    activations_by_emotion = extract_all_activations(
        model=model,
        tokenizer=tokenizer,
        stories=stories,
        batch_size=args.batch_size,
        start_token=args.start_token,
        layers=layers,
        max_stories_per_emotion=args.max_stories_per_emotion,
        min_tokens=args.min_tokens,
    )

    # Compute emotion vectors
    print("Computing emotion vectors...")
    emotion_vectors = compute_emotion_vectors(activations_by_emotion, layers)

    # Optional: PCA denoising
    if args.neutral_stories_dir:
        print("Applying PCA denoising...")
        neutral_stories = load_stories(args.neutral_stories_dir)

        # Extract activations from neutral stories
        neutral_activations = {}
        for layer in layers:
            neutral_activations[layer] = []

        for topic_data in neutral_stories.values():
            for data in topic_data.values():
                for story in data.get("stories", []):
                    batch_acts = extract_activations_batch(
                        model, tokenizer, [story], args.start_token, layers
                    )
                    for layer in layers:
                        neutral_activations[layer].append(batch_acts[layer])

        for layer in layers:
            neutral_activations[layer] = torch.cat(neutral_activations[layer], dim=0)

        emotion_vectors = apply_pca_denoising(
            emotion_vectors,
            neutral_activations,
            args.pca_variance,
        )

    # Save results
    metadata = {
        "model": args.model,
        "start_token": args.start_token,
        "batch_size": args.batch_size,
        "pca_denoising": args.neutral_stories_dir is not None,
        "pca_variance": args.pca_variance if args.neutral_stories_dir else None,
    }

    save_results(emotion_vectors, args.output_dir, metadata)

    print("\nDone!")
    print(f"Emotion vectors saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
