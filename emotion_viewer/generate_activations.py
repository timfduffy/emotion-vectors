"""
Generate text and save activations to a file.
Run this first, then use viewer.py to visualize.
"""

import torch
import numpy as np
from pathlib import Path
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"H:\Models\huggingface\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
OUTPUT_DIR = Path(__file__).parent / "activations"


def get_model_config(model):
    """Get model config handling Gemma 4 structure."""
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'num_layers', 36)
    hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', 2048)
    return num_layers, hidden_size


def load_model():
    """Load model and tokenizer."""
    print(f"Loading model: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    print("Model loaded!")
    return model, tokenizer


def generate_with_activations(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
):
    """Generate text and capture activations."""
    print(f"Generating response (max {max_new_tokens} tokens)...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Get tokens
    full_ids = outputs.sequences[0]
    tokens = [tokenizer.decode([tid]) for tid in full_ids]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)

    print(f"Generated {len(tokens)} tokens")

    # Collect activations
    num_layers, hidden_size = get_model_config(model)
    all_activations = []

    # First step: prompt tokens
    first_step_hidden = outputs.hidden_states[0]
    prompt_activations = torch.stack(
        [first_step_hidden[layer_idx][0] for layer_idx in range(1, len(first_step_hidden))],
        dim=0
    )
    all_activations.append(prompt_activations)

    # Subsequent steps: generated tokens
    for step_idx in range(1, len(outputs.hidden_states)):
        step_hidden = outputs.hidden_states[step_idx]
        token_activations = torch.stack(
            [step_hidden[layer_idx][0, -1:, :] for layer_idx in range(1, len(step_hidden))],
            dim=0
        )
        all_activations.append(token_activations)

    activations = torch.cat(all_activations, dim=1).float().cpu().numpy()
    print(f"Activations shape: {activations.shape}")

    return full_text, tokens, activations


def save_activations(prompt, full_text, tokens, activations, output_path):
    """Save activations to files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metadata as JSON
    metadata = {
        "prompt": prompt,
        "full_text": full_text,
        "tokens": tokens,
        "num_layers": activations.shape[0],
        "num_tokens": activations.shape[1],
        "hidden_size": activations.shape[2],
    }
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save activations as numpy
    np.save(output_path / "activations.npy", activations)

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate text and save activations")
    parser.add_argument("prompt", type=str, help="The prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output", type=str, default=None, help="Output directory name")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model()

    # Generate
    full_text, tokens, activations = generate_with_activations(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Save
    output_name = args.output or "latest"
    output_path = OUTPUT_DIR / output_name
    save_activations(args.prompt, full_text, tokens, activations, output_path)

    print("\nGenerated text:")
    print("-" * 40)
    print(full_text)
    print("-" * 40)
    print(f"\nRun 'python viewer.py' to visualize the activations")


if __name__ == "__main__":
    main()
