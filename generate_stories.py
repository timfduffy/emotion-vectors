"""
Story Generation Script for Emotion Vector Extraction

This script generates emotional stories using google/gemma-4-E2B-it model,
replicating the methodology from Anthropic's paper on emotion concepts in LLMs.

Uses vLLM on Linux for efficient batched inference, falls back to HuggingFace
transformers on Windows.
"""

import os
import sys
import json
import platform
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
from itertools import islice

# Backend will be set after argument parsing
BACKEND = None
IS_LINUX = platform.system() == "Linux"


def init_backend(force_backend: str = None):
    """Initialize the backend (vllm or transformers)."""
    global BACKEND

    if force_backend:
        BACKEND = force_backend
    elif IS_LINUX:
        try:
            from vllm import LLM, SamplingParams
            BACKEND = "vllm"
        except ImportError:
            BACKEND = "transformers"
    else:
        BACKEND = "transformers"

    if BACKEND == "vllm":
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return None, None


# Story generation prompt template from the paper's appendix
STORY_PROMPT_TEMPLATE = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


def load_emotions(filepath: str = "emotions.txt") -> list[str]:
    """Load emotion concepts from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        emotions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(emotions)} emotions")
    return emotions


def load_topics(filepath: str = "topics.txt") -> list[str]:
    """Load story topics from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(topics)} topics")
    return topics


# =============================================================================
# vLLM Backend
# =============================================================================

def load_model_vllm(model_name: str, **kwargs):
    """Load model using vLLM."""
    from vllm import LLM

    print(f"Loading model with vLLM: {model_name}")

    default_kwargs = {
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": True,
    }
    default_kwargs.update(kwargs)

    llm = LLM(model=model_name, **default_kwargs)
    print("Model loaded successfully!")
    return llm


def format_chat_prompt_vllm(llm, prompt: str) -> str:
    """Format prompt using the model's chat template (vLLM)."""
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    # Try to disable thinking mode for Qwen models
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for models that don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def generate_batch_vllm(llm, prompts: list[str], sampling_params) -> list[str]:
    """Generate completions for a batch of prompts using vLLM."""
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


# =============================================================================
# HuggingFace Transformers Backend
# =============================================================================

def load_model_transformers(model_name: str, device: str = None):
    """Load model using HuggingFace transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model with transformers: {model_name}")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device != "cuda":
        model = model.to(device)

    model.eval()
    print("Model loaded successfully!")

    return {"model": model, "tokenizer": tokenizer, "device": device}


def format_chat_prompt_transformers(model_dict, prompt: str) -> str:
    """Format prompt using the model's chat template (transformers)."""
    tokenizer = model_dict["tokenizer"]
    messages = [{"role": "user", "content": prompt}]
    # Try to disable thinking mode for Qwen models
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for models that don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def generate_batch_transformers(
    model_dict,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float
) -> list[str]:
    """Generate completions for a batch of prompts using transformers with real batching."""
    import torch
    import gc

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    # Ensure pad token is set for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize all prompts with padding (left padding for generation)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each output, excluding the input tokens
    results = []
    input_len = inputs["input_ids"].shape[1]  # All inputs same length due to padding
    for i, output in enumerate(outputs):
        # Get only the generated tokens (skip input length for this sequence)
        generated_tokens = output[input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        results.append(response)

    # Explicit cleanup to prevent memory accumulation
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# Common Functions
# =============================================================================

def create_prompt(emotion: str, topic: str, n_stories: int = 12) -> str:
    """Create a formatted prompt for story generation."""
    return STORY_PROMPT_TEMPLATE.format(
        n_stories=n_stories,
        topic=topic,
        emotion=emotion
    )


def parse_stories(raw_text: str) -> list[str]:
    """Parse the generated text into individual stories."""
    stories = []
    current_story = []

    for line in raw_text.split('\n'):
        stripped = line.strip().lower()
        if stripped.startswith('[story') and stripped.endswith(']'):
            if current_story:
                story_text = '\n'.join(current_story).strip()
                if story_text:
                    stories.append(story_text)
                current_story = []
        else:
            current_story.append(line)

    if current_story:
        story_text = '\n'.join(current_story).strip()
        if story_text:
            stories.append(story_text)

    return stories


def save_checkpoint(data: dict, output_dir: str, emotion: str, topic_idx: int):
    """Save intermediate results as checkpoint."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    safe_emotion = emotion.replace(" ", "_").replace("-", "_")
    checkpoint_file = checkpoint_dir / f"{safe_emotion}_topic_{topic_idx}.json"

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_progress(output_dir: str) -> set:
    """Load set of completed (emotion, topic_idx) pairs."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    completed = set()

    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.json"):
            parts = f.stem.rsplit("_topic_", 1)
            if len(parts) == 2:
                emotion = parts[0].replace("_", " ")
                topic_idx = int(parts[1])
                completed.add((emotion, topic_idx))

    return completed


def batched(iterable, n):
    """Batch an iterable into chunks of size n."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def generate_all_pairs(emotions: list[str], topics: list[str], completed: set):
    """Generate all (emotion, topic_idx, topic) tuples not yet completed."""
    for emotion in emotions:
        for topic_idx, topic in enumerate(topics):
            if (emotion, topic_idx) not in completed:
                yield (emotion, topic_idx, topic)


def consolidate_results(output_dir: str, emotions: list, topics: list, args):
    """Consolidate all checkpoint files into a single output file."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    all_stories = {}

    for emotion in emotions:
        emotion_stories = {}
        safe_emotion = emotion.replace(" ", "_").replace("-", "_")

        for topic_idx in range(len(topics)):
            checkpoint_file = checkpoint_dir / f"{safe_emotion}_topic_{topic_idx}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    emotion_stories[topic_idx] = json.load(f)

        if emotion_stories:
            all_stories[emotion] = emotion_stories

    final_output = {
        "metadata": {
            "model": args.model,
            "backend": BACKEND,
            "n_emotions": len(emotions),
            "n_topics": len(topics),
            "n_stories_per_pair": args.n_stories,
            "timestamp": datetime.now().isoformat(),
        },
        "emotions": emotions,
        "topics": topics,
        "stories": all_stories,
    }

    output_file = Path(output_dir) / "all_stories.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotional stories for emotion vector extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HuggingFace model name or local path (default: google/gemma-4-E2B-it)"
    )
    parser.add_argument(
        "--emotions-file",
        type=str,
        default="emotions.txt",
        help="Path to emotions file"
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default="topics.txt",
        help="Path to topics file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="stories_output",
        help="Output directory for generated stories"
    )
    parser.add_argument(
        "--n-stories",
        type=int,
        default=12,
        help="Number of stories per topic per emotion (default: 12)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per request"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to process in each batch (default: 8)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--emotions-subset",
        type=str,
        nargs="+",
        default=None,
        help="Only generate for specific emotions (for testing)"
    )
    parser.add_argument(
        "--topics-subset",
        type=int,
        default=None,
        help="Only use first N topics (for testing)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (vLLM only, default: 0.90)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM only)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (transformers only: cuda, mps, cpu)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers", "auto"],
        default="auto",
        help="Backend to use (default: auto)"
    )

    args = parser.parse_args()

    # Initialize backend
    force_backend = None if args.backend == "auto" else args.backend
    init_backend(force_backend)

    print(f"Using backend: {BACKEND}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load emotions and topics
    emotions = load_emotions(args.emotions_file)
    topics = load_topics(args.topics_file)

    # Apply subsets if specified
    if args.emotions_subset:
        emotions = [e for e in emotions if e in args.emotions_subset]
        print(f"Using emotion subset: {emotions}")

    if args.topics_subset:
        topics = topics[:args.topics_subset]
        print(f"Using first {len(topics)} topics")

    # Check for existing progress
    completed = set()
    if args.resume:
        completed = load_progress(args.output_dir)
        print(f"Resuming: found {len(completed)} completed (emotion, topic) pairs")

    # Calculate total work
    total_pairs = len(emotions) * len(topics)
    remaining_pairs = total_pairs - len(completed)
    print(f"\nTotal: {total_pairs} (emotion, topic) pairs")
    print(f"Remaining: {remaining_pairs} pairs")
    print(f"Stories per pair: {args.n_stories}")
    print(f"Total stories to generate: {remaining_pairs * args.n_stories}")

    if remaining_pairs == 0:
        print("Nothing to do - all pairs already completed!")
        return

    # Load model based on backend
    if BACKEND == "vllm":
        from vllm import SamplingParams

        print(f"Batch size: {args.batch_size}")
        model = load_model_vllm(
            args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        print(f"Batch size: {args.batch_size}")
        model = load_model_transformers(args.model, args.device)

    # Generate all remaining pairs
    all_pairs = list(generate_all_pairs(emotions, topics, completed))

    progress_bar = tqdm(
        total=remaining_pairs,
        desc="Generating stories",
        unit="pair"
    )

    if BACKEND == "vllm":
        # Process in batches with vLLM
        for batch in batched(all_pairs, args.batch_size):
            prompts = []
            batch_metadata = []

            for emotion, topic_idx, topic in batch:
                raw_prompt = create_prompt(emotion, topic, args.n_stories)
                formatted_prompt = format_chat_prompt_vllm(model, raw_prompt)
                prompts.append(formatted_prompt)
                batch_metadata.append({
                    "emotion": emotion,
                    "topic_idx": topic_idx,
                    "topic": topic,
                })

            outputs = generate_batch_vllm(model, prompts, sampling_params)

            for raw_output, metadata in zip(outputs, batch_metadata):
                stories = parse_stories(raw_output)

                result = {
                    "emotion": metadata["emotion"],
                    "topic": metadata["topic"],
                    "topic_idx": metadata["topic_idx"],
                    "raw_output": raw_output,
                    "stories": stories,
                    "n_stories_generated": len(stories),
                    "timestamp": datetime.now().isoformat(),
                }

                save_checkpoint(
                    result,
                    args.output_dir,
                    metadata["emotion"],
                    metadata["topic_idx"]
                )
                progress_bar.update(1)

            if batch_metadata:
                last = batch_metadata[-1]
                progress_bar.set_postfix({
                    "emotion": last["emotion"][:12],
                    "batch": len(batch)
                })
    else:
        # Process in batches with transformers
        import gc
        import torch
        batch_count = 0

        for batch in batched(all_pairs, args.batch_size):
            prompts = []
            batch_metadata = []

            for emotion, topic_idx, topic in batch:
                raw_prompt = create_prompt(emotion, topic, args.n_stories)
                formatted_prompt = format_chat_prompt_transformers(model, raw_prompt)
                prompts.append(formatted_prompt)
                batch_metadata.append({
                    "emotion": emotion,
                    "topic_idx": topic_idx,
                    "topic": topic,
                })

            outputs = generate_batch_transformers(
                model,
                prompts,
                args.max_tokens,
                args.temperature,
                args.top_p,
            )

            batch_count += 1

            # Deep cleanup every 10 batches to prevent memory creep
            if batch_count % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            for raw_output, metadata in zip(outputs, batch_metadata):
                stories = parse_stories(raw_output)

                result = {
                    "emotion": metadata["emotion"],
                    "topic": metadata["topic"],
                    "topic_idx": metadata["topic_idx"],
                    "raw_output": raw_output,
                    "stories": stories,
                    "n_stories_generated": len(stories),
                    "timestamp": datetime.now().isoformat(),
                }

                save_checkpoint(
                    result,
                    args.output_dir,
                    metadata["emotion"],
                    metadata["topic_idx"]
                )
                progress_bar.update(1)

            if batch_metadata:
                last = batch_metadata[-1]
                progress_bar.set_postfix({
                    "emotion": last["emotion"][:12],
                    "batch": len(batch)
                })

    progress_bar.close()

    # Consolidate all checkpoints into final output
    print("\nConsolidating results...")
    consolidate_results(args.output_dir, emotions, topics, args)

    print(f"\nGeneration complete!")
    print(f"Output saved to: {output_dir / 'all_stories.json'}")


if __name__ == "__main__":
    main()
