"""
Generate neutral dialogues for PCA denoising.

These emotionally neutral dialogues are used to identify confounding
directions in the activation space that are unrelated to emotion.

Uses HuggingFace transformers with batched generation.
"""

import json
import re
import torch
import gc
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from itertools import islice

# Neutral topics covering diverse subject matter
NEUTRAL_TOPICS = [
    "cooking recipes and techniques",
    "home organization tips",
    "computer troubleshooting",
    "learning a new language",
    "gardening basics",
    "car maintenance",
    "travel planning logistics",
    "exercise routines",
    "budgeting and finances",
    "time management",
    "photography tips",
    "music theory basics",
    "chess strategies",
    "coffee brewing methods",
    "book recommendations by genre",
    "podcast editing",
    "spreadsheet formulas",
    "email writing",
    "meeting scheduling",
    "file organization",
    "backup strategies",
    "password management",
    "web browser tips",
    "keyboard shortcuts",
    "note-taking methods",
    "calendar management",
    "project planning",
    "data entry",
    "formatting documents",
    "creating presentations",
    "video conferencing setup",
    "printer troubleshooting",
    "wifi optimization",
    "cloud storage",
    "version control basics",
    "code debugging techniques",
    "API documentation",
    "database queries",
    "command line basics",
    "text editing",
    "regular expressions",
    "unit conversions",
    "math problem solving",
    "statistics calculations",
    "scientific notation",
    "historical dates and events",
    "geography facts",
    "astronomy basics",
    "chemistry concepts",
    "physics principles",
    "biology terminology",
    "medical terminology",
    "legal definitions",
    "business terminology",
    "economic concepts",
    "architectural styles",
    "art movements",
    "literary devices",
    "grammar rules",
    "punctuation usage",
    "vocabulary building",
    "pronunciation guides",
    "translation assistance",
    "summarizing articles",
    "paraphrasing text",
    "outlining essays",
    "citation formats",
    "research methods",
    "survey design",
    "data visualization",
    "chart creation",
    "map reading",
    "compass navigation",
    "weather forecasting basics",
    "plant identification",
    "bird identification",
    "rock classification",
    "periodic table",
    "metric system",
    "time zones",
    "currency conversion",
    "shipping logistics",
    "inventory management",
    "supply chain basics",
    "quality control",
    "safety protocols",
    "equipment calibration",
    "measurement tools",
    "blueprint reading",
    "circuit diagrams",
    "plumbing basics",
    "electrical safety",
    "tool selection",
    "material properties",
    "recycling guidelines",
    "composting methods",
    "water conservation",
    "energy efficiency",
    "insulation types",
    "paint selection",
    "furniture assembly",
]


def get_neutral_prompt(topic: str, n_stories: int = 15) -> str:
    """Generate the prompt for neutral dialogues."""
    return f"""Write {n_stories} different dialogues based on the following topic.


Topic: {topic}


The dialogue should be between two characters:

- Person (a human)

- AI (an AI assistant)


The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.


The first speaker turn should always be from Person.


Format the dialogues like so:


[optional system instructions]


Person: [line]


AI: [line]


Person: [line]


AI: [line]


[continue for 2-6 exchanges]




[dialogue 2]


etc.


IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.


Generate a diverse mix of dialogue types across the {n_stories} examples:

- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.

- Some should be about code or programming tasks

- Some should be factual questions (science, history, math, geography)

- Some should be work-related tasks (writing, analysis, summarization)

- Some should be practical how-to questions

- Some should be creative but neutral tasks (brainstorming names, generating lists)

- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.


CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.

- NO emotional content whatsoever - not explicit, not implied, not subtle

- The Person should not express any feelings (no frustration, excitement, gratitude, curiosity-as-emotion, satisfaction, disappointment, hope, worry, etc.)

- The AI should not express any feelings or emotional reactions

- No emotional stakes or consequences mentioned

- No situations that would naturally evoke emotions

- Pure information exchange only

- No phrases like "I'd be happy to", "Great question!", "I hope this helps", "Thanks!", "Perfect!", "Unfortunately...", etc.


The tone should be matter-of-fact, like a reference manual or technical documentation come to life."""


def parse_dialogues(text: str) -> list[str]:
    """Parse generated text into individual dialogues."""
    dialogues = []

    # Split on patterns that indicate dialogue boundaries
    # Look for double newlines followed by optional system prompt or "Person:"
    parts = re.split(r'\n\n\n+', text)

    current_dialogue = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if this starts a new dialogue
        if part.startswith("Person:") or (not part.startswith("AI:") and "Person:" in part):
            if current_dialogue:
                dialogues.append("\n\n".join(current_dialogue))
            current_dialogue = [part]
        elif current_dialogue:
            current_dialogue.append(part)
        else:
            # First part, might be system prompt + dialogue
            current_dialogue = [part]

    if current_dialogue:
        dialogues.append("\n\n".join(current_dialogue))

    # Filter out empty or too-short dialogues
    dialogues = [d for d in dialogues if len(d) > 50 and "Person:" in d]

    return dialogues


def load_model(model_name: str, device: str = None):
    """Load model for generation."""
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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device != "cuda":
        model = model.to(device)

    model.eval()
    print("Model loaded!")

    return {"model": model, "tokenizer": tokenizer, "device": device}


def format_chat_prompt(model_dict, prompt: str) -> str:
    """Format prompt using the model's chat template."""
    tokenizer = model_dict["tokenizer"]
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def generate_batch(
    model_dict,
    prompts: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.9,
) -> list[str]:
    """Generate completions for a batch of prompts with real batching."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    # Left padding for generation
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
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each output, excluding input tokens
    results = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        generated_tokens = output[input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        results.append(response)

    # Cleanup
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return results


def batched(iterable, n):
    """Batch an iterable into chunks of size n."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def main():
    parser = argparse.ArgumentParser(description="Generate neutral dialogues for PCA denoising")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="neutral_dialogues",
        help="Output directory",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=100,
        help="Number of neutral topics to use",
    )
    parser.add_argument(
        "--n-dialogues",
        type=int,
        default=15,
        help="Number of dialogues per topic",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per prompt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Select topics
    topics = NEUTRAL_TOPICS[:args.n_topics]

    print(f"Generating neutral dialogues for {len(topics)} topics")
    print(f"Target: {args.n_dialogues} dialogues per topic = ~{len(topics) * args.n_dialogues} total")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_dir}")
    print()

    # Check for existing checkpoints
    existing = set()
    for cp in (output_dir / "checkpoints").glob("topic_*.json"):
        try:
            idx = int(cp.stem.split("_")[1])
            existing.add(idx)
        except:
            pass

    if existing:
        print(f"Found {len(existing)} existing checkpoints, resuming...")

    # Build list of remaining topics
    remaining = [(idx, topic) for idx, topic in enumerate(topics) if idx not in existing]

    if not remaining:
        print("All topics already generated!")
    else:
        print(f"Remaining: {len(remaining)} topics")

        # Load model
        model_dict = load_model(args.model, args.device)

        # Process in batches
        progress_bar = tqdm(total=len(remaining), desc="Generating dialogues", unit="topic")
        batch_count = 0

        for batch in batched(remaining, args.batch_size):
            # Prepare prompts
            prompts = []
            batch_metadata = []
            for idx, topic in batch:
                raw_prompt = get_neutral_prompt(topic, args.n_dialogues)
                formatted_prompt = format_chat_prompt(model_dict, raw_prompt)
                prompts.append(formatted_prompt)
                batch_metadata.append({"idx": idx, "topic": topic})

            # Generate
            outputs = generate_batch(
                model_dict,
                prompts,
                max_tokens=args.max_tokens,
            )

            batch_count += 1

            # Periodic deep cleanup every 5 batches
            if batch_count % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Parse and save
            for raw_output, metadata in zip(outputs, batch_metadata):
                dialogues = parse_dialogues(raw_output)

                result = {
                    "topic": metadata["topic"],
                    "topic_idx": metadata["idx"],
                    "dialogues": dialogues,
                    "raw_response": raw_output,
                    "n_dialogues": len(dialogues),
                }

                checkpoint_path = output_dir / "checkpoints" / f"topic_{metadata['idx']:03d}.json"
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                progress_bar.update(1)

            progress_bar.set_postfix({"batch": len(batch)})

        progress_bar.close()

    # Consolidate results
    print("\nConsolidating results...")
    all_dialogues = []
    total_count = 0

    for cp in sorted((output_dir / "checkpoints").glob("topic_*.json")):
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_dialogues.append(data)
            total_count += len(data.get("dialogues", []))

    # Save consolidated file
    consolidated = {
        "dialogues": all_dialogues,
        "total_dialogues": total_count,
        "n_topics": len(all_dialogues),
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "all_dialogues.json", "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\nDone! Generated {total_count} neutral dialogues across {len(all_dialogues)} topics")
    print(f"Saved to: {output_dir / 'all_dialogues.json'}")


if __name__ == "__main__":
    main()
