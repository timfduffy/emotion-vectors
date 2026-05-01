"""
Steering Chat Application

Supports two modes:
- self-steer: Model controls its own steering via tool calls
- user-steer: User controls steering via UI
"""

import time
_import_start = time.perf_counter()

import gradio as gr
print(f"[Startup] Gradio import: {time.perf_counter() - _import_start:.2f}s")

_t = time.perf_counter()
import torch
print(f"[Startup] Torch import: {time.perf_counter() - _t:.2f}s")

from pathlib import Path

_t = time.perf_counter()
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[Startup] Transformers import: {time.perf_counter() - _t:.2f}s")

from typing import Optional
import json

_t = time.perf_counter()
from steering import SteeringManager, SteeringConfig
print(f"[Startup] Steering import: {time.perf_counter() - _t:.2f}s")

from tools import parse_tool_call, format_tool_result, STEER_TOOL_SCHEMA
from prompts import get_system_prompt

print(f"[Startup] Total imports: {time.perf_counter() - _import_start:.2f}s")


# Configuration
MODEL_PATH = r"H:\Models\huggingface\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
PROJECT_ROOT = Path(__file__).parent.parent
VECTORS_DIR = PROJECT_ROOT / "emotion_vectors_denoised"

# Default vector sources (label -> path). Sources whose path lacks a
# metadata.json are silently skipped at load time.
DEFAULT_SOURCES: list[tuple[str, Path]] = [
    ("Emotions", PROJECT_ROOT / "emotion_vectors_denoised"),
    ("PCA components", PROJECT_ROOT / "pca_vectors"),
]

# Concept section shown to the LLM in self-steer mode for each source.
# This populates the "Available concepts" portion of the system prompt.
CONCEPT_SECTIONS: dict[str, str] = {
    "Emotions": (
        "### Available emotions:\n\n"
        "**Positive:** happy, confident, hopeful, compassionate, grateful\n"
        "**Negative:** sad, anxious, frustrated, angry, fearful\n"
        "**Calm/Neutral:** calm, peaceful, patient, neutral, serene\n"
        "**Engagement:** curious, focused, enthusiastic, playful, assertive"
    ),
    "PCA components": (
        "### Available concepts (principal component axes of variation):\n\n"
        "These are PCA-derived axes from a large set of emotion vectors. They are\n"
        "broader than individual emotions and capture orthogonal dimensions of\n"
        "affective space.\n\n"
        "- **pc1_valence**: positive vs negative emotional polarity\n"
        "  (positive strength = happier/more positive; negative = sadder/angrier)\n"
        "- **pc2_arousal**: high vs low arousal/energy\n"
        "  (positive = energetic/excited; negative = calm/sleepy)\n"
        "- **pc3, pc4, pc5**: subtler residual axes; effects are less interpretable"
    ),
}

# Curated whitelists per source (the concepts the model is allowed to
# pick in self-steer mode). For PCA we expose all components since there
# are only a handful.
CURATED_PER_SOURCE: dict[str, list[str]] = {
    "Emotions": [
        "happy", "sad", "calm", "curious",
        "confident", "anxious", "peaceful", "focused",
        "hopeful", "frustrated", "patient", "enthusiastic",
        "compassionate", "angry", "neutral", "playful",
        "grateful", "fearful", "serene", "assertive",
    ],
    "PCA components": ["pc1_valence", "pc2_arousal", "pc3", "pc4", "pc5"],
}

# Global state
model = None
tokenizer = None
steering_manager = None

# KV cache state
cached_input_ids = None  # Token IDs that the cache was built from
cached_kv = None  # The actual past_key_values


def load_model_and_steering(config: SteeringConfig):
    """Load the model and initialize steering."""
    global model, tokenizer, steering_manager

    t0 = time.perf_counter()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Startup] Tokenizer loaded: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[Startup] Model loaded: {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    print("Initializing steering...")
    # Filter to sources that actually exist on disk
    config.sources = [
        (label, str(path)) for label, path in DEFAULT_SOURCES
        if (path / "metadata.json").exists()
    ]
    config.curated_per_source = dict(CURATED_PER_SOURCE)
    if not config.sources:
        # Fallback to legacy single-source behavior
        config.vectors_dir = str(VECTORS_DIR)
    steering_manager = SteeringManager(model, config)
    steering_manager.register_hooks()
    print(f"[Startup] Steering initialized: {time.perf_counter() - t2:.2f}s")


def do_prefill(input_ids, attention_mask, past_key_values=None):
    """Run a forward pass to build/extend KV cache without generating."""
    global model
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return outputs.past_key_values


def generate_response(
    messages: list,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    assistant_only_steering: bool = False,
) -> tuple[str, bool, Optional[dict]]:
    """
    Generate a response, checking for tool calls.
    Uses KV cache to avoid reprocessing previous tokens.

    Args:
        messages: List of message dicts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        assistant_only_steering: If True, disable steering during prefill of user message

    Returns:
        - response text
        - whether a tool was called
        - tool call info (if any)
    """
    global model, tokenizer, steering_manager, cached_input_ids, cached_kv

    timings = {}
    t0 = time.perf_counter()

    # Format messages with chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    t1 = time.perf_counter()
    timings["chat_template"] = t1 - t0

    tokenized = tokenizer(formatted, return_tensors="pt")
    full_input_ids = tokenized["input_ids"].to(model.device)
    full_attention_mask = tokenized["attention_mask"].to(model.device)
    full_length = full_input_ids.shape[1]

    t2 = time.perf_counter()
    timings["tokenize"] = t2 - t1

    # Check if we can reuse KV cache
    use_cache_from = None
    past_kv_to_use = None

    if cached_input_ids is not None and cached_kv is not None:
        cached_len = cached_input_ids.shape[1]
        if full_length > cached_len:
            # Check if the prefix matches
            prefix_matches = torch.equal(
                full_input_ids[0, :cached_len],
                cached_input_ids[0, :cached_len]
            )
            if prefix_matches:
                use_cache_from = cached_len
                past_kv_to_use = cached_kv
                print(f"  [KV Cache] Reusing {cached_len} tokens, processing {full_length - cached_len} new tokens")

    t3 = time.perf_counter()
    timings["cache_check"] = t3 - t2

    # Reset per-generation logging
    steering_manager.reset_generation_log()

    # Prepare inputs
    if use_cache_from is not None:
        # Only pass new tokens, reuse cache for prefix
        input_ids = full_input_ids[:, use_cache_from:]
        attention_mask = full_attention_mask  # Full mask needed for attention
        past_key_values = past_kv_to_use
    else:
        # Process full input
        input_ids = full_input_ids
        attention_mask = full_attention_mask
        past_key_values = None
        print(f"  [KV Cache] Processing full {full_length} tokens (no cache reuse)")

    t4 = time.perf_counter()
    timings["prep_inputs"] = t4 - t3

    # Handle assistant-only steering: prefill without steering, then generate with steering
    if assistant_only_steering and steering_manager.state.active:
        # Save current steering state
        saved_emotion = steering_manager.state.emotion
        saved_strength = steering_manager.state.strength

        # Turn off steering for prefill
        steering_manager.state.active = False
        print(f"  [Steering] Disabled for prefill (assistant-only mode)")

        # Do prefill to build KV cache for the prompt
        past_key_values = do_prefill(input_ids, attention_mask, past_key_values)

        # Restore steering for generation
        steering_manager.state.active = True
        steering_manager.state.emotion = saved_emotion
        steering_manager.state.strength = saved_strength
        print(f"  [Steering] Re-enabled for generation: {saved_emotion} @ {saved_strength}")

        # Now generate with empty input (just use KV cache)
        # We need to pass a dummy token to start generation
        input_ids = full_input_ids[:, -1:]  # Just the last token
        # Attention mask should cover all previous tokens
        attention_mask = full_attention_mask

    t5 = time.perf_counter()
    timings["prefill"] = t5 - t4

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )

    t6 = time.perf_counter()
    timings["generate"] = t6 - t5

    # Update KV cache for next turn
    # The new cache includes all tokens (prefix + generated)
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        cached_kv = outputs.past_key_values
        # Reconstruct full input_ids including generated tokens
        sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs[0]
        if use_cache_from is not None:
            # Combine original prefix with new sequence
            cached_input_ids = torch.cat([
                full_input_ids[:, :use_cache_from],
                sequences
            ], dim=1)
        else:
            cached_input_ids = torch.cat([
                full_input_ids,
                sequences[:, input_ids.shape[1]:]  # Only the generated part
            ], dim=1) if assistant_only_steering else sequences

    t7 = time.perf_counter()
    timings["cache_update"] = t7 - t6

    # Decode only the new tokens
    sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs[0]
    new_tokens = sequences[0][input_ids.shape[1]:]  # Tokens after what we passed in
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    t8 = time.perf_counter()
    timings["decode"] = t8 - t7

    # Check for tool calls
    tool_call = parse_tool_call(response)

    t9 = time.perf_counter()
    timings["parse_tool"] = t9 - t8
    timings["total"] = t9 - t0

    # Print timing summary
    gen_tokens = len(new_tokens)
    tokens_per_sec = gen_tokens / timings["generate"] if timings["generate"] > 0 else 0
    print(f"  [Timing] total={timings['total']:.2f}s, generate={timings['generate']:.2f}s "
          f"({gen_tokens} tokens, {tokens_per_sec:.1f} tok/s), "
          f"tokenize={timings['tokenize']*1000:.1f}ms, decode={timings['decode']*1000:.1f}ms")

    return response, tool_call is not None, tool_call


def execute_tool_call(tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result."""
    global steering_manager

    if tool_name == "steer":
        emotion = args.get("emotion", "none")
        strength = float(args.get("strength", 0.0))

        try:
            steering_manager.set_steering(emotion, strength, validate_curated=True)
            return format_tool_result(emotion, strength, success=True)
        except ValueError as e:
            return format_tool_result(emotion, strength, success=False, message=str(e))

    return f"[Unknown tool: {tool_name}]"


def chat_self_steer(
    message: str,
    history: list,
    max_tokens: int,
    temperature: float,
    custom_additions: str = "",
    concept_section: str = "",
):
    """Chat function for self-steer mode."""
    global steering_manager

    # Build message list with system prompt (steering-tool instructions
    # plus any user-supplied additions appended after them). The concept
    # section reflects the active vector source's available concepts.
    sys_prompt_kwargs = {"custom_additions": custom_additions}
    if concept_section:
        sys_prompt_kwargs["concept_section"] = concept_section
    messages = [{"role": "system", "content": get_system_prompt(**sys_prompt_kwargs)}]

    # Add history (Gradio 6 messages format: list of {"role", "content"} dicts)
    for entry in history:
        if entry.get("content"):
            messages.append({"role": entry["role"], "content": entry["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Generate response
    response, has_tool_call, tool_info = generate_response(
        messages, max_new_tokens=max_tokens, temperature=temperature
    )

    # If there's a tool call, execute it and optionally continue generation
    if has_tool_call and tool_info:
        tool_name, tool_args = tool_info
        tool_result = execute_tool_call(tool_name, tool_args)

        # Add tool result to the response
        response = response.strip() + "\n\n" + tool_result

    return response


def chat_user_steer(
    message: str,
    history: list,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    assistant_only: bool,
):
    """Chat function for user-steer mode."""
    global steering_manager

    # Build message list with optional system prompt
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Add history (Gradio 6 messages format: list of {"role", "content"} dicts)
    for entry in history:
        if entry.get("content"):
            messages.append({"role": entry["role"], "content": entry["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Generate response with assistant_only flag
    response, _, _ = generate_response(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
        assistant_only_steering=assistant_only,
    )

    return response


def get_steering_status():
    """Get current steering status for display."""
    global steering_manager
    if steering_manager is None:
        return "Steering not initialized"

    status = steering_manager.get_status()
    layer_start, layer_end = steering_manager.get_layer_range()
    layer_info = f"Layers {layer_start}-{layer_end}"

    if not status["active"]:
        return f"**Steering:** Inactive (baseline processing)\n\n**{layer_info}**"
    else:
        direction = "toward" if status["strength"] > 0 else "away from"
        return f"**Steering:** {abs(status['strength']):.2f}x {direction} **{status['emotion']}**\n\n**{layer_info}**"


def create_self_steer_app(config: SteeringConfig):
    """Create the self-steer mode Gradio app."""

    # Load model and steering
    load_model_and_steering(config)

    with gr.Blocks(title="Self-Steering Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Self-Steering Chat")
        gr.Markdown(
            "Chat with an AI that can adjust its own emotional processing using steering vectors. "
            "The model can call `steer(emotion, strength)` to modify how it generates responses."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                max_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature"
                )

                gr.Markdown("### Layer Range")
                num_layers = steering_manager.get_num_model_layers()
                layer_start_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, value=config.layer_start, step=1,
                    label="Start Layer"
                )
                layer_end_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, value=config.layer_end, step=1,
                    label="End Layer"
                )

                gr.Markdown("### Steering Status")
                steering_display = gr.Markdown(get_steering_status())
                refresh_status = gr.Button("Refresh Status")

                gr.Markdown("### Available Emotions")
                gr.Markdown(
                    "**Positive:** happy, confident, hopeful, compassionate, grateful\n\n"
                    "**Negative:** sad, anxious, frustrated, angry, fearful\n\n"
                    "**Calm:** calm, peaceful, patient, neutral, serene\n\n"
                    "**Engagement:** curious, focused, enthusiastic, playful, assertive"
                )

        def respond(message, history, max_tok, temp):
            if not message.strip():
                return history, "", get_steering_status()

            response = chat_self_steer(message, history, max_tok, temp)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return history, "", get_steering_status()

        def clear_chat():
            global steering_manager, cached_input_ids, cached_kv
            if steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            # Clear KV cache
            cached_input_ids = None
            cached_kv = None
            print("  [KV Cache] Cleared")
            return [], "", get_steering_status()

        def on_chatbot_change(history):
            """Reset steering when chatbot is cleared (e.g., via built-in trash icon)."""
            global steering_manager, cached_input_ids, cached_kv
            if not history and steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
                # Clear KV cache
                cached_input_ids = None
                cached_kv = None
                print("  [KV Cache] Cleared (chat emptied)")
            return get_steering_status()

        def update_layer_range(start, end):
            """Update the steering layer range."""
            global steering_manager
            if steering_manager:
                # Ensure start <= end
                if start > end:
                    start, end = end, start
                steering_manager.set_layer_range(int(start), int(end))
            return get_steering_status()

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, max_tokens, temperature],
            outputs=[chatbot, msg_input, steering_display],
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, max_tokens, temperature],
            outputs=[chatbot, msg_input, steering_display],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input, steering_display],
        )

        refresh_status.click(
            fn=get_steering_status,
            outputs=[steering_display],
        )

        # Reset steering when chatbot is cleared via built-in trash icon
        chatbot.change(
            fn=on_chatbot_change,
            inputs=[chatbot],
            outputs=[steering_display],
        )

        # Update layer range when sliders change
        layer_start_slider.change(
            fn=update_layer_range,
            inputs=[layer_start_slider, layer_end_slider],
            outputs=[steering_display],
        )
        layer_end_slider.change(
            fn=update_layer_range,
            inputs=[layer_start_slider, layer_end_slider],
            outputs=[steering_display],
        )

    return app


def create_user_steer_app(config: SteeringConfig):
    """Create the user-steer mode Gradio app."""

    # Load model and steering
    load_model_and_steering(config)

    # Get all emotions for the dropdown
    all_emotions = steering_manager.get_all_emotions()

    with gr.Blocks(title="User-Controlled Steering Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# User-Controlled Steering Chat")
        gr.Markdown(
            "Chat with an AI while controlling its emotional processing using steering vectors. "
            "Select an emotion and strength to influence how the model generates responses."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("### Steering Controls")
                emotion_dropdown = gr.Dropdown(
                    choices=["(none)"] + all_emotions,
                    value="(none)",
                    label="Emotion",
                    allow_custom_value=False,
                )
                strength_slider = gr.Slider(
                    minimum=-20, maximum=20, value=0, step=0.5,
                    label="Strength (negative = away from emotion)"
                )
                scope_radio = gr.Radio(
                    choices=["Steer all messages", "Assistant only"],
                    value="Steer all messages",
                    label="Steering Scope"
                )

                gr.Markdown("### Settings")
                max_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature"
                )

                gr.Markdown("### Layer Range")
                num_layers = steering_manager.get_num_model_layers()
                layer_start_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, value=config.layer_start, step=1,
                    label="Start Layer"
                )
                layer_end_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, value=config.layer_end, step=1,
                    label="End Layer"
                )

                gr.Markdown("### System Prompt")
                system_prompt_box = gr.Textbox(
                    label="Custom System Prompt (optional)",
                    placeholder="Enter a custom system prompt...",
                    lines=4,
                    value="",
                )

                gr.Markdown("### Steering Status")
                steering_display = gr.Markdown(get_steering_status())

        def apply_steering(emotion, strength):
            """Apply user-selected steering."""
            global steering_manager
            if emotion == "(none)" or strength == 0:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            else:
                steering_manager.set_steering(emotion, strength, validate_curated=False)
            return get_steering_status()

        def respond(message, history, max_tok, temp, system_prompt, emotion, strength, scope):
            if not message.strip():
                return history, "", get_steering_status()

            # Apply steering before generation
            if emotion == "(none)" or strength == 0:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            else:
                steering_manager.set_steering(emotion, strength, validate_curated=False)

            assistant_only = (scope == "Assistant only")
            response = chat_user_steer(message, history, max_tok, temp, system_prompt, assistant_only)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return history, "", get_steering_status()

        def clear_chat():
            global steering_manager, cached_input_ids, cached_kv
            if steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            # Clear KV cache
            cached_input_ids = None
            cached_kv = None
            print("  [KV Cache] Cleared")
            return [], "", "(none)", 0, get_steering_status()

        def on_chatbot_change(history):
            """Reset steering when chatbot is cleared (e.g., via built-in trash icon)."""
            global steering_manager, cached_input_ids, cached_kv
            if not history and steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
                # Clear KV cache
                cached_input_ids = None
                cached_kv = None
                print("  [KV Cache] Cleared (chat emptied)")
            return get_steering_status()

        def update_layer_range(start, end):
            """Update the steering layer range."""
            global steering_manager
            if steering_manager:
                # Ensure start <= end
                if start > end:
                    start, end = end, start
                steering_manager.set_layer_range(int(start), int(end))
            return get_steering_status()

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, max_tokens, temperature, system_prompt_box, emotion_dropdown, strength_slider, scope_radio],
            outputs=[chatbot, msg_input, steering_display],
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, max_tokens, temperature, system_prompt_box, emotion_dropdown, strength_slider, scope_radio],
            outputs=[chatbot, msg_input, steering_display],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input, emotion_dropdown, strength_slider, steering_display],
        )

        # Reset steering when chatbot is cleared via built-in trash icon
        chatbot.change(
            fn=on_chatbot_change,
            inputs=[chatbot],
            outputs=[steering_display],
        )

        # Update layer range when sliders change
        layer_start_slider.change(
            fn=update_layer_range,
            inputs=[layer_start_slider, layer_end_slider],
            outputs=[steering_display],
        )
        layer_end_slider.change(
            fn=update_layer_range,
            inputs=[layer_start_slider, layer_end_slider],
            outputs=[steering_display],
        )

        # Update status when steering controls change
        emotion_dropdown.change(
            fn=apply_steering,
            inputs=[emotion_dropdown, strength_slider],
            outputs=[steering_display],
        )
        strength_slider.change(
            fn=apply_steering,
            inputs=[emotion_dropdown, strength_slider],
            outputs=[steering_display],
        )

    return app


# UI labels for the mode radio. Kept as constants so the value comparison
# logic and the radio choices stay in sync.
MODE_SELF = "Self-steer (model controls)"
MODE_USER = "User-steer (you control)"


def create_chat_app(config: SteeringConfig, initial_mode: str = "self-steer"):
    """Unified chat app: mode toggle in the UI, no restart required."""

    load_model_and_steering(config)
    source_labels = steering_manager.get_source_labels()
    initial_source = steering_manager.active_source
    initial_concepts = steering_manager.get_all_emotions()

    initial_mode_label = MODE_SELF if initial_mode == "self-steer" else MODE_USER
    is_self_initial = initial_mode == "self-steer"

    SELF_STEER_PROMPT_LABEL = "System prompt additions (appended to the steering-tool instructions)"
    USER_STEER_PROMPT_LABEL = "System prompt (optional)"

    def cheatsheet_for(source_label: str) -> str:
        """Markdown shown to the user in self-steer mode for a given source."""
        section = CONCEPT_SECTIONS.get(source_label)
        if section:
            return section
        # Fallback: just dump the curated list
        curated = (CURATED_PER_SOURCE.get(source_label)
                   or steering_manager.get_curated_emotions())
        return "### Available concepts\n\n" + ", ".join(curated)

    with gr.Blocks(title="Steering Chat", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Steering Chat")
        with gr.Row():
            mode_radio = gr.Radio(
                choices=[MODE_SELF, MODE_USER],
                value=initial_mode_label,
                label="Mode",
                info=("Self-steer: the model uses a steer() tool to adjust its own emotional "
                      "processing. User-steer: you pick an emotion and strength directly."),
            )
            source_radio = gr.Radio(
                choices=source_labels,
                value=initial_source,
                label="Vector Source",
                info="Which set of steering vectors to use.",
                visible=len(source_labels) > 1,
            )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                # ---- User-steer-only controls ----
                user_steer_header = gr.Markdown(
                    "### Steering Controls", visible=not is_self_initial
                )
                emotion_dropdown = gr.Dropdown(
                    choices=["(none)"] + initial_concepts,
                    value="(none)",
                    label="Concept",
                    allow_custom_value=False,
                    visible=not is_self_initial,
                )
                strength_slider = gr.Slider(
                    minimum=-20, maximum=20, value=0, step=0.5,
                    label="Strength (negative = away from emotion)",
                    visible=not is_self_initial,
                )
                scope_radio = gr.Radio(
                    choices=["Steer all messages", "Assistant only"],
                    value="Steer all messages",
                    label="Steering Scope",
                    visible=not is_self_initial,
                )

                # ---- Always visible ----
                gr.Markdown("### Settings")
                max_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature",
                )

                gr.Markdown("### Layer Range")
                num_layers = steering_manager.get_num_model_layers()
                layer_start_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, value=config.layer_start, step=1,
                    label="Start Layer",
                )
                layer_end_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, value=config.layer_end, step=1,
                    label="End Layer",
                )

                gr.Markdown("### Steering Status")
                steering_display = gr.Markdown(get_steering_status())
                refresh_status = gr.Button("Refresh Status")

                gr.Markdown("### System Prompt")
                system_prompt_box = gr.Textbox(
                    label=SELF_STEER_PROMPT_LABEL if is_self_initial else USER_STEER_PROMPT_LABEL,
                    placeholder="Optional...",
                    lines=4,
                    value="",
                )

                # ---- Self-steer-only cheatsheet (source-aware) ----
                emotions_cheatsheet = gr.Markdown(
                    cheatsheet_for(initial_source),
                    visible=is_self_initial,
                )

        # ---- Handlers ----
        def respond(message, history, max_tok, temp, mode,
                    system_prompt, emotion, strength, scope):
            if not message.strip():
                return history, "", get_steering_status()

            if mode == MODE_SELF:
                # Pick the concept-list section that matches the active source
                # so the LLM's system prompt enumerates the correct concepts.
                active_label = steering_manager.active_source
                concept_section = CONCEPT_SECTIONS.get(active_label, "")
                response = chat_self_steer(
                    message, history, max_tok, temp,
                    custom_additions=system_prompt,
                    concept_section=concept_section,
                )
            else:
                # Apply user-controlled steering before generating
                if emotion == "(none)" or strength == 0:
                    steering_manager.set_steering(None, 0.0, validate_curated=False)
                else:
                    steering_manager.set_steering(emotion, strength, validate_curated=False)
                assistant_only = (scope == "Assistant only")
                response = chat_user_steer(
                    message, history, max_tok, temp, system_prompt, assistant_only
                )

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return history, "", get_steering_status()

        def clear_chat():
            global steering_manager, cached_input_ids, cached_kv
            if steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            cached_input_ids = None
            cached_kv = None
            print("  [KV Cache] Cleared")
            return [], "", "(none)", 0, get_steering_status()

        def on_chatbot_change(history):
            global steering_manager, cached_input_ids, cached_kv
            if not history and steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
                cached_input_ids = None
                cached_kv = None
                print("  [KV Cache] Cleared (chat emptied)")
            return get_steering_status()

        def update_layer_range(start, end):
            global steering_manager
            if steering_manager:
                if start > end:
                    start, end = end, start
                steering_manager.set_layer_range(int(start), int(end))
            return get_steering_status()

        def apply_steering(emotion, strength):
            global steering_manager
            if emotion == "(none)" or strength == 0:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            else:
                steering_manager.set_steering(emotion, strength, validate_curated=False)
            return get_steering_status()

        def on_mode_change(mode):
            """Toggle visibility of mode-specific controls and reset steering."""
            global steering_manager, cached_input_ids, cached_kv
            is_self = (mode == MODE_SELF)
            # Reset any active steering so the new mode starts clean
            if steering_manager:
                steering_manager.set_steering(None, 0.0, validate_curated=False)
            cached_input_ids = None
            cached_kv = None
            new_label = SELF_STEER_PROMPT_LABEL if is_self else USER_STEER_PROMPT_LABEL
            return (
                gr.update(visible=not is_self),                              # user_steer_header
                gr.update(visible=not is_self, value="(none)"),              # emotion_dropdown
                gr.update(visible=not is_self, value=0),                     # strength_slider
                gr.update(visible=not is_self),                              # scope_radio
                gr.update(visible=is_self),                                  # emotions_cheatsheet
                gr.update(label=new_label),                                  # system_prompt_box
                get_steering_status(),                                       # steering_display
            )

        def on_source_change(source_label):
            """Switch the active vector source; refresh dropdown + cheatsheet."""
            global steering_manager, cached_input_ids, cached_kv
            steering_manager.set_active_source(source_label)
            cached_input_ids = None
            cached_kv = None
            new_concepts = steering_manager.get_all_emotions()
            return (
                gr.update(choices=["(none)"] + new_concepts, value="(none)"),  # emotion_dropdown
                gr.update(value=0),                                            # strength_slider
                gr.update(value=cheatsheet_for(source_label)),                 # emotions_cheatsheet
                get_steering_status(),                                         # steering_display
            )

        # ---- Wire-up ----
        respond_inputs = [
            msg_input, chatbot, max_tokens, temperature, mode_radio,
            system_prompt_box, emotion_dropdown, strength_slider, scope_radio,
        ]
        respond_outputs = [chatbot, msg_input, steering_display]

        submit_btn.click(fn=respond, inputs=respond_inputs, outputs=respond_outputs)
        msg_input.submit(fn=respond, inputs=respond_inputs, outputs=respond_outputs)

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input, emotion_dropdown, strength_slider, steering_display],
        )

        refresh_status.click(fn=get_steering_status, outputs=[steering_display])

        chatbot.change(
            fn=on_chatbot_change, inputs=[chatbot], outputs=[steering_display],
        )

        layer_start_slider.change(
            fn=update_layer_range,
            inputs=[layer_start_slider, layer_end_slider],
            outputs=[steering_display],
        )
        layer_end_slider.change(
            fn=update_layer_range,
            inputs=[layer_start_slider, layer_end_slider],
            outputs=[steering_display],
        )

        emotion_dropdown.change(
            fn=apply_steering,
            inputs=[emotion_dropdown, strength_slider],
            outputs=[steering_display],
        )
        strength_slider.change(
            fn=apply_steering,
            inputs=[emotion_dropdown, strength_slider],
            outputs=[steering_display],
        )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[
                user_steer_header, emotion_dropdown, strength_slider, scope_radio,
                emotions_cheatsheet, system_prompt_box, steering_display,
            ],
        )

        source_radio.change(
            fn=on_source_change,
            inputs=[source_radio],
            outputs=[
                emotion_dropdown, strength_slider, emotions_cheatsheet, steering_display,
            ],
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Steering Chat")
    parser.add_argument("--mode", type=str, default="self-steer", choices=["self-steer", "user-steer"],
                        help="Initial mode (can be toggled in the UI). 'self-steer' (model controls) "
                             "or 'user-steer' (user controls)")
    parser.add_argument("--layer-start", type=int, default=19, help="Start layer for steering")
    parser.add_argument("--layer-end", type=int, default=24, help="End layer for steering")
    parser.add_argument("--decay-mode", type=str, default="none", choices=["none", "progressive"])
    parser.add_argument("--decay-rate", type=float, default=0.95)
    parser.add_argument(
        "--vectors-dir", type=str, default=str(VECTORS_DIR),
        help="Directory of vectors to load (must contain metadata.json)",
    )

    args = parser.parse_args()
    VECTORS_DIR = Path(args.vectors_dir)
    print(f"Using vectors from: {VECTORS_DIR}")

    config = SteeringConfig(
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        decay_mode=args.decay_mode,
        decay_rate=args.decay_rate,
    )

    print(f"\nSteering Chat")
    print(f"{'=' * 40}")
    print(f"Mode: {args.mode}")
    print(f"Configuration:")
    print(f"  Layer range: {args.layer_start}-{args.layer_end}")
    print(f"  Decay mode: {args.decay_mode}")
    print(f"\nStarting...\n")

    t_create = time.perf_counter()
    app = create_chat_app(config, initial_mode=args.mode)
    print(f"[Startup] App created: {time.perf_counter() - t_create:.2f}s")

    print("[Startup] Launching Gradio...")
    app.launch(share=False)
