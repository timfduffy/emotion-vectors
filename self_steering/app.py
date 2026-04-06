"""
Self-Steering Chat Application

A chat interface where the model can adjust its own emotional processing
using steering vectors.
"""

import gradio as gr
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import json

from steering import SteeringManager, SteeringConfig
from tools import parse_tool_call, format_tool_result, STEER_TOOL_SCHEMA
from prompts import get_system_prompt


# Configuration
MODEL_PATH = r"H:\Models\huggingface\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
VECTORS_DIR = Path(__file__).parent.parent / "emotion_vectors_denoised"

# Global state
model = None
tokenizer = None
steering_manager = None
past_key_values = None  # KV cache persistence


def load_model_and_steering(config: SteeringConfig):
    """Load the model and initialize steering."""
    global model, tokenizer, steering_manager

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")

    print("Initializing steering...")
    config.vectors_dir = str(VECTORS_DIR)
    steering_manager = SteeringManager(model, config)
    steering_manager.register_hooks()
    print("Steering initialized!")


def generate_response(
    messages: list,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, bool, Optional[dict]]:
    """
    Generate a response, checking for tool calls.

    Returns:
        - response text
        - whether a tool was called
        - tool call info (if any)
    """
    global model, tokenizer, steering_manager, past_key_values

    # Format messages with chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Reset per-generation logging
    steering_manager.reset_generation_log()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )

    # Store KV cache for potential future use
    # Note: Cross-turn caching with varying steering states is complex;
    # for now we generate fresh each turn but this preserves the cache structure
    if hasattr(outputs, 'past_key_values'):
        past_key_values = outputs.past_key_values

    # Decode only the new tokens
    sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs[0]
    new_tokens = sequences[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Check for tool calls
    tool_call = parse_tool_call(response)

    return response, tool_call is not None, tool_call


def execute_tool_call(tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result."""
    global steering_manager

    if tool_name == "steer":
        emotion = args.get("emotion", "none")
        strength = float(args.get("strength", 0.0))

        try:
            steering_manager.set_steering(emotion, strength)
            return format_tool_result(emotion, strength, success=True)
        except ValueError as e:
            return format_tool_result(emotion, strength, success=False, message=str(e))

    return f"[Unknown tool: {tool_name}]"


def chat(
    message: str,
    history: list,
    max_tokens: int,
    temperature: float,
):
    """Main chat function for Gradio."""
    global steering_manager

    # Build message list with system prompt
    messages = [{"role": "system", "content": get_system_prompt()}]

    # Add history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

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


def get_steering_status():
    """Get current steering status for display."""
    global steering_manager
    if steering_manager is None:
        return "Steering not initialized"

    status = steering_manager.get_status()
    if not status["active"]:
        return "**Steering:** Inactive (baseline processing)"
    else:
        direction = "toward" if status["strength"] > 0 else "away from"
        return f"**Steering:** {abs(status['strength']):.2f}x {direction} **{status['emotion']}**"


def create_app(config: SteeringConfig):
    """Create the Gradio app."""

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
                chatbot = gr.Chatbot(height=500, label="Conversation", type="tuples")
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

            response = chat(message, history, max_tok, temp)
            history = history + [(message, response)]
            return history, "", get_steering_status()

        def clear_chat():
            global steering_manager
            if steering_manager:
                steering_manager.set_steering(None, 0.0)
            return [], "", get_steering_status()

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

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-Steering Chat")
    parser.add_argument("--layer-start", type=int, default=19, help="Start layer for steering")
    parser.add_argument("--layer-end", type=int, default=24, help="End layer for steering")
    parser.add_argument("--decay-mode", type=str, default="none", choices=["none", "progressive"])
    parser.add_argument("--decay-rate", type=float, default=0.95)

    args = parser.parse_args()

    config = SteeringConfig(
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        decay_mode=args.decay_mode,
        decay_rate=args.decay_rate,
    )

    app = create_app(config)
    app.launch(share=False)
