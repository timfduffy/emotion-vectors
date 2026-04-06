"""
Emotion Activation Viewer - Gradio Web App

Visualizes emotion concept activations in model outputs by highlighting
tokens based on their projection onto emotion vectors.
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import html

# Global variables for model and vectors
model = None
tokenizer = None
emotion_vectors = None
emotions = None
layers = None


def load_emotion_vectors(vectors_dir: str) -> tuple[dict, list, list]:
    """Load denoised emotion vectors."""
    vectors_dir = Path(vectors_dir)

    with open(vectors_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    emotion_list = metadata["emotions"]
    layer_list = metadata["layers"]

    vectors_by_layer = {}
    for layer in layer_list:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        vectors_by_layer[layer] = {e: data[e] for e in emotion_list}

    return vectors_by_layer, emotion_list, layer_list


def load_model(model_path: str, device: str = "cuda"):
    """Load the model and tokenizer."""
    global model, tokenizer

    print(f"Loading model: {model_path}")
    print("  Step 1: Loading tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("  Tokenizer loaded!")
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        raise

    # Clear CUDA cache before loading
    print("  Step 2: Clearing CUDA cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  CUDA available. Device: {torch.cuda.get_device_name(0)}")
        print(f"  Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    import sys

    print("  Step 3: Loading model weights...", flush=True)
    print("  (This may take 30-60 seconds)", flush=True)

    try:
        # Match exact loading pattern from working scripts
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        print("  Weights loaded to GPU!", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"  ERROR loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    print("  Setting eval mode...", flush=True)
    model.eval()
    print("  Eval mode set!", flush=True)

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Memory after loading: {mem:.1f} GB", flush=True)

    print("  Testing model access...", flush=True)
    num_layers, hidden_size = get_model_config(model)
    print(f"  Model config: {num_layers} layers, {hidden_size} hidden", flush=True)

    print("Model loaded successfully!", flush=True)
    sys.stdout.flush()

    return model, tokenizer


def get_model_config(model):
    """Get model config handling Gemma 4 structure."""
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'num_layers', 36)
    hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', 2048)
    return num_layers, hidden_size


def generate_with_activations(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, list, torch.Tensor]:
    """
    Generate text and capture residual stream activations for all tokens.

    Returns:
        - full_text: The complete text (prompt + response)
        - tokens: List of token strings
        - activations: Tensor of shape (num_layers, num_tokens, hidden_size)
    """
    global model, tokenizer

    print("Starting generation...")

    # Format prompt using chat template for instruction-tuned model
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"Formatted prompt: {formatted_prompt[:100]}...")

    # Tokenize formatted prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    print(f"Prompt tokenized: {prompt_length} tokens")

    # Generate with output_hidden_states
    print("Running model.generate()...")
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
    print("Generation complete!")

    # Get full sequence
    full_ids = outputs.sequences[0]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    print(f"Generated {len(full_ids)} total tokens")

    # Get individual tokens for display
    tokens = [tokenizer.decode([tid]) for tid in full_ids]

    # Debug: check hidden_states structure
    print(f"Hidden states structure: {len(outputs.hidden_states)} generation steps")
    if len(outputs.hidden_states) > 0:
        print(f"First step has {len(outputs.hidden_states[0])} layers")
        if len(outputs.hidden_states[0]) > 0:
            print(f"First layer shape: {outputs.hidden_states[0][0].shape}")

    num_layers, hidden_size = get_model_config(model)
    print(f"Model config: {num_layers} layers, {hidden_size} hidden size")

    # For the prompt tokens, we get hidden states from the first generation step
    # For generated tokens, we get them from subsequent steps

    all_activations = []

    # First step contains all prompt tokens
    first_step_hidden = outputs.hidden_states[0]  # tuple of (num_layers+1,) tensors
    # Each tensor is (batch, prompt_len, hidden_size)

    # Stack layers for prompt tokens (skip embedding layer [0])
    print("Processing prompt activations...")
    prompt_activations = torch.stack(
        [first_step_hidden[layer_idx][0] for layer_idx in range(1, len(first_step_hidden))],
        dim=0
    )  # (num_layers, prompt_len, hidden_size)
    all_activations.append(prompt_activations)
    print(f"Prompt activations shape: {prompt_activations.shape}")

    # For each generated token
    print(f"Processing {len(outputs.hidden_states) - 1} generated tokens...")
    for step_idx in range(1, len(outputs.hidden_states)):
        step_hidden = outputs.hidden_states[step_idx]
        # Each tensor is (batch, 1, hidden_size) for the new token
        token_activations = torch.stack(
            [step_hidden[layer_idx][0, -1:, :] for layer_idx in range(1, len(step_hidden))],
            dim=0
        )  # (num_layers, 1, hidden_size)
        all_activations.append(token_activations)

    # Concatenate along sequence dimension
    print("Concatenating activations...")
    activations = torch.cat(all_activations, dim=1)  # (num_layers, total_tokens, hidden_size)
    print(f"Final activations shape: {activations.shape}")

    # Move to CPU and convert
    print("Moving to CPU...")
    result = activations.float().cpu()

    # Clear GPU memory
    del activations, all_activations, outputs
    torch.cuda.empty_cache()
    print("Done!")

    return full_text, tokens, result


def compute_emotion_projections(
    activations,
    emotion: str,
    layer: int,
) -> np.ndarray:
    """
    Compute projection of activations onto an emotion vector.

    Args:
        activations: (num_layers, num_tokens, hidden_size) - tensor or numpy array
        emotion: Name of emotion
        layer: Layer index to use

    Returns:
        projections: (num_tokens,) array of projection values
    """
    global emotion_vectors

    # Get emotion vector for this layer
    emotion_vec = emotion_vectors[layer][emotion]
    emotion_vec = emotion_vec / np.linalg.norm(emotion_vec)  # Normalize

    # Get activations for this layer (handle both tensor and numpy)
    if hasattr(activations, 'numpy'):
        layer_activations = activations[layer].numpy()
    else:
        layer_activations = activations[layer]

    # Compute dot product (projection)
    projections = layer_activations @ emotion_vec

    return projections


def get_text_color(r, g, b):
    """Return black or white text color based on background luminance."""
    # Calculate luminance using standard formula
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "black" if luminance > 140 else "white"


def norm_to_rgb(norm_val):
    """Convert normalized value [-1, 1] to RGB tuple."""
    if norm_val < 0:
        intensity = abs(norm_val)
        r = int(255 * (1 - intensity))
        g = int(255 * (1 - intensity))
        b = 255
    else:
        intensity = norm_val
        r = 255
        g = int(255 * (1 - intensity))
        b = int(255 * (1 - intensity))
    return r, g, b


def projections_to_html_single(
    tokens: list,
    projections: np.ndarray,
    emotion: str,
    layer: int,
) -> str:
    """Single mode: one emotion, one layer, full display."""
    max_abs = max(abs(projections.min()), abs(projections.max()), 1e-6)
    normalized = projections / max_abs

    html_parts = []
    html_parts.append('<div style="font-family: monospace; font-size: 14px; line-height: 1.8; white-space: pre-wrap;">')
    html_parts.append(f'<div style="margin-bottom: 10px; font-weight: bold;">Emotion: {emotion} | Layer: {layer}</div>')
    html_parts.append(f'<div style="margin-bottom: 10px;">Projection range: [{projections.min():.2f}, {projections.max():.2f}]</div>')

    for token, proj, norm_val in zip(tokens, projections, normalized):
        r, g, b = norm_to_rgb(norm_val)
        text_color = get_text_color(r, g, b)
        safe_token = html.escape(token)

        html_parts.append(
            f'<span style="background-color: rgb({r},{g},{b}); color: {text_color}; padding: 1px 2px;" '
            f'title="{emotion}: {proj:.3f}">{safe_token}</span>'
        )

    html_parts.append('</div>')
    html_parts.append(get_color_legend())

    return ''.join(html_parts)


def projections_to_html_multi_layer(
    tokens: list,
    activations: np.ndarray,
    emotion: str,
    layer_list: list,
) -> str:
    """Multi-layer mode: one emotion, all layers as rows."""
    global emotion_vectors

    html_parts = []
    html_parts.append('<div style="font-family: monospace; font-size: 12px; overflow-x: auto;">')
    html_parts.append(f'<div style="margin-bottom: 10px; font-weight: bold;">Emotion: {emotion} | All Layers</div>')
    html_parts.append('<table style="border-collapse: collapse;">')

    # Prepare tokens with newlines escaped
    display_tokens = [html.escape(t).replace('\n', '\\n').replace('\r', '\\r') for t in tokens]

    for layer in layer_list:
        # Compute projections for this layer
        emotion_vec = emotion_vectors[layer][emotion]
        emotion_vec = emotion_vec / np.linalg.norm(emotion_vec)
        layer_activations = activations[layer]
        projections = layer_activations @ emotion_vec

        max_abs = max(abs(projections.min()), abs(projections.max()), 1e-6)
        normalized = projections / max_abs

        html_parts.append(f'<tr><td style="padding: 2px 8px; font-weight: bold; white-space: nowrap;">L{layer}</td><td>')

        for token, proj, norm_val in zip(display_tokens, projections, normalized):
            r, g, b = norm_to_rgb(norm_val)
            text_color = get_text_color(r, g, b)
            html_parts.append(
                f'<span style="background-color: rgb({r},{g},{b}); color: {text_color}; padding: 1px 2px;" '
                f'title="L{layer} {emotion}: {proj:.3f}">{token}</span>'
            )

        html_parts.append('</td></tr>')

    html_parts.append('</table></div>')
    html_parts.append(get_color_legend())

    return ''.join(html_parts)


def projections_to_html_multi_concept(
    tokens: list,
    activations: np.ndarray,
    layer: int,
    emotion_list: list,
) -> str:
    """Multi-concept mode: all emotions, one layer as rows."""
    global emotion_vectors

    html_parts = []
    html_parts.append('<div style="font-family: monospace; font-size: 12px; overflow-x: auto;">')
    html_parts.append(f'<div style="margin-bottom: 10px; font-weight: bold;">Layer: {layer} | All Emotions</div>')
    html_parts.append('<table style="border-collapse: collapse;">')

    # Prepare tokens with newlines escaped
    display_tokens = [html.escape(t).replace('\n', '\\n').replace('\r', '\\r') for t in tokens]

    layer_activations = activations[layer]

    for emotion in emotion_list:
        emotion_vec = emotion_vectors[layer][emotion]
        emotion_vec = emotion_vec / np.linalg.norm(emotion_vec)
        projections = layer_activations @ emotion_vec

        max_abs = max(abs(projections.min()), abs(projections.max()), 1e-6)
        normalized = projections / max_abs

        html_parts.append(f'<tr><td style="padding: 2px 8px; font-weight: bold; white-space: nowrap;">{emotion}</td><td>')

        for token, proj, norm_val in zip(display_tokens, projections, normalized):
            r, g, b = norm_to_rgb(norm_val)
            text_color = get_text_color(r, g, b)
            html_parts.append(
                f'<span style="background-color: rgb({r},{g},{b}); color: {text_color}; padding: 1px 2px;" '
                f'title="{emotion}: {proj:.3f}">{token}</span>'
            )

        html_parts.append('</td></tr>')

    html_parts.append('</table></div>')
    html_parts.append(get_color_legend())

    return ''.join(html_parts)


def projections_to_html_layer_range(
    tokens: list,
    activations: np.ndarray,
    layer_start: int,
    layer_end: int,
    layer_list: list,
    emotion_list: list,
) -> str:
    """Layer range mode: all emotions, mean projection across a range of layers."""
    global emotion_vectors

    # Find indices for the layer range
    start_idx = layer_list.index(layer_start)
    end_idx = layer_list.index(layer_end)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    selected_layers = layer_list[start_idx:end_idx + 1]

    html_parts = []
    html_parts.append('<div style="font-family: monospace; font-size: 12px; overflow-x: auto;">')
    html_parts.append(f'<div style="margin-bottom: 10px; font-weight: bold;">Layers {layer_start}-{layer_end} (mean of {len(selected_layers)} layers) | All Emotions</div>')
    html_parts.append('<table style="border-collapse: collapse;">')

    # Prepare tokens with newlines escaped
    display_tokens = [html.escape(t).replace('\n', '\\n').replace('\r', '\\r') for t in tokens]

    for emotion in emotion_list:
        # Compute projections for each layer and average
        all_projections = []
        for layer in selected_layers:
            emotion_vec = emotion_vectors[layer][emotion]
            emotion_vec = emotion_vec / np.linalg.norm(emotion_vec)
            layer_activations = activations[layer]
            projections = layer_activations @ emotion_vec
            all_projections.append(projections)

        # Mean across layers
        mean_projections = np.mean(all_projections, axis=0)

        max_abs = max(abs(mean_projections.min()), abs(mean_projections.max()), 1e-6)
        normalized = mean_projections / max_abs

        html_parts.append(f'<tr><td style="padding: 2px 8px; font-weight: bold; white-space: nowrap;">{emotion}</td><td>')

        for token, proj, norm_val in zip(display_tokens, mean_projections, normalized):
            r, g, b = norm_to_rgb(norm_val)
            text_color = get_text_color(r, g, b)
            html_parts.append(
                f'<span style="background-color: rgb({r},{g},{b}); color: {text_color}; padding: 1px 2px;" '
                f'title="{emotion} (L{layer_start}-{layer_end}): {proj:.3f}">{token}</span>'
            )

        html_parts.append('</td></tr>')

    html_parts.append('</table></div>')
    html_parts.append(get_color_legend())

    return ''.join(html_parts)


def get_color_legend():
    """Return HTML for the color legend."""
    return '''
        <div style="margin-top: 20px; font-size: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>Negative</span>
                <div style="width: 200px; height: 20px; background: linear-gradient(to right, rgb(0,0,255), rgb(255,255,255), rgb(255,0,0));"></div>
                <span>Positive</span>
            </div>
        </div>
    '''


def projections_to_html(
    tokens: list,
    projections: np.ndarray,
    emotion: str,
    layer: int,
) -> str:
    """Legacy wrapper for single mode."""
    return projections_to_html_single(tokens, projections, emotion, layer)


def analyze_text(
    prompt: str,
    emotion: str,
    layer: int,
    max_tokens: int,
    temperature: float,
):
    """Main function called by Gradio interface."""
    global emotions, layers

    if model is None:
        return "Error: Model not loaded. Please wait for initialization."

    # Generate with activations
    try:
        full_text, tokens, activations = generate_with_activations(
            prompt, max_new_tokens=max_tokens, temperature=temperature
        )
    except Exception as e:
        return f"Error during generation: {str(e)}"

    # Compute projections
    projections = compute_emotion_projections(activations, emotion, layer)

    # Create HTML visualization
    html_output = projections_to_html(tokens, projections, emotion, layer)

    return html_output


def update_visualization(
    emotion: str,
    layer: int,
    stored_data: dict,
):
    """Update visualization when emotion or layer changes (without regenerating)."""
    if stored_data is None or "activations" not in stored_data:
        return "Please generate text first."

    tokens = stored_data["tokens"]
    activations = torch.tensor(stored_data["activations"])

    projections = compute_emotion_projections(activations, emotion, layer)
    html_output = projections_to_html(tokens, projections, emotion, layer)

    return html_output


# Paths - adjust as needed
MODEL_PATH = r"H:\Models\huggingface\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
VECTORS_DIR = Path(__file__).parent.parent / "emotion_vectors_denoised"

def initialize_all():
    """Load emotion vectors AND model at startup."""
    global emotion_vectors, emotions, layers, model, tokenizer

    print("Loading emotion vectors...")
    emotion_vectors, emotions, layers = load_emotion_vectors(VECTORS_DIR)
    print(f"Loaded {len(emotions)} emotions, {len(layers)} layers")

    print("\nLoading model (this takes 30-60 seconds)...")
    load_model(MODEL_PATH)

    return emotions, layers


def ensure_model_loaded():
    """Check model is loaded (should already be loaded at startup)."""
    global model
    if model is None:
        raise RuntimeError("Model not loaded! This should not happen.")
    return True


# Create Gradio interface
def create_app():
    """Create and return the Gradio app."""
    global emotions, layers

    # Load everything before starting Gradio
    emotions, layers = initialize_all()

    # Default to mid-late layer
    default_layer = layers[int(len(layers) * 0.67)]

    with gr.Blocks(title="Emotion Activation Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Emotion Activation Viewer")
        gr.Markdown(
            "Enter a prompt to generate a response, then visualize how strongly each token "
            "activates different emotion concepts. Hover over tokens to see exact values."
        )

        # Store activations between updates
        stored_data = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                )

                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=16, maximum=512, value=128, step=16,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature"
                    )

                generate_btn = gr.Button("Generate & Analyze", variant="primary")

            with gr.Column(scale=1):
                display_mode = gr.Radio(
                    choices=["Single", "Multi-layer", "Multi-concept", "Layer range"],
                    value="Single",
                    label="Display Mode",
                )
                emotion_dropdown = gr.Dropdown(
                    choices=emotions,
                    value="happy",
                    label="Emotion Concept",
                    interactive=True,
                )
                layer_dropdown = gr.Dropdown(
                    choices=layers,
                    value=default_layer,
                    label="Layer (for Single/Multi-concept)",
                    interactive=True,
                )
                with gr.Row():
                    layer_start = gr.Dropdown(
                        choices=layers,
                        value=layers[int(len(layers) * 0.5)],
                        label="Layer Start",
                        interactive=True,
                    )
                    layer_end = gr.Dropdown(
                        choices=layers,
                        value=layers[int(len(layers) * 0.75)],
                        label="Layer End",
                        interactive=True,
                    )
                update_btn = gr.Button("Update Visualization")

        output_html = gr.HTML(label="Visualization")

        # Generate button: run model and create visualization
        def generate_and_visualize(prompt, emotion, layer, l_start, l_end, max_tok, temp, mode):
            if not prompt.strip():
                return "Please enter a prompt.", None

            # Ensure model is loaded (lazy loading)
            try:
                ensure_model_loaded()
            except Exception as e:
                return f"Error loading model: {str(e)}", None

            try:
                full_text, tokens, activations = generate_with_activations(
                    prompt, max_new_tokens=max_tok, temperature=temp
                )
            except Exception as e:
                import traceback
                return f"Error: {str(e)}\n\n{traceback.format_exc()}", None

            # Store for later updates
            data = {
                "tokens": tokens,
                "activations": activations.numpy().tolist(),
            }

            # Create visualization based on mode
            html_output = create_visualization(tokens, activations.numpy(), emotion, layer, l_start, l_end, mode)

            return html_output, data

        def create_visualization(tokens, activations, emotion, layer, l_start, l_end, mode):
            """Create HTML visualization based on display mode."""
            if mode == "Multi-layer":
                return projections_to_html_multi_layer(tokens, activations, emotion, layers)
            elif mode == "Multi-concept":
                return projections_to_html_multi_concept(tokens, activations, layer, emotions)
            elif mode == "Layer range":
                return projections_to_html_layer_range(tokens, activations, l_start, l_end, layers, emotions)
            else:  # Single
                projections = compute_emotion_projections(activations, emotion, layer)
                return projections_to_html_single(tokens, projections, emotion, layer)

        generate_btn.click(
            fn=generate_and_visualize,
            inputs=[prompt_input, emotion_dropdown, layer_dropdown, layer_start, layer_end, max_tokens, temperature, display_mode],
            outputs=[output_html, stored_data],
        )

        # Update button: just re-visualize with new emotion/layer/mode
        def update_viz(emotion, layer, l_start, l_end, mode, data):
            if data is None:
                return "Please generate text first."

            tokens = data["tokens"]
            activations = np.array(data["activations"])

            return create_visualization(tokens, activations, emotion, layer, l_start, l_end, mode)

        update_btn.click(
            fn=update_viz,
            inputs=[emotion_dropdown, layer_dropdown, layer_start, layer_end, display_mode, stored_data],
            outputs=[output_html],
        )

        # Also update when dropdown/radio changes
        emotion_dropdown.change(
            fn=update_viz,
            inputs=[emotion_dropdown, layer_dropdown, layer_start, layer_end, display_mode, stored_data],
            outputs=[output_html],
        )
        layer_dropdown.change(
            fn=update_viz,
            inputs=[emotion_dropdown, layer_dropdown, layer_start, layer_end, display_mode, stored_data],
            outputs=[output_html],
        )
        layer_start.change(
            fn=update_viz,
            inputs=[emotion_dropdown, layer_dropdown, layer_start, layer_end, display_mode, stored_data],
            outputs=[output_html],
        )
        layer_end.change(
            fn=update_viz,
            inputs=[emotion_dropdown, layer_dropdown, layer_start, layer_end, display_mode, stored_data],
            outputs=[output_html],
        )
        display_mode.change(
            fn=update_viz,
            inputs=[emotion_dropdown, layer_dropdown, layer_start, layer_end, display_mode, stored_data],
            outputs=[output_html],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
