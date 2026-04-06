"""
Gradio viewer for pre-generated activations.
No model loading required - just loads saved activations and emotion vectors.
"""

import gradio as gr
import numpy as np
from pathlib import Path
import json
import html

# Paths
VECTORS_DIR = Path(__file__).parent.parent / "emotion_vectors_denoised"
ACTIVATIONS_DIR = Path(__file__).parent / "activations"

# Global state
emotion_vectors = None
emotions = None
layers = None


def load_emotion_vectors():
    """Load denoised emotion vectors."""
    global emotion_vectors, emotions, layers

    print("Loading emotion vectors...")
    vectors_dir = VECTORS_DIR

    with open(vectors_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    emotions = metadata["emotions"]
    layers = metadata["layers"]

    emotion_vectors = {}
    for layer in layers:
        data = np.load(vectors_dir / f"emotion_vectors_layer_{layer}.npz")
        emotion_vectors[layer] = {e: data[e] for e in emotions}

    print(f"Loaded {len(emotions)} emotions, {len(layers)} layers")
    return emotions, layers


def get_available_activations():
    """List available activation files."""
    if not ACTIVATIONS_DIR.exists():
        return []

    dirs = [d.name for d in ACTIVATIONS_DIR.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
    return dirs


def load_activations(name: str):
    """Load saved activations."""
    path = ACTIVATIONS_DIR / name

    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    activations = np.load(path / "activations.npy")

    return metadata, activations


def compute_projections(activations: np.ndarray, emotion: str, layer: int) -> np.ndarray:
    """Compute projection of activations onto emotion vector."""
    emotion_vec = emotion_vectors[layer][emotion]
    emotion_vec = emotion_vec / np.linalg.norm(emotion_vec)

    layer_activations = activations[layer]  # (num_tokens, hidden_size)
    projections = layer_activations @ emotion_vec

    return projections


def create_visualization(tokens, projections, emotion, layer):
    """Create HTML visualization with highlighted tokens."""
    max_abs = max(abs(projections.min()), abs(projections.max()), 1e-6)
    normalized = projections / max_abs

    html_parts = []
    html_parts.append('<div style="font-family: monospace; font-size: 14px; line-height: 1.8; white-space: pre-wrap; padding: 10px;">')
    html_parts.append(f'<div style="margin-bottom: 10px; font-weight: bold;">Emotion: {emotion} | Layer: {layer}</div>')
    html_parts.append(f'<div style="margin-bottom: 15px; color: #666;">Projection range: [{projections.min():.3f}, {projections.max():.3f}]</div>')
    html_parts.append('<div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">')

    for token, proj, norm_val in zip(tokens, projections, normalized):
        if norm_val < 0:
            intensity = min(abs(norm_val), 1.0)
            r = int(255 * (1 - intensity * 0.7))
            g = int(255 * (1 - intensity * 0.7))
            b = 255
        else:
            intensity = min(norm_val, 1.0)
            r = 255
            g = int(255 * (1 - intensity * 0.7))
            b = int(255 * (1 - intensity * 0.7))

        safe_token = html.escape(token)
        html_parts.append(
            f'<span style="background-color: rgb({r},{g},{b}); padding: 2px 0; border-radius: 2px;" '
            f'title="{emotion}: {proj:.4f}">{safe_token}</span>'
        )

    html_parts.append('</div></div>')

    # Legend
    html_parts.append('''
        <div style="margin-top: 20px; padding: 10px; font-size: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>Negative</span>
                <div style="width: 200px; height: 20px; background: linear-gradient(to right, rgb(77,77,255), rgb(255,255,255), rgb(255,77,77)); border-radius: 3px;"></div>
                <span>Positive</span>
            </div>
        </div>
    ''')

    return ''.join(html_parts)


def visualize(activation_file, emotion, layer):
    """Main visualization function."""
    if not activation_file:
        return "<p>Please select an activation file or generate new activations first.</p>"

    try:
        metadata, activations = load_activations(activation_file)
    except Exception as e:
        return f"<p>Error loading activations: {e}</p>"

    tokens = metadata["tokens"]
    projections = compute_projections(activations, emotion, layer)

    return create_visualization(tokens, projections, emotion, layer)


def show_text(activation_file):
    """Show the generated text."""
    if not activation_file:
        return "No file selected"

    try:
        metadata, _ = load_activations(activation_file)
        return f"**Prompt:**\n{metadata['prompt']}\n\n**Generated text:**\n{metadata['full_text']}"
    except:
        return "Error loading file"


def create_app():
    """Create Gradio app."""
    global emotions, layers

    emotions, layers = load_emotion_vectors()
    default_layer = layers[int(len(layers) * 0.67)]

    with gr.Blocks(title="Emotion Activation Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Emotion Activation Viewer")
        gr.Markdown(
            "Visualize how strongly each token activates different emotion concepts.\n\n"
            "**To generate new activations**, run in terminal:\n"
            "```\npython generate_activations.py \"Your prompt here\"\n```"
        )

        with gr.Row():
            with gr.Column(scale=1):
                available = get_available_activations()
                activation_dropdown = gr.Dropdown(
                    choices=available,
                    value=available[0] if available else None,
                    label="Activation File",
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh Files")

                emotion_dropdown = gr.Dropdown(
                    choices=emotions,
                    value="happy",
                    label="Emotion Concept",
                    interactive=True,
                )
                layer_dropdown = gr.Dropdown(
                    choices=layers,
                    value=default_layer,
                    label="Layer",
                    interactive=True,
                )

                text_display = gr.Markdown(label="Generated Text")

            with gr.Column(scale=2):
                output_html = gr.HTML(label="Visualization")

        # Event handlers
        def refresh_files():
            available = get_available_activations()
            return gr.update(choices=available, value=available[0] if available else None)

        refresh_btn.click(fn=refresh_files, outputs=[activation_dropdown])

        # Update visualization when any input changes
        for component in [activation_dropdown, emotion_dropdown, layer_dropdown]:
            component.change(
                fn=visualize,
                inputs=[activation_dropdown, emotion_dropdown, layer_dropdown],
                outputs=[output_html],
            )

        activation_dropdown.change(
            fn=show_text,
            inputs=[activation_dropdown],
            outputs=[text_display],
        )

        # Initial load
        app.load(
            fn=visualize,
            inputs=[activation_dropdown, emotion_dropdown, layer_dropdown],
            outputs=[output_html],
        )
        app.load(
            fn=show_text,
            inputs=[activation_dropdown],
            outputs=[text_display],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
