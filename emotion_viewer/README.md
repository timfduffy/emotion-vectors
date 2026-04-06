# Emotion Viewer

Interactive Gradio application for visualizing emotion concept activations in language model outputs.

## Features

- **Generate and Analyze**: Enter a prompt, generate a response, and see token-level emotion activations
- **Multiple Display Modes**:
  - **Single**: One emotion, one layer - full text with highlighting
  - **Multi-layer**: One emotion across all 35 layers
  - **Multi-concept**: All 171 emotions for one layer
  - **Layer range**: All emotions averaged across a layer range
- **Dynamic Text Color**: Automatically adjusts text color (black/white) based on background
- **Hover Tooltips**: See exact projection values for each token

## Usage

### Quick Start

```bash
# Windows
run.bat

# Or directly
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

### Interface

1. **Left Panel**:
   - Enter your prompt
   - Adjust max tokens and temperature
   - Click "Generate & Analyze"

2. **Right Panel**:
   - Select display mode
   - Choose emotion concept (171 options)
   - Select layer(s) to visualize
   - Changes update instantly without regenerating

### Color Scale

- **Blue** = Negative projection (text activates AWAY from this emotion)
- **White** = Neutral (near zero projection)
- **Red** = Positive projection (text activates TOWARD this emotion)

## Files

- `app.py` - Main Gradio application with model loading and generation
- `viewer.py` - Standalone viewer for pre-saved activations (no model needed)
- `generate_activations.py` - CLI tool to generate and save activations
- `run.bat` - Windows launcher for app.py

## How It Works

1. **Generation**: Model generates text with `output_hidden_states=True`
2. **Activation Capture**: Residual stream activations saved for all tokens and layers
3. **Projection**: Each token's activation is projected onto emotion vectors via dot product
4. **Visualization**: Projections normalized and mapped to diverging colormap

## Technical Details

- Uses denoised emotion vectors from `emotion_vectors_denoised/`
- Applies chat template formatting for Gemma 4
- Caches activations for instant mode/layer switching
- Handles newlines as `\n` in multi-row displays for compactness
