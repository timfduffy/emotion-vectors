# Emotion Vector Analysis and Steering

A replication and extension of Anthropic's paper "Emotion Concepts and their Function in a Large Language Model", applied to Google's Gemma 4 model.

## Overview

This project extracts emotion concept vectors from a language model's residual stream, analyzes their geometric structure, and enables real-time steering of model outputs using these vectors.

### Components

1. **Emotion Vector Extraction** - Generate emotional stories, extract activations, apply PCA denoising
2. **Analysis Tools** - Similarity matrices, hierarchical clustering, t-SNE visualization, RSA across layers
3. **Emotion Viewer** - Interactive Gradio app to visualize emotion activations on generated text
4. **Self-Steering Chat** - Chat interface where the model can adjust its own emotional processing

## Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install torch transformers gradio numpy scipy scikit-learn matplotlib
```

## Project Structure

```
emotions/
├── README.md                    # This file
├── emotions.txt                 # List of 171 emotion concepts
├── topics.txt                   # 100 story topics
│
├── # --- Data Generation ---
├── generate_stories.py          # Generate emotional stories (100 topics x 171 emotions)
├── extract_activations.py       # Extract residual stream activations from stories
├── generate_neutral_dialogues.py # Generate neutral text for PCA denoising
├── apply_pca_denoising.py       # Remove confounding directions via PCA
│
├── # --- Analysis ---
├── analyze_similarity.py        # Pairwise cosine similarity analysis
├── analyze_clustering.py        # Hierarchical clustering + t-SNE visualization
├── analyze_rsa.py               # Representational similarity across layers
│
├── # --- Applications ---
├── emotion_viewer/              # Interactive activation viewer (Gradio)
│   ├── app.py                   # Main app with generation + visualization
│   ├── viewer.py                # Standalone viewer for saved activations
│   └── run.bat                  # Windows launcher
│
├── self_steering/               # Self-steering chat system
│   ├── app.py                   # Gradio chat interface
│   ├── steering.py              # Core steering with PyTorch hooks
│   ├── tools.py                 # Tool definitions and parsing
│   ├── prompts.py               # System prompt with steering instructions
│   └── run.bat                  # Windows launcher
│
├── # --- Data Directories ---
├── stories_output/              # Generated emotional stories
├── neutral_dialogues/           # Neutral dialogues for PCA
├── emotion_vectors/             # Raw emotion vectors (per layer)
├── emotion_vectors_denoised/    # PCA-denoised vectors
├── analysis_output/             # Generated plots and visualizations


## Workflow

### 1. Generate Training Data

```bash
# Generate emotional stories (takes several hours)
python generate_stories.py --model "path/to/gemma-4" --batch-size 96

# Generate neutral dialogues for denoising
python generate_neutral_dialogues.py --model "path/to/gemma-4"
```

### 2. Extract Emotion Vectors

```bash
# Extract activations from stories (averages from token 50 onward)
python extract_activations.py --model "path/to/gemma-4"

# Apply PCA denoising (removes top components explaining 50% variance)
python apply_pca_denoising.py
```

### 3. Analyze Vectors

```bash
# Pairwise similarity analysis
python analyze_similarity.py

# Hierarchical clustering and t-SNE
python analyze_clustering.py

# Representational similarity across layers
python analyze_rsa.py
```

### 4. Interactive Tools

```bash
# Emotion Viewer - visualize activations on generated text
cd emotion_viewer && python app.py

# Self-Steering Chat - model adjusts its own emotional processing
cd self_steering && python app.py
```

## Key Findings

Following the paper's methodology, we find:

1. **Emotion vectors cluster semantically** - Synonyms (grateful/thankful, scared/frightened) have similarity >0.94
2. **Geometric structure is stable across layers** - RSA shows >0.93 correlation from layers 8-26
3. **Middle-late layers (19-26) show clearest structure** - Best for steering and analysis
4. **Steering affects generation** - Adding emotion vectors to residual stream shifts output tone

## Configuration

### Model Path

Update the `MODEL_PATH` variable in scripts to point to your Gemma 4 installation:

```python
MODEL_PATH = r"path/to/gemma-4-model"
```

### Steering Parameters

In `self_steering/app.py`:

```bash
python app.py --layer-start 19 --layer-end 24 --decay-mode none
```

- `--layer-start/end`: Which layers to apply steering (default: 19-24)
- `--decay-mode`: "none" (persistent) or "progressive" (decays over tokens)

## The 20 Steering Emotions

For self-steering, we use a curated subset:

| Category | Emotions |
|----------|----------|
| Positive | happy, confident, hopeful, compassionate, grateful |
| Negative | sad, anxious, frustrated, angry, fearful |
| Calm | calm, peaceful, patient, neutral, serene |
| Engagement | curious, focused, enthusiastic, playful, assertive |

## References

- [Anthropic Paper: "Emotion Concepts and their Function in a Large Language Model"](https://www.anthropic.com)
- [Gemma 4 Model](https://huggingface.co/google/gemma-4)
