# Self-Steering Chat

A chat interface where the language model can adjust its own emotional processing in real-time using steering vectors.

## Concept

The model has access to a `steer` tool that modifies its residual stream during generation. When called, emotion vectors are added to the hidden states at specified layers, nudging the model's outputs toward or away from emotional states.

## Features

- **Self-Directed Steering**: Model decides when and how to adjust its emotional processing
- **20 Curated Emotions**: Balanced set of positive, negative, calm, and engagement emotions
- **Configurable Layers**: Choose which layers to apply steering (default: 19-24)
- **Persistence**: Steering remains active until explicitly changed
- **Debug Logging**: Terminal shows when steering hooks fire

## Usage

### Quick Start

```bash
# Windows
run.bat

# Or with options
python app.py --layer-start 19 --layer-end 24 --decay-mode none
```

Open http://127.0.0.1:7860 and start chatting.

### Example Prompts

- "Can you try steering toward curiosity and tell me something interesting?"
- "Steer toward calm and explain a complex topic"
- "Try steering away from anxious (negative strength) and give me advice"
- "How does it feel different when you steer toward confident vs peaceful?"

### Steering Tool Format

The model calls the tool with JSON:

```json
{"name": "steer", "arguments": {"emotion": "curious", "strength": 10.0}}
```

### Strength Guidelines

| Range | Effect |
|-------|--------|
| 1-3 | Subtle |
| 3-7 | Moderate |
| 7-15 | Strong |
| 15-20 | Very strong |

Negative values steer AWAY from the emotion.

## Available Emotions

| Category | Emotions |
|----------|----------|
| **Positive** | happy, confident, hopeful, compassionate, grateful |
| **Negative** | sad, anxious, frustrated, angry, fearful |
| **Calm** | calm, peaceful, patient, neutral, serene |
| **Engagement** | curious, focused, enthusiastic, playful, assertive |

## Files

- `app.py` - Gradio chat interface
- `steering.py` - Core steering logic with PyTorch forward hooks
- `tools.py` - Tool schema, parsing, and result formatting
- `prompts.py` - System prompt explaining steering to the model
- `run.bat` - Windows launcher

## How It Works

1. **Hooks Registered**: Forward hooks attached to layers 19-24
2. **Model Generates**: During generation, hooks intercept hidden states
3. **Steering Applied**: If active, emotion vector × strength added to hidden states
4. **Tool Parsing**: App detects tool calls in output and executes them
5. **State Persists**: Steering remains until model calls `steer(emotion="none")`

## Configuration

```bash
python app.py [OPTIONS]

Options:
  --layer-start INT    First layer to steer (default: 19)
  --layer-end INT      Last layer to steer (default: 24)
  --decay-mode STR     "none" (persistent) or "progressive" (decays)
  --decay-rate FLOAT   Decay multiplier per token (default: 0.95)
```

## Verifying Steering Works

1. **Check Terminal**: Look for `[Hook L19] Steering curious @ 18.0, delta_norm=X.XX`
2. **Adversarial Test**: Steer toward "sad", ask for something joyful
3. **Extreme Strength**: Try strength 50+ to see obvious effects
4. **Use Emotion Viewer**: Analyze steered outputs to measure projection differences

## Technical Notes

- Steering vectors are normalized before application
- Hidden states modified in-place during forward pass
- Works with Gemma 4 multimodal architecture (hooks on language model layers)
- System prompt explains the tool; model learns to use it contextually
