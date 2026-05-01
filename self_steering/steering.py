"""
Core steering module for applying emotion vectors to model activations.

Uses PyTorch hooks to modify the residual stream during generation.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Optional, Callable
from dataclasses import dataclass, field


# Default curated lists per source label. For Emotions, we use the original
# 20 hand-picked concepts; for PCA, we expose all components (since there
# are only a handful, no curation is needed).
DEFAULT_EMOTION_CURATED = [
    "happy", "sad", "calm", "curious",
    "confident", "anxious", "peaceful", "focused",
    "hopeful", "frustrated", "patient", "enthusiastic",
    "compassionate", "angry", "neutral", "playful",
    "grateful", "fearful", "serene", "assertive",
]


@dataclass
class SteeringConfig:
    """Configuration for steering behavior."""
    layer_start: int = 19
    layer_end: int = 24
    decay_mode: str = "none"  # "none" or "progressive"
    decay_rate: float = 0.95  # multiplier per token for progressive decay

    # Single-source legacy field (used as fallback if `sources` is empty).
    vectors_dir: str = ""

    # Multi-source list of (label, path). When non-empty this takes
    # precedence over `vectors_dir`.
    sources: list = field(default_factory=list)

    # Curated whitelists per source label. For self-steer mode the model
    # may only steer toward concepts in the active source's curated list.
    # If a source isn't listed here, all of its concepts are considered
    # curated (i.e., no restriction).
    curated_per_source: dict = field(default_factory=dict)

    # Backward-compat shim: legacy code reads this directly. It now mirrors
    # the *active* source's curated list and is kept in sync by
    # SteeringManager.set_active_source().
    available_emotions: list = field(default_factory=lambda: list(DEFAULT_EMOTION_CURATED))


@dataclass
class SteeringState:
    """Current steering state."""
    emotion: Optional[str] = None
    strength: float = 0.0
    active: bool = False
    tokens_since_set: int = 0


class SteeringManager:
    """Manages steering vector application to model layers."""

    def __init__(self, model, config: SteeringConfig):
        self.model = model
        self.config = config
        self.state = SteeringState()
        self.hooks = []

        # Per-source state, populated by _load_vectors():
        #   sources_loaded[label] = {
        #     "emotion_vectors": {layer: {name: tensor}},
        #     "all_emotions":    [name, ...],
        #     "layers":          [layer_idx, ...],
        #     "path":            str,
        #   }
        self.sources_loaded: dict = {}
        self.active_source: str = ""

        # These three are kept pointing at the *active* source for backward
        # compatibility with code that touches them directly.
        self.emotion_vectors: dict = {}
        self.all_emotions: list = []
        self.all_available_layers: list = []

        # Load all configured sources
        self._load_vectors()

        # Get model layer structure
        self._setup_layers()

    def _resolve_sources(self) -> list[tuple[str, Path]]:
        """Pick which sources to load: explicit `sources` list wins, else fall
        back to the legacy single `vectors_dir`."""
        if self.config.sources:
            return [(label, Path(p)) for label, p in self.config.sources]
        if self.config.vectors_dir:
            return [("Emotions", Path(self.config.vectors_dir))]
        raise ValueError("SteeringConfig has neither `sources` nor `vectors_dir` set")

    def _load_vectors(self):
        """Load steering vectors from one or more source directories."""
        sources = self._resolve_sources()

        for label, path in sources:
            if not (path / "metadata.json").exists():
                print(f"  [skip] source '{label}': no metadata.json at {path}")
                continue
            with open(path / "metadata.json", "r") as f:
                metadata = json.load(f)
            layers = metadata["layers"]
            names = metadata["emotions"]

            vecs_by_layer: dict = {}
            for layer in layers:
                data = np.load(path / f"emotion_vectors_layer_{layer}.npz")
                vecs_by_layer[layer] = {
                    n: torch.tensor(data[n], dtype=torch.bfloat16)
                    for n in names
                    if n in data
                }

            self.sources_loaded[label] = {
                "emotion_vectors": vecs_by_layer,
                "all_emotions": names,
                "layers": layers,
                "path": str(path),
            }
            print(f"  [ok]   source '{label}': {len(layers)} layers, {len(names)} concepts ({path.name})")

        if not self.sources_loaded:
            raise RuntimeError("No valid steering vector sources loaded")

        # Activate the first source by default
        self.set_active_source(next(iter(self.sources_loaded)))

    def set_active_source(self, label: str):
        """Switch the active steering source. Clears any active steering."""
        if label not in self.sources_loaded:
            raise ValueError(f"Unknown source '{label}'. Loaded: {list(self.sources_loaded)}")

        src = self.sources_loaded[label]
        self.active_source = label
        self.emotion_vectors = src["emotion_vectors"]
        self.all_emotions = src["all_emotions"]
        self.all_available_layers = src["layers"]

        # Update the curated whitelist used in self-steer validation
        if label in self.config.curated_per_source:
            self.config.available_emotions = list(self.config.curated_per_source[label])
        else:
            # No explicit curation for this source -> expose everything
            self.config.available_emotions = list(self.all_emotions)

        # Switching sources invalidates any in-flight steering
        self.state = SteeringState()
        print(f"Active source: '{label}' ({len(self.all_emotions)} concepts, "
              f"{len(self.config.available_emotions)} curated)")

    def get_source_labels(self) -> list:
        """All loaded source labels in their original order."""
        return list(self.sources_loaded)

    def _setup_layers(self):
        """Identify model layers to hook."""
        # Handle different model architectures
        self.layers = None

        # Try various common paths to find layers
        paths_to_try = [
            lambda m: m.model.language_model.layers,  # Gemma 4 multimodal
            lambda m: m.model.language_model.model.layers,  # Gemma 4 multimodal (alt)
            lambda m: m.language_model.model.layers,  # Gemma 4 multimodal (direct)
            lambda m: m.model.model.layers,  # Some nested models
            lambda m: m.model.layers,  # Gemma, Llama
            lambda m: m.model.decoder.layers,  # Some decoder models
            lambda m: m.transformer.h,  # GPT-2 style
            lambda m: m.transformer.layers,  # Some transformer models
        ]

        for get_layers in paths_to_try:
            try:
                self.layers = get_layers(self.model)
                if self.layers is not None:
                    break
            except AttributeError:
                continue

        if self.layers is None:
            # Debug: print model structure
            print("Could not find layers. Model structure:")
            print(f"  model type: {type(self.model)}")
            if hasattr(self.model, 'model'):
                print(f"  model.model type: {type(self.model.model)}")
                print(f"  model.model attributes: {[a for a in dir(self.model.model) if not a.startswith('_')]}")
            raise ValueError("Could not find model layers")

        print(f"Found {len(self.layers)} model layers")

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            if not self.state.active:
                return output

            # Get the steering vector for this layer
            if layer_idx not in self.emotion_vectors:
                return output

            emotion_vecs = self.emotion_vectors[layer_idx]
            if self.state.emotion not in emotion_vecs:
                return output

            steering_vec = emotion_vecs[self.state.emotion].to(output[0].device)

            # Normalize the steering vector
            steering_vec = steering_vec / steering_vec.norm()

            # Calculate effective strength (with decay if enabled)
            effective_strength = self.state.strength
            if self.config.decay_mode == "progressive":
                effective_strength *= (self.config.decay_rate ** self.state.tokens_since_set)

            # Apply steering to the hidden states
            # output is typically (hidden_states, ...) tuple
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add steering vector to all positions
                modified = hidden_states + effective_strength * steering_vec

                # Debug: log first application per generation
                if not hasattr(self, '_logged_this_gen'):
                    self._logged_this_gen = True
                    delta_norm = (effective_strength * steering_vec).norm().item()
                    hidden_norm = hidden_states.norm().item() / hidden_states.numel() * 1000
                    print(f"  [Hook L{layer_idx}] Steering {self.state.emotion} @ {effective_strength:.1f}, delta_norm={delta_norm:.2f}")

                return (modified,) + output[1:]
            else:
                return output + effective_strength * steering_vec

        return hook

    def reset_generation_log(self):
        """Call before each generation to reset per-generation logging."""
        if hasattr(self, '_logged_this_gen'):
            del self._logged_this_gen

    def register_hooks(self):
        """Register forward hooks on the steering layers."""
        self.remove_hooks()  # Clear any existing hooks

        for layer_idx in range(self.config.layer_start, self.config.layer_end + 1):
            if layer_idx < len(self.layers):
                hook = self.layers[layer_idx].register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} steering hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_steering(self, emotion: Optional[str], strength: float = 0.0, validate_curated: bool = True):
        """Set the current steering state.

        Args:
            emotion: The emotion to steer toward (or None to clear)
            strength: Steering strength (positive = toward, negative = away)
            validate_curated: If True, only allow curated emotions (for self-steer mode)
        """
        if emotion is None or emotion.lower() == "none":
            self.state = SteeringState(active=False)
            print("Steering cleared")
        else:
            # Validate emotion exists
            if emotion not in self.all_emotions:
                raise ValueError(f"Unknown emotion: {emotion}")

            # For self-steer mode, also check it's in the curated list
            if validate_curated and emotion not in self.config.available_emotions:
                raise ValueError(f"Emotion '{emotion}' not in curated list. Available: {self.config.available_emotions}")

            self.state = SteeringState(
                emotion=emotion,
                strength=strength,
                active=True,
                tokens_since_set=0,
            )
            print(f"Steering set: {emotion} @ {strength}")

    def on_token_generated(self):
        """Called after each token is generated (for decay tracking)."""
        if self.state.active:
            self.state.tokens_since_set += 1

            # Check if decay has reduced strength to negligible
            if self.config.decay_mode == "progressive":
                effective = self.state.strength * (self.config.decay_rate ** self.state.tokens_since_set)
                if abs(effective) < 0.01:
                    self.state.active = False
                    print("Steering decayed to zero")

    def get_status(self) -> dict:
        """Get current steering status."""
        if not self.state.active:
            return {"active": False, "emotion": None, "strength": 0.0}

        effective_strength = self.state.strength
        if self.config.decay_mode == "progressive":
            effective_strength *= (self.config.decay_rate ** self.state.tokens_since_set)

        return {
            "active": True,
            "emotion": self.state.emotion,
            "strength": self.state.strength,
            "effective_strength": effective_strength,
            "tokens_since_set": self.state.tokens_since_set,
        }

    def set_layer_range(self, layer_start: int, layer_end: int):
        """Update the layer range and re-register hooks."""
        self.config.layer_start = layer_start
        self.config.layer_end = layer_end
        self.register_hooks()
        print(f"Layer range updated to {layer_start}-{layer_end}")

    def get_layer_range(self) -> tuple[int, int]:
        """Get the current layer range."""
        return self.config.layer_start, self.config.layer_end

    def get_num_model_layers(self) -> int:
        """Get total number of model layers."""
        return len(self.layers) if self.layers else 0

    def get_all_emotions(self) -> list:
        """Get all available emotions (for user-steer mode)."""
        return self.all_emotions

    def get_curated_emotions(self) -> list:
        """Get curated emotions (for self-steer mode)."""
        return self.config.available_emotions

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
