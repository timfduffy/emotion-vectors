"""
Tool definitions and execution for self-steering.
"""

import json
import re
from typing import Optional, Tuple


# Tool schema for the steer function
STEER_TOOL_SCHEMA = {
    "name": "steer",
    "description": "Adjust your emotional processing by applying a steering vector. This modifies how you generate subsequent text by nudging your internal representations toward or away from an emotion. Use positive strength to steer toward an emotion, negative to steer away. The effect persists until you call steer again or clear it with emotion='none'.",
    "parameters": {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": "The emotion to steer toward/away from, or 'none' to clear steering",
                "enum": [
                    "happy", "sad", "calm", "curious",
                    "confident", "anxious", "peaceful", "focused",
                    "hopeful", "frustrated", "patient", "enthusiastic",
                    "compassionate", "angry", "neutral", "playful",
                    "grateful", "fearful", "serene", "assertive",
                    "none"
                ]
            },
            "strength": {
                "type": "number",
                "description": "Steering strength. Positive values steer toward the emotion, negative values steer away. Typical range is -2.0 to 2.0, with 0.5 being a moderate effect."
            }
        },
        "required": ["emotion", "strength"]
    }
}


def parse_tool_call(text: str) -> Optional[Tuple[str, dict]]:
    """
    Parse a tool call from model output.

    Supports multiple formats:
    - JSON block: ```tool\n{"name": "steer", "arguments": {...}}\n```
    - Bare JSON: {"name": "steer", "arguments": {...}}
    - Function call: steer(emotion="calm", strength=0.5)
    - XML-style: <tool_call>{"name": "steer", ...}</tool_call>
    """
    # Try JSON block format
    json_match = re.search(r'```(?:tool|json)?\s*\n?\s*(\{.*?\})\s*\n?```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "name" in data and data["name"] == "steer":
                args = data.get("arguments", data.get("parameters", {}))
                return ("steer", args)
        except json.JSONDecodeError:
            pass

    # Try bare JSON format: {"name": "steer", "arguments": {...}}
    bare_json_match = re.search(r'\{"name":\s*"steer"[^}]*"arguments":\s*\{[^}]*\}\s*\}', text)
    if bare_json_match:
        try:
            data = json.loads(bare_json_match.group(0))
            if data.get("name") == "steer":
                args = data.get("arguments", data.get("parameters", {}))
                return ("steer", args)
        except json.JSONDecodeError:
            pass

    # Try simpler bare JSON (might have nested braces)
    bare_json_match2 = re.search(r'(\{"name":\s*"steer".*?\})\s*\}', text, re.DOTALL)
    if bare_json_match2:
        try:
            # Add closing brace since regex might not capture it
            json_str = bare_json_match2.group(0)
            data = json.loads(json_str)
            if data.get("name") == "steer":
                args = data.get("arguments", data.get("parameters", {}))
                return ("steer", args)
        except json.JSONDecodeError:
            pass

    # Try function call format: steer(emotion="calm", strength=0.5)
    func_match = re.search(r'steer\s*\(\s*emotion\s*=\s*["\'](\w+)["\']\s*,\s*strength\s*=\s*([-\d.]+)\s*\)', text)
    if func_match:
        return ("steer", {
            "emotion": func_match.group(1),
            "strength": float(func_match.group(2))
        })

    # Try XML-style
    xml_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if xml_match:
        try:
            data = json.loads(xml_match.group(1))
            if data.get("name") == "steer":
                return ("steer", data.get("arguments", {}))
        except json.JSONDecodeError:
            pass

    return None


def format_tool_result(emotion: str, strength: float, success: bool, message: str = "") -> str:
    """Format the result of a tool call for the model."""
    if success:
        if emotion.lower() == "none":
            return f"[Steering cleared. Your responses will now use baseline processing.]"
        else:
            direction = "toward" if strength > 0 else "away from"
            return f"[Steering active: {abs(strength):.1f}x {direction} '{emotion}'. This affects your subsequent responses until changed.]"
    else:
        return f"[Steering failed: {message}]"


DEFAULT_CONCEPT_SECTION = """### Available emotions:

**Positive:** happy, confident, hopeful, compassionate, grateful
**Negative:** sad, anxious, frustrated, angry, fearful
**Calm/Neutral:** calm, peaceful, patient, neutral, serene
**Engagement:** curious, focused, enthusiastic, playful, assertive"""


def get_tool_description(concept_section: str = DEFAULT_CONCEPT_SECTION) -> str:
    """Build the steer-tool description for the system prompt.

    `concept_section` is a Markdown blob (typically starting with a heading
    like "### Available emotions:") that lists the concepts the model may
    pass as the `emotion` argument. The caller supplies this so the same
    tool description adapts to whichever vector source is active (raw
    emotions, PCA components, etc.).
    """
    return f"""
## Steering Tool

You have access to a `steer` tool that allows you to adjust your emotional processing in real-time. When you call this tool, it modifies how you generate text by applying an "emotion vector" to your internal representations.

### How to use it:

Call the steer function with an emotion and strength:

```tool
{{"name": "steer", "arguments": {{"emotion": "calm", "strength": 0.5}}}}
```

### Parameters:

- **emotion**: One of the available concepts (listed below), or "none" to clear
- **strength**: A number indicating intensity
  - Positive values steer TOWARD the concept
  - Negative values steer AWAY from the concept
  - Typical range: -20.0 to 20.0
  - 1-3 = subtle effect
  - 3-7 = moderate effect
  - 7-15 = strong effect
  - 15-20 = very strong effect (may affect coherence)

{concept_section}

### When to use steering:

- When you want to adjust your tone or emotional approach
- When the conversation context calls for a different emotional register
- When you notice your responses might benefit from more/less of a quality
- To experiment with how different states affect your responses

### Important notes:

- Steering persists until you change it or clear it with emotion="none"
- You can see the effect in your own responses after calling steer
- Be thoughtful about when and why you adjust steering
- The effect is on your generation, not on your understanding
"""
