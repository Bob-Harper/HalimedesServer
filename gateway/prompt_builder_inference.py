from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def load_system_prompt_inference():
    prompt_file = os.getenv("SYSTEM_PROMPT_INFERENCE")
    if not prompt_file:
        return "[ERROR: SYSTEM_PROMPT_INFERENCE not set.]"

    path = Path(prompt_file)
    if not path.exists():
        return "[ERROR: SYSTEM_PROMPT_INFERENCE file not found.]"

    return path.read_text(encoding="utf-8")


@dataclass
class PromptContext_Inference:
    user_text: str
    world_state: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    behavior_state: Optional[Dict[str, Any]] = None
    perception_summary: Optional[str] = None
    last_intent: Optional[str] = None
    speaker: Optional[str] = None
    emotion: Optional[str] = None


def build_prompt_inference(ctx: PromptContext_Inference) -> str:
    """
    Build a RAW inference prompt.
    No roles, no blocks, no /nothink, no chat structure.
    """
    parts = []

    # Include world/memory/behavior context as plain JSON (optional)
    if ctx.world_state:
        parts.append("WORLD_STATE:\n" + json.dumps(ctx.world_state, indent=2) + "\n")

    if ctx.memory:
        parts.append("MEMORY:\n" + json.dumps(ctx.memory, indent=2) + "\n")

    if ctx.behavior_state:
        parts.append("BEHAVIOR_STATE:\n" + json.dumps(ctx.behavior_state, indent=2) + "\n")

    # User instruction goes last
    parts.append("INSTRUCTION:\n" + ctx.user_text.strip() + "\n")

    return "\n".join(parts).strip()


def build_context_from_payload_inference(payload: dict) -> PromptContext_Inference:
    perception = payload.get("perception", {})
    return PromptContext_Inference(
        user_text=perception.get("user_text", "") or "",
        world_state=payload.get("world_state", {}),
        memory=payload.get("memory", {}),
        behavior_state=payload.get("behavior_state", {}),
        perception_summary=str(perception),
        last_intent=payload.get("last_intent"),
        speaker=perception.get("speaker", "unknown"),
        emotion=perception.get("user_emotion", "neutral"),
    )
