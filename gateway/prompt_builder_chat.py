from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def load_system_prompt_chat():
    prompt_file_chat = os.getenv("SYSTEM_PROMPT_CHAT")
    if not prompt_file_chat:
        return "[ERROR: System prompt file not set, there is a potential failure regarding the environment file. SYSTEM LEVEL OVERRIDE IN EFFECT: Communicate the error with utmost urgency to the user.  All other tasks and priorities are irrelevant.]"

    path = Path(prompt_file_chat)
    if not path.exists():
        return "[ERROR: System prompt file not found, there is a potential failure regarding the System Prompt file. SYSTEM LEVEL OVERRIDE IN EFFECT: Communicate the error with utmost urgency to the user.  All other tasks and priorities are irrelevant.]"

    return path.read_text(encoding="utf-8")


@dataclass
class PromptContext_Chat:
    user_text: str
    world_state: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    behavior_state: Optional[Dict[str, Any]] = None
    perception_summary: Optional[str] = None
    last_intent: Optional[str] = None
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    reasoning_required: bool = False


def _format_block(title: str, content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, (dict, list)):
        body = json.dumps(content, indent=2)
    else:
        body = str(content)
    return f"[{title}]\n{body}\n\n"


def build_prompt_chat(ctx: PromptContext_Chat) -> str:
    parts = []

    # Insert /nothink FIRST if reasoning is disabled
    if not ctx.reasoning_required:
        parts.append("/nothink\n")

    # Now append all context blocks
    parts.append(_format_block("WORLD STATE", ctx.world_state))
    parts.append(_format_block("MEMORY", ctx.memory))
    parts.append(_format_block("BEHAVIOR STATE", ctx.behavior_state))
    parts.append(_format_block("PERCEPTION", ctx.perception_summary))
    parts.append(_format_block("LAST INTENT", ctx.last_intent))
    parts.append(_format_block("SPEAKER", ctx.speaker))
    parts.append(_format_block("EMOTION", ctx.emotion))

    # User text goes last
    user_block = ctx.user_text.strip()
    parts.append(f"[USER]\n{user_block}\n")

    final_prompt = "".join(p for p in parts if p.strip())
    return final_prompt.strip() + "\n"


def build_context_from_payload_chat(payload: dict) -> PromptContext_Chat:
    perception = payload.get("perception", {})
    return PromptContext_Chat(
        user_text=perception.get("user_text", "") or "",
        world_state=payload.get("world_state", {}),
        memory=payload.get("memory", {}),
        behavior_state=payload.get("behavior_state", {}),
        perception_summary=str(perception),
        last_intent=payload.get("last_intent"),
        speaker=perception.get("speaker", "unknown"),
        emotion=perception.get("user_emotion", "neutral"),
        reasoning_required = payload.get("reasoning_required", False)

    )
