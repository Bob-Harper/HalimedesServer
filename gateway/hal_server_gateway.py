import os
import re
import json
import base64
import asyncio
from dotenv import load_dotenv
from modules.llm_module import LLMModule
from modules.sql_module import SQLModule
from modules.vosk_module import VoskModule
import tempfile
import subprocess
from aiohttp import web
from prompt_builder_chat import build_prompt_chat, build_context_from_payload_chat, load_system_prompt_chat
from prompt_builder_inference import build_prompt_inference, build_context_from_payload_inference, load_system_prompt_inference
DEBUG_LOG = "logging/debug/debug.txt"

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(ENV_PATH)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

LLM_CONNECT_HOST = os.getenv("LLM_CONNECT_HOST", "127.0.0.1")
LLM_CONNECT_PORT = int(os.getenv("LLM_CONNECT_PORT", 8765))
LLM_SERVER_URL = f"ws://{LLM_CONNECT_HOST}:{LLM_CONNECT_PORT}"

GATEWAY_BIND_HOST = os.getenv("GATEWAY_BIND_HOST", "0.0.0.0")
GATEWAY_BIND_PORT = int(os.getenv("GATEWAY_BIND_PORT", 9000))


def debug_log(*lines):
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")
        f.write("\n")

def strip_think_blocks(text: str) -> str:
    """
    NOTE:
    Qwen ALWAYS emits <think>...</think> blocks.
    We ALWAYS strip them (inclusive) before JSON parsing.
    Empty block = model obeyed /nothink (correct).
    This is REQUIRED for both reasoning and no-reasoning modes and safe for non reasoning models.
    Do NOT remove this logic — the LLM output contract depends on it.
    """
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def extract_speech_text(llm_reply):
    raw_reply = llm_reply.get("response", "") or ""
    reply_text = raw_reply.strip()
    clean_reply = strip_think_blocks(reply_text)

    try:
        model_packet = json.loads(clean_reply)
        speech_text = model_packet.get("speech", {}).get("text", "")
    except Exception:
        speech_text = clean_reply

    return speech_text, clean_reply


class HalServerGateway:
    def __init__(self):
        self.sql = SQLModule(DB_CONFIG)
        vosk_uri = f"ws://{os.getenv('VOSK_CONNECT_HOST')}:{os.getenv('VOSK_CONNECT_PORT')}"
        self.vosk = VoskModule(uri=vosk_uri)
        self.llm_chat = LLMModule("ws://localhost:8765")
        self.llm_infer = LLMModule("ws://localhost:8766")

    #  Conversation based endpoint
    async def handle_chat(self, request: web.Request):
        try:
            payload = await request.json()

            # --- Extract perception -----------------------------------------
            perception = payload.get("perception", {})
            user_text = perception.get("user_text", "") or ""
            user_emotion = perception.get("user_emotion", "neutral")
            speaker = perception.get("speaker", "unknown")

            # --- Build LLM prompt -------------------------------------------
            ctx = build_context_from_payload_chat(payload)

            # Build ONLY the user-facing prompt
            final_user_prompt = build_prompt_chat(ctx)
            system_prompt = load_system_prompt_chat()

            llm_reply = await self.llm_chat.chat(
                model="medium",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_prompt},
                ],
            )

            speech_text, clean_reply = extract_speech_text(llm_reply)

            # --- Nonverbal suggestion ---------------------------------------
            nonverbal = {
                "gaze": "user",
                "expression": "neutral",
            }

            # --- Memory write instructions ----------------------------------
            memory_write = []
            if user_text or clean_reply:
                memory_write.append(
                    {
                        "type": "conversation_log",
                        "speaker": speaker,
                        "user_text": user_text,
                        "reply_text": speech_text,
                        "tags": ["conversation"],
                    }
                )

            # --- World state updates ----------------------------------------
            world_state_update = []

            # --- Unified response -------------------------------------------
            intent = "conversation"

            result = {
                "intent": intent,
                "speech": {
                    "utterances": [
                        {
                            "text": speech_text,
                            "emotion": user_emotion or "neutral",
                            "style": "default",
                            "priority": 1,
                        }
                    ]
                    if clean_reply
                    else []
                },
                "nonverbal": nonverbal,
                "memory": {
                    "write": memory_write,
                },
                "world_state": {
                    "update": world_state_update,
                },
            }

            return web.json_response(result)

        except Exception as e:
            return web.json_response(
                {
                    "error": "gateway handle_chat failure",
                    "message": str(e),
                },
                status=500,
            )

    #  Inference/Instruction based endpoint
    async def handle_inference(self, request: web.Request):
        try:
            payload = await request.json()

            # --- Extract perception -----------------------------------------
            perception = payload.get("perception", {})
            user_text = perception.get("user_text", "") or ""
            user_emotion = perception.get("user_emotion", "neutral")
            speaker = perception.get("speaker", "unknown")

            # --- Build inference prompt -------------------------------------
            ctx = build_context_from_payload_inference(payload)

            # Build ONLY the raw inference prompt (no roles)
            final_user_prompt = build_prompt_inference(ctx)
            system_prompt = load_system_prompt_inference()

            # --- RAW INFERENCE CALL (NOT CHAT) -------------------------------
            llm_reply = await self.llm_infer.infer(
                model="medium",
                system_prompt=system_prompt,
                user_prompt=final_user_prompt
            )

            # llm_reply["response"] contains the raw JSON string
            content = llm_reply.get("response", "")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {}

            # --- Unified response -------------------------------------------
            # Inference mode returns the FINAL JSON, so we just pass it through.
            return web.json_response(parsed)

        except Exception as e:
            return web.json_response(
                {
                    "error": "gateway_inference_failure",
                    "message": str(e),
                },
                status=500,
            )

    #  Transcription-Only endpoint
    async def handle_transcribe(self, request: web.Request):
        try:
            data = await request.json()

            audio_b64 = data.get("audio_b64")
            if not audio_b64:
                return web.json_response({"error": "missing audio_b64"}, status=400)

            # Decode base64 → bytes
            audio_bytes = base64.b64decode(audio_b64)

            # VoskModule now handles normalization internally
            transcript = await self.vosk.transcribe_audio(audio_bytes)

            text = transcript.get("text", "")

            result = {
                "text": text,
                "speaker": "unknown",
                "emotion": "neutral"
            }

            return web.json_response(result)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # -------------------------
    # Server  Endpoints
    # -------------------------


    async def start_http_server(self):
        app = web.Application(client_max_size=10 * 1024 * 1024)

        app.router.add_post("/api/chat", self.handle_chat)
        app.router.add_post("/api/inference", self.handle_inference)
        app.router.add_post("/api/transcribe", self.handle_transcribe)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, GATEWAY_BIND_HOST, GATEWAY_BIND_PORT)
        print(f"[Gateway] HTTP server on {GATEWAY_BIND_HOST}:{GATEWAY_BIND_PORT}")
        await site.start()


async def main():
    gateway = HalServerGateway()
    # await gateway.async_init()
    await gateway.start_http_server()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
