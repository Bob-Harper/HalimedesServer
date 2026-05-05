import os
import re
import json
import requests
import base64
import asyncio
from dotenv import load_dotenv
from modules.llm_module import LLMModule
from modules.sql_module import SQLModule
from modules.vosk_module import VoskModule
from aiohttp import web
from typing import cast
# Logging added below
import uuid
# import logging

# # Resolve the server root directory (one level above gateway/)
# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# # Correct logging directory at the server root
# LOG_DIR = os.path.join(ROOT_DIR, "logging")
# os.makedirs(LOG_DIR, exist_ok=True)

# log_path = os.path.join(LOG_DIR, "gateway.log")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
#     handlers=[
#         logging.FileHandler(log_path, mode="a", encoding="utf-8"),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger("gateway")

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
SYSTEM_PROMPT_INFERENCE = os.getenv("SYSTEM_PROMPT_INFERENCE")
GATEWAY_BIND_HOST = os.getenv("GATEWAY_BIND_HOST", "0.0.0.0")
GATEWAY_BIND_PORT = int(os.getenv("GATEWAY_BIND_PORT", 9000))
HALIMEDES_IP = os.getenv("HALIMEDES_IP")
HALIMEDES_PORT = int(os.getenv("HALIMEDES_PORT", 8123))

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

async def hardware_proxy(request):
    data = await request.json()
    components = data.get("components", [])

    url = f"http://{HALIMEDES_IP}:{HALIMEDES_PORT}/api/hardware"

    resp = requests.post(
        url,
        json={"components": components},
        timeout=2
    )

    return web.json_response(resp.json())


class HalServerGateway:
    def __init__(self):
        self.sql = SQLModule(DB_CONFIG)
        vosk_uri = f"ws://{os.getenv('VOSK_CONNECT_HOST')}:{os.getenv('VOSK_CONNECT_PORT')}"
        self.vosk = VoskModule(uri=vosk_uri)
        self.llm_infer = LLMModule("ws://localhost:8765")

    #  Transcription-Only endpoint
    async def handle_transcribe(self, request: web.Request):
        req_id = str(uuid.uuid4())

        # Read JSON
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": "json_parse_error", "detail": repr(e)}, status=400)

        # Extract base64
        audio_b64 = data.get("audio_b64")
        if not audio_b64:
            return web.json_response({"error": "missing_audio_b64"}, status=400)

        # Decode base64
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return web.json_response({"error": "b64_decode_error", "detail": repr(e)}, status=400)
        # Call Vosk
        try:
            transcript = await self.vosk.transcribe_audio(audio_bytes, req_id=req_id)

        except Exception as e:
            return web.json_response({"error": "vosk_exception", "detail": repr(e)}, status=500)

        # Build response
        text = transcript.get("text", "")

        words = cast(list[dict], transcript.get("result", []))
        if words:
            confidence = sum(w.get("conf", 0) for w in words) / len(words)
            duration = words[-1].get("end", None)
        else:
            confidence = None
            duration = None

        result = {
            "text": text,
            "confidence": confidence,
            "duration": duration,
        }

        return web.json_response(result)

    #  Inference/Instruction based endpoint
    async def handle_inference(self, request: web.Request):
        try:
            payload = await request.json()

            # HAL now sends the fully assembled prompt
            final_user_prompt = payload.get("prompt", "") or ""

            messages = [{"role": "user", "content": final_user_prompt}]

            llm_reply = await self.llm_infer.infer(model="medium", messages=messages)
            # logger.info(f"\n[handle_inference]\n{llm_reply}\n")
            content = llm_reply.get("response", "") or ""

            # Strip markdown fences
            content = content.replace("```json", "").replace("```", "").strip()

            # Strip <tool_call> blocks
            content = strip_think_blocks(content)

            try:
                parsed = json.loads(content)
                # logger.info(f"\n[handle_inference]\n{llm_reply}\n")

            except Exception:
                parsed = {}


            return web.json_response(parsed)

        except Exception as e:
            # logger.exception("handle_inference failed")

            # Always return a valid cognition packet so HAL never breaks
            fallback = {
                "intent": "error",
                "speech": ["This"],
                "actions": ["is"],
                "memory_updates": ["a"],
                "world_updates": ["fail"],
                "error_detail": str(e)
            }

            return web.json_response(fallback)

    async def handle_semantic_write(self, request):
        data = await request.json()
        key = data.get("key")
        value = data.get("value")

        if not key or value is None:
            return web.json_response({"error": "key and value required"}, status=400)

        result = await self.sql.semantic_write(key, value)
        return web.json_response(result)

    async def handle_semantic_read(self, request):
        data = await request.json()
        key = data.get("key")

        if not key:
            return web.json_response({"error": "key required"}, status=400)

        result = await self.sql.semantic_read(key)
        return web.json_response(result)

    async def handle_vector_write(self, request):
        data = await request.json()

        content = data.get("content")
        vector = data.get("vector")
        timestamp = data.get("timestamp")

        if not content or vector is None or timestamp is None:
            return web.json_response({"error": "content, vector, timestamp required"}, status=400)

        result = await self.sql.vector_write(content, vector, timestamp)
        return web.json_response(result)

    async def handle_vector_search(self, request):
        data = await request.json()

        query_vector = data.get("vector")
        top_k = data.get("top_k", 5)

        if query_vector is None:
            return web.json_response({"error": "vector required"}, status=400)

        result = await self.sql.vector_search(query_vector, top_k)
        return web.json_response(result)

    # -------------------------
    # Server  Endpoints
    # -------------------------

    async def start_http_server(self):
        app = web.Application(client_max_size=10 * 1024 * 1024)
        # Vosk transcribe endpoint (forwarding to Vosk server and returning text + confidence)
        app.router.add_post("/api/transcribe", self.handle_transcribe)

        # Hardware proxy endpoint (forwards to Halimedes)
        app.router.add_post("/api/hardware", hardware_proxy)

        # LLM inference endpoint (HAL sends fully assembled prompt, LLM returns structured JSON)
        app.router.add_post("/api/inference", self.handle_inference)

        # Semantic memory endpoints (Hard data. searchable by explicit tags and content, but not vector similarity)
        app.router.add_post("/api/memory/semantic/write", self.handle_semantic_write)
        app.router.add_post("/api/memory/semantic/read", self.handle_semantic_read)

        # Episodic memory endpoints (Objective memory, conversations, user details. Searchable by vector similarity and optionally tags/content)
        app.router.add_post("/api/memory/episodic/write", self.handle_vector_write)
        app.router.add_post("/api/memory/episodic/search", self.handle_vector_search)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, GATEWAY_BIND_HOST, GATEWAY_BIND_PORT)
        # print(f"[Gateway] HTTP server on {GATEWAY_BIND_HOST}:{GATEWAY_BIND_PORT}")
        await site.start()


async def main():
    gateway = HalServerGateway()
    await gateway.start_http_server()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
