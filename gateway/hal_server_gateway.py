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
from prompt_builder_inference import build_prompt_inference, build_context_from_payload_inference, load_system_prompt_inference
# Logging added below
import time
import uuid
import logging

LOG_DIR = os.path.join(os.path.dirname(__file__), "logging")
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "gateway.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler()  # optional: remove if you don't want terminal output
    ]
)

logger = logging.getLogger("gateway")

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
        self.llm_infer = LLMModule("ws://localhost:8766")


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
            final_user_prompt = build_prompt_inference(ctx)
            system_prompt = load_system_prompt_inference()

            # --- RAW INFERENCE CALL -------------------------------
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
    async def original_handle_transcribe(self, request: web.Request):
        try:
            data = await request.json()

            audio_b64 = data.get("audio_b64")
            if not audio_b64:
                return web.json_response({"error": "missing audio_b64"}, status=400)

            # Decode base64 → bytes
            audio_bytes = base64.b64decode(audio_b64)

            # VoskModule now handles normalization internally
            transcript = await self.vosk.transcribe_audio(audio_bytes, req_id=req_id)

            text = transcript.get("text", "")

            result = {
                "text": text
            }

            return web.json_response(result)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    #  Transcription-Only endpoint EXTREME DEBUG EDITION
    async def handle_transcribe(self, request: web.Request):
        req_id = str(uuid.uuid4())
        t0 = time.monotonic()

        logger.info(f"[Gateway] [{req_id}] /api/transcribe START")

        # -----------------------------
        # 1. Read JSON
        # -----------------------------
        try:
            data = await request.json()
            logger.info(f"[Gateway] [{req_id}] JSON received: keys={list(data.keys())}")
        except Exception as e:
            logger.error(f"[Gateway] [{req_id}] JSON parse error: {repr(e)}")
            return web.json_response({"error": "json_parse_error", "detail": repr(e)}, status=400)

        # -----------------------------
        # 2. Extract base64
        # -----------------------------
        audio_b64 = data.get("audio_b64")
        if not audio_b64:
            logger.error(f"[Gateway] [{req_id}] Missing audio_b64")
            return web.json_response({"error": "missing_audio_b64"}, status=400)

        logger.info(f"[Gateway] [{req_id}] audio_b64 length={len(audio_b64)}")

        # -----------------------------
        # 3. Decode base64
        # -----------------------------
        try:
            audio_bytes = base64.b64decode(audio_b64)
            logger.info(f"[Gateway] [{req_id}] Decoded audio bytes={len(audio_bytes)}")
        except Exception as e:
            logger.error(f"[Gateway] [{req_id}] Base64 decode error: {repr(e)}")
            return web.json_response({"error": "b64_decode_error", "detail": repr(e)}, status=400)

        # -----------------------------
        # 4. Call Vosk
        # -----------------------------
        try:
            t_vosk0 = time.monotonic()
            transcript = await self.vosk.transcribe_audio(audio_bytes, req_id=req_id)
            t_vosk1 = time.monotonic()

            logger.info(f"[Gateway] [{req_id}] Vosk returned in {(t_vosk1 - t_vosk0)*1000:.2f} ms")
            logger.info(f"[Gateway] [{req_id}] Vosk transcript preview: {str(transcript)[:300]}")
        except Exception as e:
            logger.error(f"[Gateway] [{req_id}] Vosk exception: {repr(e)}")
            return web.json_response({"error": "vosk_exception", "detail": repr(e)}, status=500)

        # -----------------------------
        # 5. Build response
        # -----------------------------
        text = transcript.get("text", "")
        result = {
            "text": text,
            "speaker": "unknown",
            "emotion": "neutral"
        }

        t1 = time.monotonic()
        logger.info(f"[Gateway] [{req_id}] /api/transcribe END total={(t1 - t0)*1000:.2f} ms")
        logger.info(f"[Gateway] [{req_id}] Response: {result}")

        return web.json_response(result)

    # -------------------------
    # Server  Endpoints
    # -------------------------


    async def start_http_server(self):
        app = web.Application(client_max_size=10 * 1024 * 1024)

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
