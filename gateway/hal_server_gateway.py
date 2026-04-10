# gateway/hal_server_gateway.py
import os
import json
import base64
import asyncio
import websockets
from dotenv import load_dotenv
from modules.llm_module import LLMModule
from modules.sql_module import SQLModule
from modules.vosk_module import VoskModule
import tempfile
import subprocess

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


class HalServerGateway:
    def __init__(self):
        self.sql = SQLModule(DB_CONFIG)
        self.vosk = VoskModule()
        self.llm = LLMModule(LLM_SERVER_URL)
        self.handlers = {}
        self.register_handlers()

    async def async_init(self):
        await self.sql.init()

    # -------------------------
    # Handler registration
    # -------------------------
    def register_handler(self, packet_type: str, func):
        self.handlers[packet_type] = func

    def register_handlers(self):
        self.register_handler("speech.transcribe", self.handle_speech_transcribe)
        self.register_handler("memory.store", self.handle_memory_store)
        self.register_handler("memory.query", self.handle_memory_query)
        self.register_handler("llm.chat", self.handle_llm_chat)
        self.register_handler("pipeline.process_audio", self.handle_pipeline_process_audio)

    # -------------------------
    # Handlers
    # -------------------------
    async def handle_speech_transcribe(self, payload: dict):
        audio = payload.get("audio")
        return await self.vosk.transcribe(audio)

    async def handle_memory_store(self, payload: dict):
        text = payload.get("text")
        tags = payload.get("tags", [])
        return await self.sql.store(text, tags)

    async def handle_memory_query(self, payload: dict):
        query = payload.get("query")
        return await self.sql.query(query)

    async def handle_llm_chat(self, payload: dict):
        model = payload.get("model", "medium")
        messages = payload.get("messages", [])
        # You can pass extra kwargs if needed (temperature, etc.)
        return await self.llm.chat(model=model, messages=messages)

    async def handle_pipeline_process_audio(self, payload: dict):
        audio_b64 = payload.get("audio")
        if not audio_b64:
            return {"error": "No audio provided"}

        # 1. Decode base64 → raw PCM bytes (normalized)
        audio_bytes = base64.b64decode(audio_b64)

        pcm = await self.normalize_to_pcm(audio_bytes)
        transcript = await self.vosk.transcribe(pcm)

        text = transcript.get("text", "")

        # 3. LLM response
        messages = [
            {"role": "user", "content": f"/nothink Respond to this in a conversational manner:\n\n{text}"}
        ]
        llm_reply = await self.llm.chat(model="medium", messages=messages)
        reply_text = llm_reply.get("response", "")

        # 4. Store memory
        mem_result = await self.sql.store(
            text=f"Heard: {text}\nReplied: {reply_text}",
            tags=["pipeline_test"]
        )
        mem_id = mem_result.get("id") if isinstance(mem_result, dict) else None


        # 5. Unified result
        return {
            "heard": text,
            "reply": reply_text,
            "memory_id": mem_id
        }


    # -------------------------
    # Preprocessors and Normalizers
    # -------------------------


    async def normalize_to_pcm(self, audio_bytes: bytes) -> bytes:
        # Write input bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            in_path = tmp_in.name

        # Prepare output temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
            out_path = tmp_out.name

        # Convert ANYTHING → PCM16 mono 44.1k → raw bytes (NO HEADER)
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", in_path,
            "-ac", "1",
            "-ar", "44100",
            "-f", "s16le",
            out_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read normalized PCM
        with open(out_path, "rb") as f:
            pcm = f.read()

        os.remove(in_path)
        os.remove(out_path)

        return pcm


    # -------------------------
    # WebSocket server
    # -------------------------
    async def handle_client(self, websocket):
        addr = websocket.remote_address
        print(f"[Gateway] Client connected: {addr}")

        try:
            async for raw in websocket:
                try:
                    packet = json.loads(raw)
                    ptype = packet.get("type")
                    payload = packet.get("payload", {})

                    handler = self.handlers.get(ptype)
                    if not handler:
                        response = {
                            "type": ptype,
                            "ok": False,
                            "error": f"Unknown packet type: {ptype}"
                        }
                    else:
                        result = await handler(payload)
                        response = {
                            "type": ptype,
                            "ok": True,
                            "result": result
                        }

                except Exception as e:
                    response = {
                        "type": packet.get("type") if 'packet' in locals() else "unknown",
                        "ok": False,
                        "error": str(e)
                    }

                await websocket.send(json.dumps(response))

        finally:
            print(f"[Gateway] Client disconnected: {addr}")

    async def start(self):
        print(f"[Gateway] Listening on {GATEWAY_BIND_HOST}:{GATEWAY_BIND_PORT}")
        async with websockets.serve(self.handle_client, GATEWAY_BIND_HOST, GATEWAY_BIND_PORT):
            await asyncio.Future()  # run forever


async def main():
    gateway = HalServerGateway()
    await gateway.async_init()   # initialize async SQL
    await gateway.start()        # start websocket server

if __name__ == "__main__":
    asyncio.run(main())
