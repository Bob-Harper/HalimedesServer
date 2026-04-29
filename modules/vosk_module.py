import asyncio
import websockets
import json
import logging

logger = logging.getLogger("VoskModule")

class VoskModule:
    def __init__(self, uri="ws://localhost:2700"):
        self.uri = uri

    async def transcribe_audio(self, audio_bytes: bytes, req_id: str | None = None):
        if req_id is None:
            req_id = "noid"

        # Strip WAV header if present
        if audio_bytes.startswith(b"RIFF"):
            pcm = audio_bytes[44:]
        else:
            pcm = audio_bytes

        logger.info(f"[VoskModule] [{req_id}] PCM bytes={len(pcm)}")

        async with websockets.connect(self.uri) as ws:
            logger.info(f"[VoskModule] [{req_id}] Connected to {self.uri}")

            chunk_size = 4000
            for i in range(0, len(pcm), chunk_size):
                chunk = pcm[i:i+chunk_size]
                await ws.send(chunk)
                logger.debug(f"[VoskModule] [{req_id}] Sent chunk {i//chunk_size} ({len(chunk)} bytes)")

            await ws.send('{"eof":1}')
            logger.info(f"[VoskModule] [{req_id}] Sent EOF")

            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info(f"[VoskModule] [{req_id}] Connection closed cleanly with no final result")
                    return {"error": "no_final_result", "text": ""}

                # logger.debug(f"[VoskModule] [{req_id}] Received raw: {msg[:200]}")
                data = json.loads(msg)

                if "result" in data:
                    logger.info(f"[VoskModule] [{req_id}] FINAL: {str(data)[:500]}")
                    return data
