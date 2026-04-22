import asyncio
import websockets
import json
import tempfile
import subprocess
import os
import logging
logger = logging.getLogger("VoskModule")
TARGET_RATE = "16000"

class VoskModule:
    def __init__(self, uri="ws://localhost:2700"):
        self.uri = uri

    async def prepare_audio(self, audio_bytes: bytes) -> bytes:
        # If it's WAV, strip header and return PCM
        if audio_bytes[:4] == b"RIFF":
            return audio_bytes[44:]  # skip WAV header

        # If it's already PCM, return as-is
        return audio_bytes


    async def original_transcribe_audio(self, audio_bytes: bytes):
        """
        Accept ANY audio format → normalize → send to Vosk → return transcript.
        """
        # Normalize first
        pcm = await self.prepare_audio(audio_bytes)

        async with websockets.connect(self.uri) as ws:
            logger.info(f"[VoskModule] Connected to {self.uri}")
            logger.info(f"[VoskModule] Normalized audio: {len(audio_bytes)} → {len(pcm)} bytes")

            # Send audio in chunks
            chunk_size = 4000
            for i in range(0, len(pcm), chunk_size):
                chunk = pcm[i:i+chunk_size]
                await ws.send(chunk)
                logger.debug(f"[VoskModule] Sent chunk {i//chunk_size} ({len(chunk)} bytes)")

            # Signal end of file
            await ws.send('{"eof" : 1}')
            logger.info("[VoskModule] Sent EOF")

            final = None

            try:
                while True:
                    msg = await ws.recv()
                    logger.debug(f"[VoskModule] Received raw: {msg[:200]}")
                    data = json.loads(msg)

                    if "result" in data:
                        return data

            except websockets.exceptions.ConnectionClosedOK:
                logger.info("[VoskModule] Connection closed cleanly")

            return final

    async def transcribe_audio(self, audio_bytes: bytes, req_id: str | None = None): # enable for massive debug log
        if req_id is None:
            req_id = "noid"

        pcm = await self.prepare_audio(audio_bytes)
        logger.info(f"[VoskModule] [{req_id}] Input bytes={len(audio_bytes)}, PCM bytes={len(pcm)}")

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

                logger.debug(f"[VoskModule] [{req_id}] Received raw: {msg[:200]}")
                data = json.loads(msg)

                if "result" in data:
                    logger.info(f"[VoskModule] [{req_id}] FINAL: {str(data)[:500]}")
                    return data
