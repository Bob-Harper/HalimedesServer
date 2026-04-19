import asyncio
import websockets
import json
import tempfile
import subprocess
import os
import logging
logger = logging.getLogger("VoskModule")


class VoskModule:
    def __init__(self, uri="ws://localhost:2700"):
        self.uri = uri

    async def prepare_audio(self, audio_bytes: bytes) -> bytes:
        """
        Convert ANY input audio into PCM16 mono 44.1k raw bytes.
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
            out_path = tmp_out.name

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", in_path,
            "-ac", "1",
            "-ar", "44100",
            "-f", "s16le",
            out_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(out_path, "rb") as f:
            pcm = f.read()

        os.remove(in_path)
        os.remove(out_path)

        return pcm

    async def transcribe(self, audio_bytes: bytes):
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

                    if "text" in data or "result" in data:
                        logger.info(f"[VoskModule] Final result: {data}")
                        final = data

            except websockets.exceptions.ConnectionClosedOK:
                logger.info("[VoskModule] Connection closed cleanly")

            return final