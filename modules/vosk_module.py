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
        # Detect WAV header
        is_wav = audio_bytes[:4] == b"RIFF"

        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in.flush()
            in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
            out_path = tmp_out.name

        # If it's raw PCM from HAL, tell ffmpeg the format explicitly
        if not is_wav:
            input_fmt = ["-f", "s16le", "-ac", "1", "-ar", "44100"]
        else:
            input_fmt = []

        subprocess.run([
            "ffmpeg",
            "-y",
            *input_fmt,
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


    async def transcribe_audio(self, audio_bytes: bytes):
        """
        Accept ANY audio format → normalize → send to Vosk → return transcript.
        """
        # Normalize first
        pcm = await self.prepare_audio(audio_bytes)

        async with websockets.connect(self.uri) as ws:
            logger.info(f"[VoskModule] Connected to {self.uri}")
            logger.info(f"[VoskModule] Normalized audio: {len(audio_bytes)} → {len(pcm)} bytes")

            # Send audio in chunks
            chunk_size = 8000
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
                    if "text" in data and data["text"].strip():
                        return data

            except websockets.exceptions.ConnectionClosedOK:
                logger.info("[VoskModule] Connection closed cleanly")

            return final