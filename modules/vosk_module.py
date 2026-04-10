import asyncio
import websockets
import json

class VoskModule:
    def __init__(self, uri="ws://localhost:2700"):
        self.uri = uri

    async def transcribe(self, audio_bytes):
        async with websockets.connect(self.uri) as ws:

            # Send audio in chunks
            chunk_size = 4000
            for i in range(0, len(audio_bytes), chunk_size):
                await ws.send(audio_bytes[i:i+chunk_size])

            # Signal end of file
            await ws.send('{"eof" : 1}')

            final = None

            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if "text" in data or "result" in data:
                        final = data

            except websockets.exceptions.ConnectionClosedOK:
                pass

            return final