import json
import websockets
import asyncio

class LLMModule:
    def __init__(self, url: str):
        self.url = url

    async def infer(self, model: str, messages: list):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0
        }

        async with websockets.connect(
            self.url,
            ping_interval=300,
            ping_timeout=300,
            close_timeout=300
        ) as ws:
            await ws.send(json.dumps(payload))
            response = await asyncio.wait_for(ws.recv(), timeout=600)
            return json.loads(response)
