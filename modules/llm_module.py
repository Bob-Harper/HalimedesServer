# modules/llm_module.py
import json
import websockets
import asyncio

class LLMModule:
    def __init__(self, url: str):
        self.url = url

    async def chat(self, model: str, messages: list, **kwargs):
        """
        Send a chat request to the LLM server.
        """
        payload = {
            "model": model,
            "messages": messages
        }
        payload.update(kwargs)

        async with websockets.connect(self.url) as ws:
            await ws.send(json.dumps(payload))
            response = await ws.recv()
            return json.loads(response)