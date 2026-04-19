import json
import websockets
import asyncio
import os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "inference_schema.json")


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
            msg = json.dumps(payload)
            print(f"[LLMModule] Sending to LLM: {msg[:200]}")
            await ws.send(msg)

            response = await ws.recv()
            print(f"[LLMModule] Received from LLM: {response[:200]}")

            return json.loads(response)

    async def infer(self, model: str, system_prompt: str, user_prompt: str):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {
                "type": "json_object",
                "schema": schema
            },
            "temperature": 0.0
        }

        async with websockets.connect(self.url) as ws:
            msg = json.dumps(payload)
            await ws.send(msg)
            response = await ws.recv()
            return json.loads(response)
