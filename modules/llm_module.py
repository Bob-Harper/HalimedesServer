# gateway/llm_module.py

import json
import websockets
import asyncio

class LLMModule:
    def __init__(self, url: str):
        self.url = url

    async def infer(self, model: str, system_prompt: str, user_prompt: str):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }

        async with websockets.connect(self.url) as ws:
            await ws.send(json.dumps(payload))
            response = await ws.recv()
            return json.loads(response)
