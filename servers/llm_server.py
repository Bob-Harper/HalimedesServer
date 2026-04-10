import asyncio
import websockets
import json
import os
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

LLM_BIND_HOST = os.getenv("LLM_BIND_HOST", "0.0.0.0")
LLM_BIND_PORT = int(os.getenv("LLM_BIND_PORT", 8765))

LLM_SMALL_PATH  = os.getenv("LLM_SMALL_PATH")
LLM_MEDIUM_PATH = os.getenv("LLM_MEDIUM_PATH")
LLM_LARGE_PATH  = os.getenv("LLM_LARGE_PATH")

# Require at least one model path
if not any([LLM_SMALL_PATH, LLM_MEDIUM_PATH, LLM_LARGE_PATH]):
    raise RuntimeError("No model path is set in the environment. Please rectify.")

MODELS = {
    "small":  LLM_SMALL_PATH,
    "medium": LLM_MEDIUM_PATH,
    "large":  LLM_LARGE_PATH
}

DEFAULT_MODEL = "medium"
CURRENT_MODEL_NAME = DEFAULT_MODEL # scaffolding for hotswapping models
CURRENT_LLM = None


# ------------------------------------------------------------
# Load a model by name (hot-swap)
# ------------------------------------------------------------
def load_model(model_name: str):
    global CURRENT_MODEL_NAME, CURRENT_LLM

    path = MODELS.get(model_name)
    if not path:
        raise RuntimeError(f"Model '{model_name}' is not configured.")

    print(f"[LLM] Loading model '{model_name}' from: {path}")

    # Unload previous model
    CURRENT_LLM = None

    CURRENT_LLM = Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4
    )

    CURRENT_MODEL_NAME = model_name
    print(f"[LLM] Model '{model_name}' loaded and ready.")


# Load default model at startup
load_model(DEFAULT_MODEL)

LLM_PORT = int(os.getenv("LLM_SERVER_PORT", 8765))


# ------------------------------------------------------------
# Handle a single WebSocket connection
# ------------------------------------------------------------
async def handle_client(websocket):
    print(f"[LLM] Client connected: {websocket.remote_address}")

    async for message in websocket:
        try:
            data = json.loads(message)

            prompt = data.get("prompt")
            model_request = data.get("model")  # optional
            messages = data.get("messages")    # optional chat format

            if not prompt and not messages:
                await websocket.send(json.dumps({"error": "Missing prompt or messages"}))
                continue

            # Hot-swap model if requested
            if model_request and model_request != CURRENT_MODEL_NAME:
                load_model(model_request)

            # Chat mode
            if messages:
                output = CURRENT_LLM.create_chat_completion(
                    messages=messages,
                    max_tokens=256,
                    temperature=0.55,
                    top_k=90,
                    top_p=0.9,
                    repeat_penalty=1.25
                )
                response_text = output["choices"][0]["message"]["content"]

            # Legacy prompt mode
            else:
                output = CURRENT_LLM(
                    prompt,
                    max_tokens=256,
                    stop=["</s>"],
                    echo=False
                )
                response_text = output["choices"][0]["text"]

            await websocket.send(json.dumps({
                "response": response_text.strip(),
                "model": CURRENT_MODEL_NAME
            }))

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

    print(f"[LLM] Client disconnected: {websocket.remote_address}")


# ------------------------------------------------------------
# Start WebSocket server
# ------------------------------------------------------------
async def main():
    print(f"[LLM] Starting WebSocket server on {LLM_BIND_HOST}:{LLM_BIND_PORT}...")
    async with websockets.serve(handle_client, LLM_BIND_HOST, LLM_BIND_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())