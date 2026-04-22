import asyncio
import websockets
import json
import os
import logging
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("LLMServer")
logging.basicConfig(level=logging.INFO)

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
CURRENT_MODEL_NAME = DEFAULT_MODEL
CURRENT_LLM = None


# ------------------------------------------------------------
# Load a model by name (hot-swap)
# ------------------------------------------------------------
def load_model(model_name: str):
    global CURRENT_MODEL_NAME, CURRENT_LLM

    path = MODELS.get(model_name)
    if not path:
        raise RuntimeError(f"Model '{model_name}' is not configured.")

    logger.info(f"[LLM] Loading model '{model_name}' from: {path}")

    CURRENT_LLM = None
    CURRENT_LLM = Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False
    )

    CURRENT_MODEL_NAME = model_name
    logger.info(f"[LLM] Model '{model_name}' loaded and ready.")


# Load default model at startup
load_model(DEFAULT_MODEL)

LLM_PORT = int(os.getenv("LLM_SERVER_PORT", 8765))


# ------------------------------------------------------------
# Handle a single WebSocket connection
# ------------------------------------------------------------
async def handle_chat(websocket):
    logger.info(f"[LLM] Client connected: {websocket.remote_address}")

    async for message in websocket:
        logger.debug(f"[LLM] Received raw: {message[:200]}")
        # Log the raw incoming message exactly as received
        with open("logging/model_input.txt", "a", encoding="utf-8") as f:
            f.write("=== RAW INCOMING MESSAGE ===\n")
            f.write(message)
            f.write("\n\n")
        try:
            data = json.loads(message)

            prompt = data.get("prompt")
            model_request = data.get("model")
            messages = data.get("messages")

            # Hot-swap model if requested
            if model_request and model_request != CURRENT_MODEL_NAME:
                logger.info(f"[LLM] Hot-swap request: {model_request}")
                load_model(model_request)

            # Chat mode
            logger.info(f"[LLM] Chat request with {len(messages)} messages")
            output = CURRENT_LLM.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.55,
                top_k=90,
                top_p=0.9,
                repeat_penalty=1.25
            )
            response_text = output["choices"][0]["message"]["content"]

            reply = json.dumps({
                "response": response_text.strip(),
                "model": CURRENT_MODEL_NAME
            })

            logger.debug(f"[LLM] Sending: {reply[:200]}")
            await websocket.send(reply)

        except Exception as e:
            err = json.dumps({"error": str(e)})
            logger.error(f"[LLM] Error: {e}")
            await websocket.send(err)

    logger.info(f"[LLM] Client disconnected: {websocket.remote_address}")


async def handle_inference(websocket):
    logger.info(f"[LLM] Inference client connected: {websocket.remote_address}")

    async for message in websocket:
        logger.debug(f"[LLM] Received raw (inference): {message[:200]}")

        try:
            data = json.loads(message)

            # NEW: Expect messages instead of prompt
            messages = data.get("messages")
            model_request = data.get("model")
            response_format = data.get("response_format")
            temperature = data.get("temperature", 0.0)

            if not messages:
                raise RuntimeError("Inference request missing 'messages' field.")

            # Hot-swap model if requested
            if model_request and model_request != CURRENT_MODEL_NAME:
                logger.info(f"[LLM] Hot-swap request: {model_request}")
                load_model(model_request)

            logger.info("[LLM] Inference request (JSON chat mode)")

            # JSON-MODE CHAT COMPLETION
            resp = CURRENT_LLM.create_chat_completion(
                messages=messages,
                response_format=response_format,
                temperature=temperature,
            )

            # Log raw model response
            with open("logging/model_output.txt", "a", encoding="utf-8") as f:
                f.write("=== RAW INFERENCE RESPONSE ===\n")
                f.write(str(resp))
                f.write("\n\n")

            # Extract JSON string
            content = resp["choices"][0]["message"]["content"]

            # Validate JSON
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {"error": "invalid_json", "raw": content}

            # Send JSON back to gateway
            await websocket.send(json.dumps({"response": content}))

        except Exception as e:
            err = json.dumps({"error": str(e)})
            logger.error(f"[LLM] Inference error: {e}")
            await websocket.send(err)

    logger.info(f"[LLM] Inference client disconnected: {websocket.remote_address}")



# ------------------------------------------------------------
# Start WebSocket server
# ------------------------------------------------------------
async def main():
    logger.info(f"[LLM] Starting WebSocket servers...")

    chat_server = await websockets.serve(handle_chat, LLM_BIND_HOST, LLM_BIND_PORT)
    inference_server = await websockets.serve(handle_inference, LLM_BIND_HOST, LLM_BIND_PORT + 1)

    # Keep running forever
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())