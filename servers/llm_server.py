import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)\

import asyncio
import websockets
import json
import logging
from llama_cpp import Llama
from dotenv import load_dotenv
load_dotenv()
from modules.llm_tools import ToolClass


# -----------------------------
# INIT
# -----------------------------

logger = logging.getLogger("LLMServer")
logging.basicConfig(
    level=logging.DEBUG,
    filename="logging/llm_server.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s"
)

LLM_BIND_HOST = os.getenv("LLM_BIND_HOST", "0.0.0.0")
LLM_BIND_PORT = int(os.getenv("LLM_BIND_PORT", 8765))

LLM_SMALL_PATH  = os.getenv("LLM_SMALL_PATH")
LLM_MEDIUM_PATH = os.getenv("LLM_MEDIUM_PATH")
LLM_LARGE_PATH  = os.getenv("LLM_LARGE_PATH")

GATEWAY_HOST = os.getenv("GATEWAY_CONNECT_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.getenv("GATEWAY_CONNECT_PORT", 9000))
GATEWAY_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

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

with open("data/system_prompt_autonomous.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT_AUTONOMOUS = f.read()

with open("data/system_prompt_chat.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT_CHAT = f.read()

with open("data/system_prompt_tool.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT_TOOL = f.read()

USER_AGENT = (
    "Halimedes/1.0 "
    "(+(+https://example.com/info); "
    "busy minion; contact=admin@yourdomain.example)"
)

last_request_time = {}


# ------------------------------------------------------------
# Load a model by name (hot-swappable)
# ------------------------------------------------------------
def load_model(model_name: str):
    global CURRENT_MODEL_NAME, CURRENT_LLM

    path = MODELS.get(model_name)
    if not path:
        raise RuntimeError(f"Model '{model_name}' is not configured.")

    logger.info(f"[LLM] Loading model '{model_name}' from: {path}")


    CURRENT_LLM = Llama(
        model_path=path,
        n_gpu_layers=-1,
        n_ctx=9096,
        n_threads=os.cpu_count() or 4,
        verbose=False
    )

    CURRENT_MODEL_NAME = model_name
    logger.info(f"[LLM] Model '{model_name}' loaded and ready.")

tools = ToolClass(GATEWAY_URL)

load_model(DEFAULT_MODEL)
tools.model = CURRENT_LLM

# Load default model at startup
load_model(DEFAULT_MODEL)

LLM_PORT = int(os.getenv("LLM_SERVER_PORT", 8765))

# -----------------------------
# WEBSOCKET SERVER
# -----------------------------
async def handle_inference(ws):
    logger.info(">>> Handle Inference <<<")

    async for raw in ws:
        logger.info(f"\n[WS RECV RAW]\n{raw}\n")
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.info(f"[WS JSON ERROR] {e}")
            await ws.send(json.dumps({"error": "invalid JSON"}))
            continue

        messages = data.get("messages", [])
        inference_type = data.get("inference_type", "chat")

        # Select the appropriate system prompt based on inference type
        if inference_type == "autonomous":
            SYSTEM_PROMPT = SYSTEM_PROMPT_AUTONOMOUS
        elif inference_type == "chat":
            SYSTEM_PROMPT = SYSTEM_PROMPT_CHAT
        elif inference_type == "tool":
            SYSTEM_PROMPT = SYSTEM_PROMPT_TOOL
        else:
            SYSTEM_PROMPT = SYSTEM_PROMPT_CHAT  # Default

        # NOW prepend system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Run with tool loop
        final_content = tools.run_with_tools(messages)
        logger.info(f"[WS SEND]\n\n{final_content}\n\n")
        resp_payload = {
            "response": final_content,
        }

        await ws.send(json.dumps(resp_payload))


# ------------------------------------------------------------
# Start WebSocket server
# ------------------------------------------------------------
async def main():
    logger.info(f"[LLM] Starting WebSocket servers...")

    # Start inference server
    await websockets.serve(handle_inference, LLM_BIND_HOST, LLM_BIND_PORT)
    print("----- LLM Inference Server Open -----")

    # Keep running forever
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())