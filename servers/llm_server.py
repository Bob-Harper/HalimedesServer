import asyncio
import websockets
import json
import os
import logging
from llama_cpp import Llama
import re
import requests
from dotenv import load_dotenv
from typing import Dict, Any, cast
load_dotenv()

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

with open("data/system_prompt_inference.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# ------------------------------------------------------------
# Load a model by name (hot-swap)
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
# -----------------------------
# TOOL IMPLEMENTATION
# -----------------------------
def fetch_url(url: str):
    logger.info(f"\n[FETCH] GET {url}")
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        logger.info(f"[FETCH STATUS] {r.status_code}")
        logger.info(f"[FETCH HEADERS] {dict(r.headers)}")
        logger.info(f"[FETCH BODY] {text[:500]}...\n")  # truncate for sanity
        return {
            "status": r.status_code,
            "headers": dict(r.headers),
            "body": text,
        }
    except Exception as e:
        logger.info(f"[FETCH ERROR] {e}\n")
        return {"error": str(e)}

# -----------------------------
# TOOL-CALL PARSING
# -----------------------------


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)


def extract_tool_call(content: str):
    """
    Return (tool_name, arguments_dict) or (None, None) if no tool call.
    Supports either:
      - <tool_call>{...}</tool_call>
      - bare JSON: {"name": "...", "arguments": {...}}
    """
    # 1) Try bare JSON first
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            args = obj.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            return obj["name"], args
    except Exception:
        pass

    # 2) Fallback to <tool_call>...</tool_call> wrapper
    start = content.find("<tool_call>")
    end = content.find("</tool_call>")

    if start == -1 or end == -1:
        return None, None

    block = content[start + len("<tool_call>") : end].strip()

    try:
        tc = json.loads(block)
        name = tc.get("name")
        args = tc.get("arguments", {}) or {}
        if not isinstance(args, dict):
            args = {}
        return name, args
    except Exception as e:
        logger.info(f"[TOOL PARSE ERROR] {e} | block={block!r}")
        return None, None


def get_hardware_state(components):
    logger.info(f"[TOOL get_hardware_state] components={components!r}")
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/api/hardware",
            json={"components": components},
            timeout=2
        )
        logger.info(f"[TOOL get_hardware_state] status={resp.status_code}, body={resp.text[:500]!r}")
        resp.raise_for_status()
        return {"data": resp.json()}
    except Exception as e:
        logger.info(f"[TOOL get_hardware_state ERROR] {e}")
        return {"error": str(e)}


def get_world_state(keys):
    return {"data": {k: "<stub>" for k in keys}}


def get_perception(sensors):
    return {"data": {s: "<stub>" for s in sensors}}


def memory_search(query):
    return {"results": []}


def memory_write(data):
    return {"status": "ok"}


TOOLS = {
    "fetch_url": fetch_url,
    "get_world_state": get_world_state,
    "get_hardware_state": get_hardware_state,
    "get_perception": get_perception,
    "memory_search": memory_search,
    "memory_write": memory_write,
}

def run_with_tools(messages, max_tool_loops: int = 8):
    """
    Run Qwen with a tool loop:
    - Call model
    - If it emits <tool_call>, execute tool, append tool result, repeat
    - Stop when no tool_call or max_tool_loops reached
    """
    loop_count = 0

    while True:
        logger.info(f"\n[MODEL CALL] loop={loop_count}")
        logger.info(f"[MESSAGES IN] {json.dumps(messages, indent=2)[:1000]}...\n")

        resp = cast(Dict[str, Any], CURRENT_LLM.create_chat_completion(
            messages=messages,
            temperature=0.0,
            response_format={ "type": "json_object" },
        ))

        msg = resp["choices"][0]["message"]
        content = msg.get("content", "") or ""

        logger.info(f"[MODEL RAW OUTPUT]\n{content}...\n")

        # Try to extract a tool call
        tool_name, tool_args = extract_tool_call(content)

        if not tool_name:
            logger.info("[TOOL LOOP] No tool_call detected. Returning final content.\n")
            return content  # final answer

        logger.info(f"[TOOL LOOP] Detected tool_call: name={tool_name}, args={tool_args}")

        # Execute tool
        tool_fn = TOOLS.get(tool_name)
        if not tool_fn:
            tool_result = {"error": f"Unknown tool '{tool_name}'"}
            logger.info(f"[TOOL ERROR] {tool_result['error']}")
        else:
            try:
                tool_args = cast(Dict[str, Any], tool_args)
                tool_result = tool_fn(**tool_args)
            except TypeError as e:
                tool_result = {"error": f"Bad arguments for tool '{tool_name}': {e}"}
                logger.info(f"[TOOL ARG ERROR] {e}")

        # Append assistant message (with tool_call) and tool response
        messages.append(msg)
        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(tool_result),
        })

        loop_count += 1
        if loop_count >= max_tool_loops:
            logger.info("[TOOL LOOP] Reached max_tool_loops. Stopping.\n")
            return content  # last content, even if it had a tool_call

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
        temperature = data.get("temperature", 0.0)

        # NOW prepend system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Run with tool loop
        final_content = run_with_tools(messages)
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

    inference_server = await websockets.serve(handle_inference, LLM_BIND_HOST, LLM_BIND_PORT)

    # Keep running forever
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())