import asyncio
import websockets
import json
import os
import logging
from llama_cpp import Llama
import re
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Any, Dict, Optional, cast
import xml.etree.ElementTree as ET
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


# Load default model at startup
load_model(DEFAULT_MODEL)

LLM_PORT = int(os.getenv("LLM_SERVER_PORT", 8765))

# -----------------------------
# TOOL-CALL PARSING
# -----------------------------
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)


def extract_tool_call(content: str):
    try:
        obj = json.loads(content)

        # Correct format
        if "name" in obj and "arguments" in obj:
            args = obj.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            return obj["name"], args

        # Model's alternate format
        if "tool_name" in obj and "tool_args" in obj:
            args = obj.get("tool_args") or {}
            if not isinstance(args, dict):
                args = {}
            return obj["tool_name"], args

    except Exception as e:
        logger.info(f"[TOOL PARSE ERROR] {e}")

    return None, None



def get_hardware_state(components):
    logger.info(f"[TOOL get_hardware_state] components={components!r}")
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/api/hardware",
            json={"components": components},
            timeout=2
        )
        logger.info(f"[TOOL get_hardware_state] status={resp.status_code}, body={resp.text[:50]!r}")
        resp.raise_for_status()
        return {"data": resp.json()}
    except Exception as e:
        logger.info(f"[TOOL get_hardware_state ERROR] {e}")
        return {"error": str(e)}


def fetch_api(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
):
    logger.info(f"[FETCH_API] url={url}")
    logger.info(f"[FETCH_API] params={params}")
    logger.info(f"[FETCH_API] headers={headers}")

    try:
        # Ensure browser-like UA
        if headers is None:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                )
            }
            logger.info("[FETCH_API] Injected default User-Agent")

        logger.info("[FETCH_API] Performing GET request...")
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        logger.info(f"[FETCH_API] Response status={r.status_code}")
        logger.info(f"[FETCH_API] Response headers={dict(r.headers)}")

        text = r.text
        logger.info(f"[FETCH_API] Raw text length={len(text)}")

        # --- NEWS PREPROCESSING ---
        if params and params.get("api_type") == "news":
            logger.info("[FETCH_API] Detected api_type=news")
            logger.info("[FETCH_API] Starting RSS preprocessing...")

            cleaned_json = extract_clean_rss_items_json(text, limit=5)
            logger.info(f"[FETCH_API] Cleaned JSON length={len(cleaned_json)}")
            logger.info(f"[FETCH_API] Cleaned JSON preview={cleaned_json[:500]}")

            parsed = json.loads(cleaned_json)
            logger.info(f"[FETCH_API] Parsed JSON keys={list(parsed[0].keys()) if parsed else 'EMPTY'}")

            return {
                "status": r.status_code,
                "reason": "ok" if r.ok else "http_error",
                "items": parsed
            }

        # --- DEFAULT RETURN ---
        logger.info("[FETCH_API] Non-news request, returning raw text/json")
        return {
            "status": r.status_code,
            "reason": "ok" if r.ok else "http_error",
            "json": r.json() if "application/json" in r.headers.get("Content-Type", "") else None,
            "text": text
        }

    except Exception as e:
        logger.error(f"[FETCH_API] ERROR: {e}", exc_info=True)
        return {
            "url": url,
            "status": "error",
            "reason": str(e),
            "json": None,
            "text": ""
        }



def get_world_state(keys):
    return {"data": {k: "<stub>" for k in keys}}


def get_perception(sensors):
    return {"data": {s: "<stub>" for s in sensors}}


def memory_search(query):
    return {"results": []}


def memory_write(data):
    return {"status": "ok"}


def load_api_instructions(module: str):
    """
    Loads instruction text for a specific API module.
    Example modules:
      - "weather"
      - "news"
      - "wikipedia"
      - "general"
    """
    base = "data/api_instructions"
    path = os.path.join(base, f"{module}.txt")

    try:
        if not os.path.exists(path):
            return {"status": "error", "error": "no_instructions"}

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "status": "ok",
            "module": module,
            "instructions": text
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


TOOLS = {
    "fetch_api": fetch_api,
    "get_world_state": get_world_state,
    "get_hardware_state": get_hardware_state,
    "get_perception": get_perception,
    "memory_search": memory_search,
    "load_api_instructions": load_api_instructions,
}

TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "fetch_api": {
        "kind": "api",
        "allowed_domains": [
            "wikipedia.org",
            "openweathermap.org",
            "cbc.ca"
        ],
    },

    "get_hardware_state": {
        "kind": "hardware",
        "allowed_components": ["battery", "cpu", "memory", "load"],
    },

    "load_api_instructions": {
        "kind": "meta",
        "allowed_modules": [
            "hardware_state",
            "conversation",
            "weather",
            "news",
            "wikipedia",
            "general",
            "memory",
            "perception"
        ],
    },
}


def apply_tool_rules(tool_name, tool_args, messages, last_tool_call):
    spec = TOOL_SPECS.get(tool_name, {})

    # Argument validation
    if "allowed_components" in spec:
        allowed = set(spec["allowed_components"])
        requested = set(tool_args.get("components", []))
        if not requested.issubset(allowed):
            return "block"

    if "allowed_domains" in spec:
        url = tool_args.get("url", "")
        host = urlparse(url).hostname or ""
        if not any(host.endswith(dom) for dom in spec["allowed_domains"]):
            return "block"

    # Never block instruction loading for redundancy
    if tool_name != "load_api_instructions":
        if last_tool_call:
            last_name, last_args = last_tool_call
            if last_name == tool_name and last_args == tool_args:
                return "block"

    # Hardware sufficiency
    if tool_name == "get_hardware_state":
        requested = set(tool_args.get("components", []))
        fetched = set()
        for m in messages:
            if m.get("role") == "tool" and m.get("name") == "get_hardware_state":
                try:
                    data = json.loads(m["content"])
                    fetched.update(data.keys())
                except:
                    pass
        if requested.issubset(fetched):
            return "stop"

    # Web sufficiency
    if tool_name == "fetch_api":
        url = tool_args.get("url")
        for m in messages:
            if m.get("role") == "tool" and m.get("name") == "fetch_api":
                try:
                    data = json.loads(m["content"])
                    if data.get("url") == url:
                        return "stop"
                except:
                    pass

    # For load_api_instructions: never "stop" on its own, just keep going
    if tool_name == "load_api_instructions":
        return "ok"

    return "ok"


# Map model-facing tool names to internal runtime tool names
TOOL_ALIASES = {
    "api": "fetch_api",
    "json": "fetch_api",
    "weather": "fetch_api",
    "news": "fetch_api",
    "wiki": "fetch_api",
    # add more aliases as desired or needed
}


def check_lists(url: str):
    host = urlparse(url).hostname or ""

    # blacklist
    if os.path.exists("blacklist.txt"):
        with open("blacklist.txt") as f:
            if host in {line.strip() for line in f}:
                return "blocked_blacklist"

    # whitelist
    if os.path.exists("whitelist.txt"):
        with open("whitelist.txt") as f:
            if host not in {line.strip() for line in f}:
                return "blocked_not_whitelisted"

    return "ok"


def map_tool_name(name: str) -> str:
    if not name:
        return name
    return TOOL_ALIASES.get(name, name)


def run_with_tools(messages, max_tool_loops=5):
    loop_count = 0

    while loop_count < max_tool_loops:
        loop_count += 1

        # NEW: sufficiency check BEFORE calling the model
        if should_finalize_based_on_last_tool(messages):
            inject_finalization_message(messages)

            # FINAL MODEL CALL — this produces the cognition JSON
            final_content, _, _ = call_llm_for_tool_reasoning(messages)

            return final_content

        content, tool_name, tool_args = call_llm_for_tool_reasoning(messages)
        last_tool_call = get_last_tool_call(messages)

        tool_name, tool_args = normalize_tool_args(tool_name, tool_args, last_tool_call)

        if should_block_tool(tool_name, last_tool_call, tool_args, messages):
            continue

        execute_tool(tool_name, tool_args, messages)

    return content

def should_finalize_based_on_last_tool(messages):
    last = get_last_tool_call(messages)
    if not last:
        return False

    tool_name, payload = last
    tool_args = payload.get("tool_args", {})

    # NEVER finalize after meta-tools
    if tool_name == "load_api_instructions":
        return False

    # Hardware finalization
    if tool_name == "get_hardware_state":
        return True

    # News finalization
    if tool_name == "fetch_api":
        params = tool_args.get("params", {})
        if params.get("api_type") == "news":
            return True

    return False


def inject_finalization_message(messages):
    """
    Inject a system message telling HAL to stop calling tools
    and produce the final cognition JSON.
    """

    # Find last tool call again
    last = None
    for m in reversed(messages):
        if m.get("role") == "tool":
            try:
                last = (m["name"], json.loads(m["content"]))
            except Exception:
                last = None
            break

    if not last:
        return

    tool_name, payload = last

    # Hardware finalization
    if tool_name == "get_hardware_state":
        messages.append({
            "role": "system",
            "content": (
                "You already retrieved the hardware status. "
                "Do NOT call any more tools. "
                "Use the hardware information to produce the final cognition JSON."
            )
        })
        return

    # News finalization
    if tool_name == "fetch_api":
        params = payload.get("tool_args", {}).get("params", {})
        if params.get("api_type") == "news":
            messages.append({
                "role": "system",
                "content": (
                    "You already retrieved the news. "
                    "Do NOT call any more tools. "
                    "Use the news information to produce the final cognition JSON."
                )
            })
            return

    # Future modules can be added here

def call_llm_for_tool_reasoning(messages):
    logger.info("\n[LLM INPUT MESSAGES]\n%s\n", json.dumps(messages, indent=2))
    resp = cast(Dict[str, Any], CURRENT_LLM.create_chat_completion(
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    ))
    logger.info("\n[LLM RAW OUTPUT]\n%s\n", resp)

    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    tool_name, tool_args = extract_tool_call(content)
    tool_args = tool_args or {}
    logger.info(f"[PARSED TOOL CALL] name={tool_name}, args={tool_args}")

    return content, tool_name, tool_args

def get_last_tool_call(messages):
    for m in reversed(messages):
        if m.get("role") == "tool":
            try:
                return (m["name"], json.loads(m["content"]))
            except Exception:
                return None
    return None


def handle_meta_tool(tool_name, tool_args, messages):
    if not isinstance(tool_name, str):
        return False

    spec = TOOL_SPECS.get(tool_name, {})
    if spec.get("kind") != "meta":
        return False

    tool_fn = TOOLS.get(tool_name)
    result = tool_fn(**tool_args) if tool_fn else None

    messages.append({
        "role": "tool",
        "name": tool_name,
        "content": json.dumps({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result
        }),
    })

    messages.append({
        "role": "system",
        "content": "Instructions loaded. Now use the tool."
    })

    return True


def normalize_tool_args(tool_name, tool_args, last_tool_call):
    # --- HARDWARE NORMALIZATION ---
    if tool_name == "get_hardware_state":
        if "component" in tool_args and "components" not in tool_args:
            comp = tool_args.get("component")
            if isinstance(comp, list):
                tool_args["components"] = comp
            elif comp is None:
                tool_args["components"] = []
            else:
                tool_args["components"] = [comp]
            tool_args.pop("component", None)

        comps = tool_args.get("components")
        if isinstance(comps, str):
            tool_args["components"] = [comps]
        if tool_args.get("components") is None:
            tool_args["components"] = []

    # --- NEWS AUTO-TAGGING + FETCH NORMALIZATION ---
    if tool_name == "fetch_api":
        tool_args.pop("method", None)

        params = tool_args.get("params")
        if not isinstance(params, dict):
            params = {}

        if last_tool_call and last_tool_call[0] == "load_api_instructions":
            if last_tool_call[1].get("tool_args", {}).get("module") == "news":
                params["api_type"] = "news"

        tool_args["params"] = params

    return tool_name, tool_args


def should_block_tool(tool_name, last_tool_call, tool_args, messages):
    # --- HARD STOP AFTER HARDWARE ---
    if last_tool_call and last_tool_call[0] == "get_hardware_state":
        messages.append({
            "role": "system",
            "content": (
                "You already have the required hardware status from get_hardware_state. "
                "Do NOT call any more tools. "
                "Use the hardware information that has been retrieved to produce the final cognition JSON."
            )
        })
        return True

    # --- ILLEGAL TOOL CHECK ---
    if tool_name not in TOOLS:
        if last_tool_call is None:
            messages.append({
                "role": "system",
                "content": (
                    f"'{tool_name}' is not a valid tool. "
                    "Choose a valid tool from the available list."
                )
            })
            return True

        messages.append({
            "role": "system",
            "content": (
                f"'{tool_name}' is not a valid tool. "
                "You already have the data you need. "
                "Do NOT call any more tools. "
                "Summarize the information and produce the final cognition JSON."
            )
        })
        return True

    # --- NEWS DOUBLE-FETCH CHECK ---
    if tool_name == "fetch_api" and tool_args.get("api_type") == "news":
        if last_tool_call and last_tool_call[0] == "fetch_api":
            prev_args = last_tool_call[1].get("tool_args", {})
            if prev_args.get("api_type") == "news":
                messages.append({
                    "role": "system",
                    "content": (
                        "You already have the required News data from the fetch_api call. "
                        "Do NOT call any more tools. "
                        "Use the news information that has been retrieved to produce the final cognition JSON."
                    )
                })
                return True

    # --- apply_tool_rules ---
    decision = apply_tool_rules(
        tool_name=tool_name,
        tool_args=tool_args,
        messages=messages,
        last_tool_call=last_tool_call,
    )

    if decision == "block":
        messages.append({
            "role": "system",
            "content": f"Tool call blocked by rules: {tool_name} {tool_args}. Adjust your request."
        })
        return True

    return False

def execute_tool(tool_name, tool_args, messages):
    tool_fn = TOOLS.get(tool_name)
    if tool_fn is None:
        messages.append({
            "role": "system",
            "content": f"Unknown tool '{tool_name}'. Do not call this tool again."
        })
        return

    try:
        result = tool_fn(**tool_args)
    except Exception as e:
        result = {"status": "error", "error": str(e)}

    messages.append({
        "role": "tool",
        "name": tool_name,
        "content": json.dumps({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result
        }),
    })

    # IMPORTANT: meta-tools do NOT fetch real data
    if tool_name == "load_api_instructions":
        messages.append({
            "role": "system",
            "content": (
                "You have loaded the API instructions for this tool. "
                "Use these instructions to fetch the actual data needed to answer the user. "
            )
        })
    else:
        messages.append({
            "role": "system",
            "content": "Continue reasoning with the tool result already fetched."
        })


def strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split()).strip()

def extract_clean_rss_items_json(xml_text, limit=5):
    """
    Parse RSS XML and return a JSON string containing:
    - title
    - pubDate
    - summary (HTML stripped)
    """
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return "[]"

    items = []
    for item in channel.findall("item")[:limit]:
        title = item.findtext("title", "").strip()
        pub = item.findtext("pubDate", "").strip()
        desc_raw = item.findtext("description", "")
        summary = strip_html(desc_raw)

        items.append({
            "title": title,
            "pubDate": pub,
            "summary": summary
        })

    return json.dumps(items, ensure_ascii=False, indent=2)
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