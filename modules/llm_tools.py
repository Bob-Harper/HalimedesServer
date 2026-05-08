import json
import os
import logging
import re
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Any, Dict, Optional, Tuple, cast
import xml.etree.ElementTree as ET

load_dotenv()

logger = logging.getLogger("LLMServer")
logging.basicConfig(
    level=logging.INFO,
    filename="logging/llm_server.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s"
)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)


class ToolClass:
    def __init__(self, url: Optional[str] = None, model=None):
        self.url = url
        self.model = model

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_tool_call(content: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            obj = json.loads(content)

            # Correct format
            if "name" in obj and "arguments" in obj:
                args = obj.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
                return obj["name"], args

            # Alternate format
            if "tool_name" in obj and "tool_args" in obj:
                args = obj.get("tool_args") or {}
                if not isinstance(args, dict):
                    args = {}
                return obj["tool_name"], args

        except Exception as e:
            logger.info(f"[TOOL_PARSE ERROR] {e}")

        return None, None

    @staticmethod
    def strip_html(text: Optional[str]) -> str:
        if not text:
            return ""
        text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return " ".join(text.split()).strip()

    # -------------------------------------------------------------------------
    # Concrete tools
    # -------------------------------------------------------------------------

    def extract_clean_rss_items_json(self, xml_text: str, limit: int = 5) -> str:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return "[]"

        items = []
        for item in channel.findall("item")[:limit]:
            title = (item.findtext("title", "") or "").strip()
            pub = (item.findtext("pubDate", "") or "").strip()
            desc_raw = item.findtext("description", "") or ""
            summary = self.strip_html(desc_raw)
            items.append({
                "title": title,
                "pubDate": pub,
                "summary": summary
            })

        return json.dumps(items, ensure_ascii=False, indent=2)

    def get_hardware_state(self, components):
        logger.info(f"[TOOL EXEC] get_hardware_state({components!r})")
        try:
            resp = requests.post(
                f"{self.url}/api/hardware",
                json={"components": components},
                timeout=2
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info("[TOOL RESULT] get_hardware_state -> ok")
            return {"status": "ok", "data": data}
        except Exception as e:
            logger.info(f"[TOOL RESULT] get_hardware_state -> error: {e}")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def substitute_env_tokens(url: str) -> str:
        # Replace ANY ALL_CAPS_TOKEN with its env var if present
        for match in re.findall(r"[A-Z0-9_]+", url):
            if match in os.environ:
                url = url.replace(match, os.environ[match])
        return url

    def fetch_api(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        url = self.substitute_env_tokens(url)
        logger.info(f"[TOOL EXEC] fetch_api(url={url!r}, params={params!r})")
        try:
            if headers is None:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"
                    )
                }

            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            text = r.text

            # NEWS PREPROCESSING
            if params and params.get("api_type") == "news":
                cleaned_json = self.extract_clean_rss_items_json(text, limit=5)
                parsed = json.loads(cleaned_json)
                logger.info("[TOOL RESULT] fetch_api -> news ok")
                return {
                    "status": r.status_code,
                    "reason": "ok" if r.ok else "http_error",
                    "items": parsed,
                    "url": url,
                }

            # DEFAULT RETURN
            logger.info("[TOOL RESULT] fetch_api -> ok")
            return {
                "status": r.status_code,
                "reason": "ok" if r.ok else "http_error",
                "json": r.json() if "application/json" in r.headers.get("Content-Type", "") else None,
                "text": text,
                "url": url,
            }

        except Exception as e:
            logger.info(f"[TOOL RESULT] fetch_api -> error: {e}")
            return {
                "url": url,
                "status": "error",
                "reason": str(e),
                "json": None,
                "text": ""
            }

    @staticmethod
    def get_world_state(keys):
        logger.info(f"[TOOL EXEC] get_world_state({keys!r})")
        data = {k: "<stub>" for k in keys}
        logger.info("[TOOL RESULT] get_world_state -> ok")
        return {"status": "ok", "data": data}

    @staticmethod
    def get_perception(sensors):
        logger.info(f"[TOOL EXEC] get_perception({sensors!r})")
        data = {s: "<stub>" for s in sensors}
        logger.info("[TOOL RESULT] get_perception -> ok")
        return {"status": "ok", "data": data}

    @staticmethod
    def memory_search(query):
        logger.info(f"[TOOL EXEC] memory_search({query!r})")
        logger.info("[TOOL RESULT] memory_search -> ok")
        return {"status": "ok", "results": []}

    @staticmethod
    def memory_write(data):
        logger.info(f"[TOOL EXEC] memory_write(...)")
        logger.info("[TOOL RESULT] memory_write -> ok")
        return {"status": "ok"}

    @staticmethod
    def load_api_instructions(module: str):
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

    # -------------------------------------------------------------------------
    # Tool registry and specs
    # -------------------------------------------------------------------------

    TOOLS: Dict[str, str] = {
        "fetch_api": "fetch_api",
        "get_world_state": "get_world_state",
        "get_hardware_state": "get_hardware_state",
        "get_perception": "get_perception",
        "memory_search": "memory_search",
        "memory_write": "memory_write",
        "load_api_instructions": "load_api_instructions",
    }

    TOOL_SPECS: Dict[str, Dict[str, Any]] = {
        "fetch_api": {
            "kind": "api",
            "allowed_domains": [
                "wikipedia.org",
                "openweathermap.org",
                "cbc.ca",
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
                "perception",
            ],
        },
    }

    TOOL_ALIASES: Dict[str, str] = {
        "api": "fetch_api",
        "json": "fetch_api",
        "weather": "fetch_api",
        "news": "fetch_api",
        "wiki": "fetch_api",
    }

    # -------------------------------------------------------------------------
    # Blacklist / whitelist
    # -------------------------------------------------------------------------

    @staticmethod
    def check_lists(url: str) -> str:
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

    # -------------------------------------------------------------------------
    # Tool call normalization
    # -------------------------------------------------------------------------

    def map_tool_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return name
        return self.TOOL_ALIASES.get(name, name)

    def normalize_tool_call(
        self,
        tool_name: Optional[str],
        tool_args: Dict[str, Any],
        last_tool_call: Optional[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        # Map aliases
        tool_name = self.map_tool_name(tool_name)

        if tool_name == "get_hardware_state":
            # Normalize component(s) -> components: List[str]
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

        if tool_name == "fetch_api":
            # Remove unused "method"
            tool_args.pop("method", None)

            params = tool_args.get("params")
            if not isinstance(params, dict):
                params = {}

            # If last tool was load_api_instructions for news, tag api_type=news
            if last_tool_call and last_tool_call[0] == "load_api_instructions":
                if last_tool_call[1].get("tool_args", {}).get("module") == "news":
                    params["api_type"] = "news"

            tool_args["params"] = params

        return tool_name, tool_args

    # -------------------------------------------------------------------------
    # Tool call validation
    # -------------------------------------------------------------------------

    def validate_tool_call(
        self,
        tool_name: Optional[str],
        tool_args: Dict[str, Any],
        messages: list,
        last_tool_call: Optional[Tuple[str, Dict[str, Any]]],
    ) -> str:
        """
        Returns: "ok", "block", or "finalize"
        """

        if not tool_name:
            return "block"

        # Illegal tool
        if tool_name not in self.TOOLS:
            messages.append({
                "role": "system",
                "content": (
                    f"'{tool_name}' is not a valid tool. "
                    "Do NOT call this tool again. Use the information you already have."
                )
            })
            return "block"

        spec = self.TOOL_SPECS.get(tool_name, {})

        # Allowed components (hardware)
        if "allowed_components" in spec:
            allowed = set(spec["allowed_components"])
            requested = set(tool_args.get("components", []))
            if not requested.issubset(allowed):
                messages.append({
                    "role": "system",
                    "content": (
                        f"Tool call blocked: invalid hardware components {requested - allowed}. "
                        "Use only allowed components."
                    )
                })
                return "block"

        # Allowed domains (API)
        if "allowed_domains" in spec:
            url = tool_args.get("url", "")
            host = urlparse(url).hostname or ""
            if not any(host.endswith(dom) for dom in spec["allowed_domains"]):
                messages.append({
                    "role": "system",
                    "content": (
                        f"Tool call blocked: domain '{host}' is not allowed for fetch_api."
                    )
                })
                return "block"

            # Blacklist/whitelist
            list_status = self.check_lists(url)
            if list_status != "ok":
                messages.append({
                    "role": "system",
                    "content": (
                        f"Tool call blocked by {list_status}. "
                        "Do not call this URL."
                    )
                })
                return "block"

        # Duplicate exact call (non-meta)
        if tool_name != "load_api_instructions" and last_tool_call:
            last_name, last_payload = last_tool_call
            last_args = last_payload.get("tool_args", {})
            if last_name == tool_name and last_args == tool_args:
                messages.append({
                    "role": "system",
                    "content": (
                        "You already called this tool with the same arguments. "
                        "Do not repeat the same tool call."
                    )
                })
                return "block"

        return "ok"

    # -------------------------------------------------------------------------
    # Finalization logic
    # -------------------------------------------------------------------------

    @staticmethod
    def get_last_tool_call(messages) -> Optional[Tuple[str, Dict[str, Any]]]:
        for m in reversed(messages):
            if m.get("role") == "tool":
                try:
                    return (m["name"], json.loads(m["content"]))
                except Exception:
                    return None
        return None

    def should_finalize(self, messages) -> bool:
        last = self.get_last_tool_call(messages)
        if not last:
            return False

        tool_name, payload = last
        result = payload.get("result", {})

        # Only finalize on success
        if not isinstance(result, dict) or result.get("status") == "error":
            return False

        tool_args = payload.get("tool_args", {})

        # Hardware finalization
        if tool_name == "get_hardware_state":
            return True

        # News finalization
        if tool_name == "fetch_api":
            params = tool_args.get("params", {})
            if params.get("api_type") == "news":
                return True

        # Future: world, memory, perception, etc.
        return False

    def inject_finalization_message(self, messages):
        last = self.get_last_tool_call(messages)
        if not last:
            return

        tool_name, payload = last
        tool_args = payload.get("tool_args", {})

        if tool_name == "get_hardware_state":
            messages.append({
                "role": "system",
                "content": (
                    "You have retrieved the hardware status. "
                    "Do NOT call any more tools. "
                    "Use the hardware information to produce the final cognition JSON."
                )
            })
            logger.info("[FINALIZE] triggered by get_hardware_state")
            return

        if tool_name == "fetch_api":
            params = tool_args.get("params", {})
            if params.get("api_type") in ("weather", "news"):
                messages.append({
                    "role": "system",
                    "content": (
                        "You have retrieved the requested data. "
                        "Do NOT call any more tools. "
                        "Use the retrieved data to produce the final cognition JSON."
                    )
                })
                logger.info(f"[FINALIZE] triggered by {tool_name}({params.get('api_type')})")
                return

    # -------------------------------------------------------------------------
    # Tool execution
    # -------------------------------------------------------------------------

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], messages: list):
        tool_attr = self.TOOLS.get(tool_name)
        if not isinstance(tool_attr, str):
            messages.append({
                "role": "system",
                "content": f"Unknown tool '{tool_name}'. Do not call this tool again."
            })
            return

        tool_fn = getattr(self, tool_attr, None)
        if tool_fn is None:
            messages.append({
                "role": "system",
                "content": f"Unknown tool '{tool_name}'. Do not call this tool again."
            })
            return

        logger.info(f"[TOOL REQUEST] name={tool_name} args={tool_args!r}")

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

        # Meta-tools: instructions only
        if tool_name == "load_api_instructions":
            messages.append({
                "role": "system",
                "content": (
                    "You have loaded the API instructions for this tool. "
                    "Use these instructions to fetch the actual data needed to answer the user."
                )
            })
            return

        # Error reporting
        if isinstance(result, dict) and result.get("status") == "error":
            messages.append({
                "role": "system",
                "content": (
                    "The tool returned an error while retrieving the requested data. "
                    "You must inform the user that the data could not be retrieved, "
                    "and avoid guessing specific values."
                )
            })
        else:
            messages.append({
                "role": "system",
                "content": "Continue reasoning with the tool result already fetched."
            })

    # -------------------------------------------------------------------------
    # LLM interaction
    # -------------------------------------------------------------------------

    def call_llm_for_tool_reasoning(self, messages, model):
        logger.info("\n[LLM INPUT MESSAGES]\n%s\n", json.dumps(messages, indent=2))
        resp = cast(Dict[str, Any], model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        ))
        logger.info("\n[LLM RAW OUTPUT]\n%s\n", resp)

        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        tool_name, tool_args = self.extract_tool_call(content)
        tool_args = tool_args or {}
        logger.info(f"[PARSED TOOL CALL] name={tool_name}, args={tool_args}")
        return content, tool_name, tool_args

    # -------------------------------------------------------------------------
    # Main tool loop
    # -------------------------------------------------------------------------

    def run_with_tools(self, messages, max_tool_loops: int = 5):
        loop_count = 0

        while loop_count < max_tool_loops:
            loop_count += 1

            # If we already have enough data, finalize
            if self.should_finalize(messages):
                self.inject_finalization_message(messages)
                final_content, _, _ = self.call_llm_for_tool_reasoning(
                    messages, model=self.model
                )
                logger.info("[PIPELINE DONE]")
                return final_content

            # Ask LLM what to do next
            content, tool_name, tool_args = self.call_llm_for_tool_reasoning(
                messages, model=self.model
            )

            # If no tool requested, treat as final cognition JSON
            if not tool_name:
                logger.info("[PIPELINE DONE] no tool requested")
                return content

            last_tool_call = self.get_last_tool_call(messages)
            tool_name, tool_args = self.normalize_tool_call(
                tool_name, tool_args, last_tool_call
            )

            decision = self.validate_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                messages=messages,
                last_tool_call=last_tool_call,
            )

            if decision == "block":
                # Let LLM see the system message and try again
                continue

            # Execute tool
            if isinstance(tool_name, str):
                self.execute_tool(tool_name, tool_args, messages)
            else:
                logger.info("[PIPELINE] Invalid tool_name (None). Skipping execution.")
                continue

        # Max loops reached: final call, no more tools
        messages.append({
            "role": "system",
            "content": (
                "You have reached the maximum number of tool calls. "
                "Do NOT call any more tools. "
                "Use the information you have to produce the final cognition JSON."
            )
        })
        final_content, _, _ = self.call_llm_for_tool_reasoning(
            messages, model=self.model
        )
        logger.info("[PIPELINE DONE] max loops reached")
        return final_content
