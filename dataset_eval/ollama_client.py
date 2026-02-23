from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OllamaResponse:
    ok: bool
    text: str
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_sec: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = int(timeout_sec)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> OllamaResponse:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        # Ollama supports a server-side JSON mode ("format": "json").
        # When enabled, the model is constrained to emit valid JSON.
        if response_format:
            payload["format"] = response_format

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return OllamaResponse(False, "", f"http_error: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            return OllamaResponse(False, "", f"url_error: {e.reason}")
        except Exception as e:
            return OllamaResponse(False, "", f"error: {e}")

        try:
            obj = json.loads(raw)
        except Exception:
            return OllamaResponse(False, raw, "response_not_json", None)

        txt = str(obj.get("response") or "")

        # If Ollama reports an error, surface it to the caller.
        err = obj.get("error")
        if err:
            return OllamaResponse(False, txt, f"ollama_error: {err}", obj)

        return OllamaResponse(True, txt, None, obj)

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
    ) -> OllamaResponse:
        """Call Ollama /api/chat.

        We prefer /api/chat for structured outputs because it is the primary
        endpoint documented for JSON/JSON-schema constrained generation.
        """

        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if response_format is not None:
            # Supports either "json" or a JSON schema object.
            payload["format"] = response_format

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return OllamaResponse(False, "", f"http_error: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            return OllamaResponse(False, "", f"url_error: {e.reason}")
        except Exception as e:
            return OllamaResponse(False, "", f"error: {e}")

        try:
            obj = json.loads(raw)
        except Exception:
            return OllamaResponse(False, raw, "response_not_json", None)

        err = obj.get("error")
        if err:
            return OllamaResponse(False, "", f"ollama_error: {err}", obj)

        msg = obj.get("message") or {}
        txt = str(msg.get("content") or "")
        return OllamaResponse(True, txt, None, obj)
