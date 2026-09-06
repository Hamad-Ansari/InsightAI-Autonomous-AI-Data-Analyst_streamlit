"""
InsightAI - Ollama client.

A thin, dependency-light wrapper around Ollama's HTTP API. It can check server
availability, list models and generate completions. If Ollama is not running the
application keeps working in deterministic mode (the UI shows a clear message).
"""

from __future__ import annotations

import json
from typing import Optional

import requests

from config.settings import get_settings


class OllamaClient:
    """Minimal Ollama API client."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[int] = None):
        s = get_settings()
        self.base_url = (base_url or s.ollama_base_url).rstrip("/")
        self.model = model or s.ollama_model
        self.timeout = timeout or s.ollama_timeout

    def is_available(self) -> bool:
        """Return True if the Ollama server responds to /api/tags."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m.get("name") for m in data.get("models", [])]
        except Exception:
            return []

    def generate(self, prompt: str, system: str = "", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, stream: bool = False) -> str:
        """Generate a completion for ``prompt``."""
        s = get_settings()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else s.ollama_temperature,
                "num_predict": max_tokens if max_tokens is not None else s.ollama_max_tokens,
            },
        }
        if system:
            payload["system"] = system
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            if stream:
                # Accumulate the (single) response for simplicity.
                text = ""
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line.decode("utf-8"))
                    text += obj.get("response", "")
                    if obj.get("done"):
                        break
                return text
            data = r.json()
            return data.get("response", "")
        except Exception as exc:
            return f"[Ollama error] {exc}"

    def chat(self, messages, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """Chat completion via /api/chat."""
        s = get_settings()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else s.ollama_temperature,
                "num_predict": max_tokens if max_tokens is not None else s.ollama_max_tokens,
            },
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            return f"[Ollama error] {exc}"
