"""
Thin wrapper around llama-server's OpenAI-compatible API.
Nothing fancy -- this is intentionally dumb so you can see exactly what's
being sent on every call.
"""

import json
import requests
import config


class LLMClient:
    def __init__(self, base_url: str = config.BASE_URL, model: str = config.MODEL):
        self.base_url = base_url
        self.model = model

    def _headers(self) -> dict:
        headers = {}
        if config.API_KEY:
            headers["Authorization"] = f"Bearer {config.API_KEY}"
        return headers

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
              temperature: float = 0.3, max_tokens: int = 4096) -> dict:
        """
        Send a chat completion request. Returns the raw 'message' dict from the
        response (role, content, tool_calls if any).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=300,
        )
        if not resp.ok:
            # The default HTTPError message hides the actual reason -- llama-server
            # puts useful detail (context overflow, bad param, etc.) in the body.
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} error from llama-server: {resp.text}"
            )
        data = resp.json()
        return data["choices"][0]["message"]

    def embed(self, text: str) -> list[float] | None:
        """
        Returns an embedding vector, or None if no embedding server is configured.
        """
        if config.EMBED_URL is None:
            return None
        resp = requests.post(
            f"{config.EMBED_URL}/embeddings",
            json={"model": config.EMBED_MODEL, "input": text},
            headers=self._headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]