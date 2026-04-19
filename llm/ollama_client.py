import time
import json
import httpx
from typing import Generator
import config


class OllamaClient:
    """HTTP client for the local Ollama API (http://localhost:11434)."""

    def __init__(self):
        self.chat_model = config.OLLAMA_CHAT_MODEL
        self.embed_model = config.OLLAMA_EMBED_MODEL
        self._client = httpx.Client(
            base_url=config.OLLAMA_BASE_URL,
            timeout=httpx.Timeout(connect=10, read=120, write=30, pool=5),
        )

    def _post_with_retry(self, path: str, json_body: dict, timeout: float = 120) -> httpx.Response:
        """POST to Ollama with up to 3 retries on network errors (exponential backoff)."""
        for attempt in range(3):
            try:
                resp = self._client.post(path, json=json_body, timeout=timeout)
                resp.raise_for_status()
                return resp
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def embed(self, text: str) -> list[float]:
        """Convert text to a 768-dim vector embedding using nomic-embed-text."""
        resp = self._post_with_retry(
            "/api/embeddings",
            {"model": self.embed_model, "prompt": text},
            timeout=60,
        )
        return resp.json()["embedding"]

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """Send a conversation to Ollama and return the full response text."""
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        resp = self._post_with_retry(
            "/api/chat",
            {"model": self.chat_model, "messages": payload_messages, "stream": False},
            timeout=120,
        )
        return resp.json()["message"]["content"]

    def chat_stream(self, messages: list[dict], system_prompt: str = "") -> Generator[str, None, None]:
        """Stream a response token-by-token from Ollama."""
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        with self._client.stream(
            "POST",
            "/api/chat",
            json={"model": self.chat_model, "messages": payload_messages, "stream": True},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if not chunk.get("done"):
                            yield chunk["message"]["content"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def close(self):
        self._client.close()
