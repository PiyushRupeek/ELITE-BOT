"""
Ollama HTTP client — wraps the local Ollama REST API.

Ollama runs LLMs locally on your Mac (no internet needed after model download).
This client is used for:
  - Embeddings (nomic-embed-text): converts text → vector for ChromaDB search
  - Chat (llama3.2): generates answers (used as fallback when Groq is unavailable)
  - Streaming chat: same as chat but yields tokens as they arrive

Connection reuse:
  A single persistent httpx.Client is created per OllamaClient instance.
  This reuses the TCP connection across calls (~3x faster than opening a new
  connection on every request).

Retry logic:
  Network-level errors (TransportError) are retried up to 3 times with
  exponential backoff (1s, 2s). API-level errors (4xx/5xx) are not retried.
"""
import time
import json
import httpx
from typing import Generator
import config


class OllamaClient:
    """HTTP client for the local Ollama API (http://localhost:11434)."""

    def __init__(self):
        self.chat_model = config.OLLAMA_CHAT_MODEL    # e.g. "llama3.2"
        self.embed_model = config.OLLAMA_EMBED_MODEL  # e.g. "nomic-embed-text"

        # Persistent connection — reuses TCP socket across multiple requests
        self._client = httpx.Client(
            base_url=config.OLLAMA_BASE_URL,
            timeout=httpx.Timeout(connect=10, read=120, write=30, pool=5),
        )

    def _post_with_retry(self, path: str, json_body: dict, timeout: float = 120) -> httpx.Response:
        """
        POST to Ollama with automatic retry on network errors.

        Retries up to 3 times on TransportError (connection reset, timeout, etc.).
        Uses exponential backoff: waits 1s after 1st failure, 2s after 2nd failure.
        Raises the error immediately on HTTP 4xx/5xx (don't retry those).
        """
        for attempt in range(3):
            try:
                resp = self._client.post(path, json=json_body, timeout=timeout)
                resp.raise_for_status()  # raise on 4xx/5xx responses
                return resp
            except httpx.TransportError:
                if attempt == 2:
                    raise  # give up after 3 attempts
                time.sleep(2 ** attempt)  # 1s after attempt 0, 2s after attempt 1

    def embed(self, text: str) -> list[float]:
        """
        Convert text into a vector embedding using nomic-embed-text.

        The returned vector (768 floats) represents the semantic meaning of the text.
        Similar texts produce similar vectors — this is how ChromaDB finds relevant code.

        Used by:
          - indexer.py at index time (to store code chunk vectors)
          - retriever.py at query time (to embed the user's question)
        """
        resp = self._post_with_retry(
            "/api/embeddings",
            {"model": self.embed_model, "prompt": text},
            timeout=60,  # embedding is faster than chat generation
        )
        return resp.json()["embedding"]

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """
        Send a conversation to Ollama and get a full text response (non-streaming).

        Args:
            messages      : list of {"role": "user"|"assistant", "content": "..."}
            system_prompt : optional instruction prepended as a {"role": "system"} message

        The system_prompt defines the LLM's persona and response format
        (e.g. "You are a debugging expert, always respond with Root Cause → Fix").

        Returns the full response text once generation is complete (~30-90s for llama3.2).
        """
        payload_messages = []
        if system_prompt:
            # System message must be first in the list for Ollama to respect it
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        resp = self._post_with_retry(
            "/api/chat",
            {"model": self.chat_model, "messages": payload_messages, "stream": False},
            timeout=120,
        )
        return resp.json()["message"]["content"]

    def chat_stream(self, messages: list[dict], system_prompt: str = "") -> Generator[str, None, None]:
        """
        Stream a conversation response token-by-token from Ollama.

        Yields text chunks as they are generated instead of waiting for the full
        response. Useful for showing partial responses in Slack as they arrive.

        Ollama streams JSONL — each line is a JSON object:
          {"message": {"content": "Hello"}, "done": false}
          {"message": {"content": " world"}, "done": false}
          {"done": true}

        Silently skips lines that fail to parse (malformed JSON).
        """
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
                        continue  # skip malformed lines without crashing

    def close(self):
        """Close the persistent HTTP connection. Call when shutting down the bot."""
        self._client.close()
