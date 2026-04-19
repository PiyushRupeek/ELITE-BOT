"""
Groq API client — cloud inference, OpenAI-compatible format.

Groq (groq.com) runs open-source models (like llama3) on custom hardware
called LPUs (Language Processing Units) that are 10-50x faster than GPUs.

API format is identical to OpenAI — same message structure, same response shape.
The only differences from OllamaClient:
  - Response key: choices[0].message.content  (vs Ollama's message.content)
  - No streaming needed — Groq responses arrive in 2-5s anyway
  - No embed() method — Groq doesn't offer an embeddings API

Error handling:
  - HTTP 429 → rate limit hit → LLMRouter catches this and falls back to Ollama
  - HTTP 401 → invalid API key
  - HTTP 500 → Groq server error
  All errors are raised as httpx.HTTPStatusError for the router to handle.

Get API key at: https://console.groq.com
"""
import httpx
import config


class GrokClient:
    """HTTP client for the Groq cloud inference API."""

    def __init__(self):
        # Single persistent client — reuses connection across requests
        self._client = httpx.Client(
            base_url=config.GROK_BASE_URL,  # https://api.groq.com/openai/v1
            headers={
                "Authorization": f"Bearer {config.GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10, read=60, write=30, pool=5),
        )
        self.model = config.GROK_MODEL  # e.g. "llama-3.1-8b-instant" or "openai/gpt-oss-120b"

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """
        Send a conversation to Groq and return the response text.

        Uses OpenAI-compatible /chat/completions endpoint.
        Raises httpx.HTTPStatusError on any non-2xx response — the caller
        (LLMRouter) catches HTTP 429 to trigger fallback to Ollama.

        Args:
            messages      : list of {"role": "user"|"assistant", "content": "..."}
            system_prompt : instruction prepended as a system message

        Returns the full response text (typically arrives in 2-5 seconds).
        """
        payload_messages = []
        if system_prompt:
            # System message must be first for the model to treat it as instructions
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        resp = self._client.post(
            "/chat/completions",
            json={"model": self.model, "messages": payload_messages},
        )
        resp.raise_for_status()  # raises HTTPStatusError on 429 (rate limit), 401, 500, etc.

        # Groq response format: {"choices": [{"message": {"content": "..."}}]}
        # Different from Ollama which uses: {"message": {"content": "..."}}
        return resp.json()["choices"][0]["message"]["content"]

    def close(self):
        """Close the persistent HTTP connection."""
        self._client.close()
