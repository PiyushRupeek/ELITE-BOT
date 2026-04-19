import httpx
import config


class GrokClient:
    """HTTP client for the Groq cloud inference API (OpenAI-compatible)."""

    def __init__(self):
        self._client = httpx.Client(
            base_url=config.GROK_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10, read=60, write=30, pool=5),
        )
        self.model = config.GROK_MODEL

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """Send a conversation to Groq and return the response text."""
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        resp = self._client.post(
            "/chat/completions",
            json={"model": self.model, "messages": payload_messages},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def close(self):
        self._client.close()
