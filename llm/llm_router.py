"""
LLM Router — smart routing between Groq (cloud) and Ollama (local).

Routing strategy:
  chat()  → tries Groq first (fast, cloud), falls back to Ollama on any failure
  embed() → always Ollama (Groq has no embedding API)

Why this design?
  - Groq gives 2-5s responses vs Ollama's 30-90s — much better UX
  - But Groq has a free-tier rate limit (~14,400 req/day, 500K tokens/min)
  - When the limit is hit (HTTP 429), Ollama ensures the bot always responds
  - Users see no difference — fallback is invisible

This class is a drop-in replacement for OllamaClient (same interface).
The agent only calls self.ollama.chat() and self.ollama.embed() —
swapping OllamaClient for LLMRouter requires no other code changes.
"""
import logging
import httpx
import config
from llm.ollama_client import OllamaClient
from llm.grok_client import GrokClient

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM calls to Groq (primary) or Ollama (fallback)."""

    def __init__(self):
        # Always initialise Ollama — it's the guaranteed fallback
        self.ollama = OllamaClient()

        # Only initialise Groq if it's enabled and a key is provided
        self.grok = GrokClient() if (config.GROK_ENABLED and config.GROK_API_KEY) else None

        if self.grok:
            logger.info("LLM Router: Grok (%s) → Ollama (%s) fallback", config.GROK_MODEL, config.OLLAMA_CHAT_MODEL)
        else:
            logger.info("LLM Router: Ollama only (%s)", config.OLLAMA_CHAT_MODEL)

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """
        Generate a chat response, trying Groq first with automatic Ollama fallback.

        Groq failures that trigger fallback:
          - HTTP 429 : rate limit exceeded
          - HTTP 401 : invalid API key
          - HTTP 5xx : Groq server error
          - Network errors : timeout, connection refused, etc.

        In all failure cases, the request is transparently retried on Ollama.
        The Slack user always gets a response — they never see the fallback happen.
        """
        if self.grok:
            try:
                logger.info("Calling Grok (%s)...", config.GROK_MODEL)
                response = self.grok.chat(messages, system_prompt)
                logger.info("Grok responded (%d chars)", len(response))
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Grok rate limit hit (429) — falling back to Ollama")
                else:
                    logger.warning("Grok API error (%d) — falling back to Ollama", e.response.status_code)
            except Exception as e:
                logger.warning("Grok failed (%s) — falling back to Ollama", e)

        # Fallback — always works as long as Ollama is running locally
        logger.info("Calling Ollama (%s)...", config.OLLAMA_CHAT_MODEL)
        return self.ollama.chat(messages, system_prompt)

    def embed(self, text: str) -> list[float]:
        """
        Convert text to a vector embedding.

        Always uses Ollama's nomic-embed-text — Groq has no embedding API.
        This is fine because embeddings are only used for ChromaDB search,
        not for generating answers, so speed is less critical here.
        """
        return self.ollama.embed(text)

    def close(self):
        """Close all HTTP connections cleanly on bot shutdown."""
        self.ollama.close()
        if self.grok:
            self.grok.close()
