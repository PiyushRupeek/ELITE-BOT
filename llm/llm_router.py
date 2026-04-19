"""
LLM Router — Grok first, Ollama fallback.

chat()  → tries Grok, falls back to Ollama on any error (429, 401, network, etc.)
embed() → always Ollama (Grok has no embedding API)

Drop-in replacement for OllamaClient — same interface.
"""
import logging
import httpx
import config
from llm.ollama_client import OllamaClient
from llm.grok_client import GrokClient

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self):
        self.ollama = OllamaClient()
        self.grok = GrokClient() if (config.GROK_ENABLED and config.GROK_API_KEY) else None

        if self.grok:
            logger.info("LLM Router: Grok (%s) → Ollama (%s) fallback", config.GROK_MODEL, config.OLLAMA_CHAT_MODEL)
        else:
            logger.info("LLM Router: Ollama only (%s)", config.OLLAMA_CHAT_MODEL)

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
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

        logger.info("Calling Ollama (%s)...", config.OLLAMA_CHAT_MODEL)
        return self.ollama.chat(messages, system_prompt)

    def embed(self, text: str) -> list[float]:
        # Embeddings always use Ollama (nomic-embed-text)
        return self.ollama.embed(text)

    def close(self):
        self.ollama.close()
        if self.grok:
            self.grok.close()
