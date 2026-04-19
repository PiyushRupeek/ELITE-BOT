import logging
import httpx
import config
from llm.ollama_client import OllamaClient
from llm.grok_client import GrokClient

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes chat() to Groq (primary) with automatic Ollama fallback. embed() always uses Ollama."""

    def __init__(self):
        self.ollama = OllamaClient()
        self.grok = GrokClient() if (config.GROK_ENABLED and config.GROK_API_KEY) else None

        if self.grok:
            logger.info("LLM Router: Groq (%s) → Ollama (%s) fallback", config.GROK_MODEL, config.OLLAMA_CHAT_MODEL)
        else:
            logger.info("LLM Router: Ollama only (%s)", config.OLLAMA_CHAT_MODEL)

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        """Try Groq first; fall back to Ollama on any error (429, 401, 5xx, network)."""
        if self.grok:
            try:
                logger.info("Calling Groq (%s)...", config.GROK_MODEL)
                response = self.grok.chat(messages, system_prompt)
                logger.info("Groq responded (%d chars)", len(response))
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("Groq rate limit hit (429) — falling back to Ollama")
                else:
                    logger.warning("Groq API error (%d) — falling back to Ollama", e.response.status_code)
            except Exception as e:
                logger.warning("Groq failed (%s) — falling back to Ollama", e)

        logger.info("Calling Ollama (%s)...", config.OLLAMA_CHAT_MODEL)
        return self.ollama.chat(messages, system_prompt)

    def embed(self, text: str) -> list[float]:
        return self.ollama.embed(text)

    def close(self):
        self.ollama.close()
        if self.grok:
            self.grok.close()
