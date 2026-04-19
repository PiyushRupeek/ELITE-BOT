#!/usr/bin/env python3
import sys
import httpx
import config


def check_ollama():
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"✓ Ollama is running  |  models: {', '.join(models) or 'none pulled yet'}")

        missing = []
        if not any(config.OLLAMA_CHAT_MODEL in m for m in models):
            missing.append(config.OLLAMA_CHAT_MODEL)
        if not any(config.OLLAMA_EMBED_MODEL in m for m in models):
            missing.append(config.OLLAMA_EMBED_MODEL)
        if missing:
            print(f"✗ Missing models: {', '.join(missing)}")
            print(f"  Run: ollama pull {' && ollama pull '.join(missing)}")
            sys.exit(1)

        print(f"✓ Required models ready: {config.OLLAMA_CHAT_MODEL}, {config.OLLAMA_EMBED_MODEL}")
    except Exception:
        print("✗ Ollama is not running. Start it with:  brew services start ollama")
        sys.exit(1)


def check_grok():
    if not config.GROK_ENABLED:
        print("  Groq disabled — using Ollama only (set GROK_ENABLED=true in .env to enable)")
        return
    if not config.GROK_API_KEY or config.GROK_API_KEY == "xai-your-key-here":
        print("✗ GROK_ENABLED=true but GROK_API_KEY is not set")
        sys.exit(1)
    print(f"✓ Groq enabled  |  model: {config.GROK_MODEL}  (Ollama is fallback)")


if __name__ == "__main__":
    print("Starting Rupeek Dev Assistant...")
    check_ollama()
    check_grok()
    from bot.slack_handler import start
    start()
