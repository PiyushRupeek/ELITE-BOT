#!/usr/bin/env python3
"""
Entry point for the Rupeek Dev Assistant Slack Bot.

Prerequisites:
  1. Copy .env.example to .env and fill in all values
  2. Pull Ollama models:  ollama pull llama3.2 && ollama pull nomic-embed-text
  3. Index the codebase:  python scripts/index_codebase.py
  4. Start the bot:       python main.py
"""
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


if __name__ == "__main__":
    print("Starting Rupeek Dev Assistant...")
    check_ollama()
    from bot.slack_handler import start
    start()
