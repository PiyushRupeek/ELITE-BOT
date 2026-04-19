import os
from dotenv import load_dotenv

# Load all variables from .env file into the environment
load_dotenv()

# ── Slack ─────────────────────────────────────────────────────────────────────
# Bot token (xoxb-) is used to post messages and add reactions
# App token (xapp-) is used for Socket Mode — keeps a persistent WebSocket
# connection to Slack so we don't need a public URL or ngrok
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
# Ollama runs models locally on your Mac — no internet required for inference
# CHAT_MODEL  : used to generate answers (llama3.2 = 3B param, fast on Apple Silicon)
# EMBED_MODEL : converts text to vectors for semantic search (nomic-embed-text)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── ChromaDB (vector database) ────────────────────────────────────────────────
# ChromaDB stores code chunk embeddings on disk so we can do semantic search
# CHROMA_PATH : folder where the vector DB is persisted (rebuild with index_codebase.py)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# ── Groq (cloud LLM — primary model) ─────────────────────────────────────────
# Groq is a fast cloud inference API (10-50x faster than local Ollama)
# GROK_ENABLED : set to true to use Groq as the primary model
# GROK_API_KEY : get from console.groq.com (key starts with gsk_)
# GROK_MODEL   : which Groq-hosted model to use (llama-3.1-8b-instant is fastest)
# GROK_BASE_URL: Groq uses OpenAI-compatible API format
GROK_ENABLED = os.getenv("GROK_ENABLED", "false").lower() == "true"
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "llama-3.1-8b-instant")
GROK_BASE_URL = "https://api.groq.com/openai/v1"

# ── Grafana (log fetching) ────────────────────────────────────────────────────
# Optional — only used during debug intent to fetch recent logs from Loki
# GRAFANA_ENABLED  : set to true once you have Grafana + Loki running
# GRAFANA_LOKI_UID : find in Grafana → Connections → Data sources → Loki → URL
GRAFANA_ENABLED = os.getenv("GRAFANA_ENABLED", "false").lower() == "true"
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
GRAFANA_LOKI_UID = os.getenv("GRAFANA_LOKI_UID", "")

# ── Repos to index ────────────────────────────────────────────────────────────
# Comma-separated absolute paths to repos that will be indexed into ChromaDB
# Add/remove repos and re-run: python scripts/index_codebase.py
REPO_PATHS = [
    p.strip()
    for p in os.getenv("REPO_PATHS", "").split(",")
    if p.strip()
]
