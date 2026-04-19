import os
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

GRAFANA_ENABLED = os.getenv("GRAFANA_ENABLED", "false").lower() == "true"
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
GRAFANA_LOKI_UID = os.getenv("GRAFANA_LOKI_UID", "")

REPO_PATHS = [
    p.strip()
    for p in os.getenv("REPO_PATHS", "").split(",")
    if p.strip()
]
