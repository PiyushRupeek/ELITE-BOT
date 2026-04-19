#!/usr/bin/env python3
"""
Index repos into ChromaDB.

Usage:
    python scripts/index_codebase.py           # incremental (skips unchanged files)
    python scripts/index_codebase.py --force   # full re-index
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.ollama_client import OllamaClient
from rag.indexer import index_repo


def main():
    force = "--force" in sys.argv

    if not config.REPO_PATHS:
        print("ERROR: REPO_PATHS is empty. Set it in your .env file.")
        sys.exit(1)

    mode = "FULL re-index (--force)" if force else "Incremental (use --force to re-index everything)"
    print(f"Mode: {mode}")

    ollama = OllamaClient()
    total_indexed = 0
    total_skipped = 0

    for repo in config.REPO_PATHS:
        if not os.path.isdir(repo):
            print(f"WARNING: {repo} does not exist, skipping.")
            continue
        print(f"\nIndexing: {repo}")
        indexed, skipped = index_repo(repo, ollama, force=force)
        total_indexed += indexed
        total_skipped += skipped

    print(f"\nTotal: {total_indexed} chunks indexed, {total_skipped} skipped")
    print(f"ChromaDB stored at: {config.CHROMA_PATH}")


if __name__ == "__main__":
    main()
