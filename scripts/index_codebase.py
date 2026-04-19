#!/usr/bin/env python3
"""
Index all configured repos into ChromaDB.

Usage:
    python scripts/index_codebase.py           # incremental — skips unchanged files (fast)
    python scripts/index_codebase.py --force   # full re-index of everything
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

    if force:
        print("Mode: FULL re-index (--force)")
    else:
        print("Mode: Incremental (skips unchanged files). Use --force to re-index everything.")

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
