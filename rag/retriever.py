"""
Retriever — semantic search over the indexed codebase.

How it works:
  1. User query is converted to a vector using nomic-embed-text (same model used at index time)
  2. ChromaDB finds the stored code chunk vectors closest to the query vector
  3. "Closest" = most similar meaning (cosine similarity), not keyword matching
  4. Returns top-N results as SearchResult objects with relevance scores

Score interpretation:
  score = 1 - cosine_distance
  score = 1.0 → perfect match
  score = 0.5 → somewhat related
  score < 0.3 → likely irrelevant (filtered out in agent.py)
"""
from dataclasses import dataclass
import chromadb
from llm.ollama_client import OllamaClient
from rag.indexer import get_collection
import config


@dataclass
class SearchResult:
    """A single code chunk returned from semantic search with its relevance score."""
    content: str       # raw source code text of the chunk
    file_path: str     # absolute path to the source file
    repo: str          # repository name
    language: str      # programming language (for syntax highlighting)
    start_line: int    # where this chunk starts in the file (1-based)
    end_line: int      # where this chunk ends in the file (1-based)
    score: float       # cosine similarity score: 0.0 (unrelated) → 1.0 (identical)

    def format(self) -> str:
        """Format this result as a markdown code block with file metadata header."""
        return (
            f"**{self.repo}** · `{self.file_path}` lines {self.start_line}-{self.end_line} "
            f"(score: {self.score:.2f})\n```{self.language}\n{self.content.strip()}\n```"
        )


class Retriever:
    """Semantic search client — embeds a query and finds the most relevant code chunks."""

    def __init__(self):
        # OllamaClient is used here only for embedding (nomic-embed-text), not for chat
        self.ollama = OllamaClient()
        # Connect to the persisted ChromaDB collection created by the indexer
        self.collection = get_collection()

    def search(self, query: str, n_results: int = 5) -> list[SearchResult]:
        """
        Find the most semantically relevant code chunks for a given query.

        Steps:
          1. Embed the query string into a vector using nomic-embed-text
          2. Query ChromaDB for the n closest stored vectors (cosine distance)
          3. Convert distances to similarity scores and wrap in SearchResult objects

        Args:
            query     : natural language or keyword search string
            n_results : how many chunks to return (adapted per intent in agent.py)

        Returns:
            List of SearchResult ordered by relevance (highest score first)
        """
        # Convert the query text to a vector — must use the same model as at index time
        embedding = self.ollama.embed(query)

        # Ask ChromaDB for the n nearest vectors to our query vector
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],  # fetch text, metadata, and distance
        )

        output = []
        # ChromaDB returns lists-of-lists (one per query) — we only send one query at a time
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(SearchResult(
                content=doc,
                file_path=meta["file_path"],
                repo=meta["repo"],
                language=meta["language"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                score=1 - dist,  # convert cosine distance (0=identical) to similarity (1=identical)
            ))

        return output
