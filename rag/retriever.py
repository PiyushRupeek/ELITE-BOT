from dataclasses import dataclass
import chromadb
from llm.ollama_client import OllamaClient
from rag.indexer import get_collection
import config


@dataclass
class SearchResult:
    content: str
    file_path: str
    repo: str
    language: str
    start_line: int
    end_line: int
    score: float

    def format(self) -> str:
        return (
            f"**{self.repo}** · `{self.file_path}` lines {self.start_line}-{self.end_line} "
            f"(score: {self.score:.2f})\n```{self.language}\n{self.content.strip()}\n```"
        )


class Retriever:
    """Semantic search over the indexed codebase using ChromaDB + nomic-embed-text."""

    def __init__(self):
        self.ollama = OllamaClient()
        self.collection = get_collection()

    def search(self, query: str, n_results: int = 5) -> list[SearchResult]:
        """Embed the query and return the top-N most relevant code chunks."""
        embedding = self.ollama.embed(query)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
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
                score=1 - dist,
            ))

        return output
