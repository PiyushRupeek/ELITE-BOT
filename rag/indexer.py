"""
Codebase Indexer — embeds code chunks and stores them in ChromaDB.

Flow:
  index_repo(repo_path)
    ├─ chunk_repo()           → splits files into CodeChunk objects
    ├─ _get_indexed_hashes()  → loads existing file hashes from ChromaDB
    ├─ for each file:
    │   ├─ compute SHA-256 hash of file content
    │   ├─ skip if hash matches stored hash (file unchanged)
    │   └─ else: embed each chunk → upsert into ChromaDB
    └─ returns (indexed_count, skipped_count)

Incremental indexing:
  Each chunk stores a SHA-256 hash of its source file in ChromaDB metadata.
  On re-index, files whose hash hasn't changed are skipped entirely — making
  subsequent runs near-instant when only a few files changed.

  Use --force flag (scripts/index_codebase.py --force) to re-index everything.
"""
import hashlib
from collections import defaultdict
import chromadb
from llm.ollama_client import OllamaClient
from rag.chunker import chunk_repo
import config


# Name of the ChromaDB collection where all code chunks are stored
COLLECTION_NAME = "codebase"


def get_collection():
    """
    Get or create the ChromaDB collection for storing code embeddings.

    Uses cosine similarity as the distance metric — this means vectors are
    compared by direction (semantic meaning), not magnitude (length).
    The collection is persisted to disk at config.CHROMA_PATH.
    """
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine = semantic similarity search
    )


def _file_hash(path: str) -> str:
    """
    Compute SHA-256 hash of a file's raw bytes.

    Used to detect whether a file has changed since last indexing.
    Returns empty string if the file can't be read (deleted/permission error).
    """
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _get_indexed_hashes(collection) -> dict[str, str]:
    """
    Load the file_path → file_hash mapping from all existing ChromaDB entries.

    This tells us which files were already indexed and what their content
    looked like at index time. Used to skip unchanged files on re-index.
    Returns an empty dict if collection is empty or on any error.
    """
    try:
        results = collection.get(include=["metadatas"])
        hashes = {}
        for meta in results["metadatas"]:
            fp = meta.get("file_path", "")
            fh = meta.get("file_hash", "")
            if fp and fh:
                hashes[fp] = fh  # one entry per chunk but hash is the same for all chunks of a file
        return hashes
    except Exception:
        return {}


def index_repo(repo_path: str, ollama: OllamaClient | None = None, force: bool = False) -> tuple[int, int]:
    """
    Index all source files in a repository into ChromaDB.

    For each file:
      - Compute SHA-256 hash of file content
      - If hash matches what's already in ChromaDB → skip (file unchanged)
      - Otherwise → embed each chunk with nomic-embed-text and upsert into ChromaDB

    Chunks are batched in groups of 50 before writing to ChromaDB for efficiency.
    Each chunk's metadata includes file_hash so future runs can skip unchanged files.

    Args:
        repo_path : absolute path to the repository root
        ollama    : OllamaClient instance (shared across repos to reuse connection)
        force     : if True, re-index all files even if hash matches (full re-index)

    Returns:
        (indexed_count, skipped_count) — number of chunks indexed vs skipped
    """
    if ollama is None:
        ollama = OllamaClient()

    collection = get_collection()
    chunks = chunk_repo(repo_path)
    print(f"  Found {len(chunks)} chunks in {repo_path}")

    # Load existing hashes to enable incremental indexing (skip if force=True)
    indexed_hashes = {} if force else _get_indexed_hashes(collection)

    # Group chunks by file so we can make a single hash check per file
    # instead of checking on every chunk individually
    chunks_by_file: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file_path].append(chunk)

    # Batch buffers — we accumulate chunks and upsert in groups of 50
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    BATCH = 50
    indexed_count = 0
    skipped_count = 0

    def flush_batch():
        """Write the current batch to ChromaDB and clear the buffers."""
        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            batch_ids.clear()
            batch_embeddings.clear()
            batch_documents.clear()
            batch_metadatas.clear()

    for file_path, file_chunks in chunks_by_file.items():
        current_hash = _file_hash(file_path)

        # Skip this file entirely if its content hasn't changed since last index
        if not force and indexed_hashes.get(file_path) == current_hash:
            skipped_count += len(file_chunks)
            continue

        # File is new or changed — embed and index all its chunks
        for chunk in file_chunks:
            # Truncate to 4000 chars to stay within the embedding model's token limit
            embedding = ollama.embed(chunk.content[:4000])

            batch_ids.append(chunk.id)               # unique key: "file_path:start-end"
            batch_embeddings.append(embedding)         # vector (768 floats from nomic-embed-text)
            batch_documents.append(chunk.content)      # raw source code text
            batch_metadatas.append({
                "file_path": chunk.file_path,
                "repo": chunk.repo,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "file_hash": current_hash,  # stored so next run can skip this file if unchanged
            })
            indexed_count += 1

            # Flush to ChromaDB every 50 chunks to avoid holding too much in memory
            if len(batch_ids) >= BATCH:
                flush_batch()
                print(f"  Indexed {indexed_count} chunks so far...")

    # Flush any remaining chunks that didn't fill a full batch
    flush_batch()
    print(f"  Done: {indexed_count} indexed, {skipped_count} skipped (unchanged) from {repo_path}")
    return indexed_count, skipped_count
