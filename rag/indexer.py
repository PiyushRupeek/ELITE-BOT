import hashlib
import chromadb
from llm.ollama_client import OllamaClient
from rag.chunker import chunk_repo
import config


COLLECTION_NAME = "codebase"


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _get_indexed_hashes(collection) -> dict[str, str]:
    """Return {file_path: hash} for all already-indexed files."""
    try:
        results = collection.get(include=["metadatas"])
        hashes = {}
        for meta in results["metadatas"]:
            fp = meta.get("file_path", "")
            fh = meta.get("file_hash", "")
            if fp and fh:
                hashes[fp] = fh
        return hashes
    except Exception:
        return {}


def index_repo(repo_path: str, ollama: OllamaClient | None = None, force: bool = False) -> tuple[int, int]:
    """
    Index a repo into ChromaDB.
    Returns (indexed_count, skipped_count).
    Skips files whose content hash hasn't changed unless force=True.
    """
    if ollama is None:
        ollama = OllamaClient()

    collection = get_collection()
    chunks = chunk_repo(repo_path)
    print(f"  Found {len(chunks)} chunks in {repo_path}")

    # Build map of already-indexed file hashes
    indexed_hashes = {} if force else _get_indexed_hashes(collection)

    # Group chunks by file so we can skip entire files at once
    from collections import defaultdict
    chunks_by_file: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[chunk.file_path].append(chunk)

    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    BATCH = 50
    indexed_count = 0
    skipped_count = 0

    def flush_batch():
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
        if not force and indexed_hashes.get(file_path) == current_hash:
            skipped_count += len(file_chunks)
            continue

        for chunk in file_chunks:
            embedding = ollama.embed(chunk.content[:4000])
            batch_ids.append(chunk.id)
            batch_embeddings.append(embedding)
            batch_documents.append(chunk.content)
            batch_metadatas.append({
                "file_path": chunk.file_path,
                "repo": chunk.repo,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "file_hash": current_hash,
            })
            indexed_count += 1

            if len(batch_ids) >= BATCH:
                flush_batch()
                print(f"  Indexed {indexed_count} chunks so far...")

    flush_batch()
    print(f"  Done: {indexed_count} indexed, {skipped_count} skipped (unchanged) from {repo_path}")
    return indexed_count, skipped_count
