"""
Code Chunker — splits source files into overlapping line-based chunks.

Why chunking?
  Embedding models have a token limit (~512 tokens). Large files must be
  split into smaller pieces before embedding. We use overlapping chunks
  so that code spanning a boundary (e.g. a function definition followed
  by its body) appears in at least one complete chunk.

Flow:
  chunk_repo(repo_path)
    └─ chunk_file(file_path)
         └─ returns list[CodeChunk] with content + metadata
"""
import os
from dataclasses import dataclass

# ── File type configuration ───────────────────────────────────────────────────

# Only these extensions are indexed — everything else is ignored
SUPPORTED_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx",   # TypeScript / JavaScript
    ".java", ".py", ".go",           # Backend languages
    ".json", ".yaml", ".yml", ".md", # Config and docs
    ".sql", ".sh", ".env.example",   # DB and shell scripts
}

# These directories are never walked — avoids indexing dependencies and build output
SKIP_DIRS = {
    "node_modules", ".git", "target", "build", "dist",
    ".idea", ".vscode", "__pycache__", ".gradle", ".mvn",
    "coverage", ".nyc_output", "out",
}

# ── Chunking parameters ───────────────────────────────────────────────────────

CHUNK_SIZE = 60   # max lines per chunk — keeps each chunk within embedding token limits
OVERLAP = 10      # lines shared between adjacent chunks — prevents losing context at boundaries


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    """Represents a single chunk of source code with its location metadata."""
    content: str       # raw source code text for this chunk
    file_path: str     # absolute path to the source file
    repo: str          # repository name (basename of repo root)
    language: str      # programming language (used for syntax highlighting in responses)
    start_line: int    # 1-based line number where this chunk starts in the file
    end_line: int      # 1-based line number where this chunk ends in the file

    @property
    def id(self) -> str:
        # Unique ID used as the key in ChromaDB — file path + line range
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _language(ext: str) -> str:
    """Map file extension to a language name used for markdown code block formatting."""
    return {
        ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript",
        ".java": "java", ".py": "python",
        ".go": "go", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml",
        ".md": "markdown", ".sql": "sql",
        ".sh": "bash",
    }.get(ext, "text")


# ── Core chunking functions ───────────────────────────────────────────────────

def chunk_file(file_path: str, repo_root: str) -> list[CodeChunk]:
    """
    Split a single source file into overlapping chunks.

    - Files with unsupported extensions are skipped (returns []).
    - Files <= CHUNK_SIZE lines are returned as a single chunk.
    - Larger files are split with a sliding window of size CHUNK_SIZE,
      advancing CHUNK_SIZE - OVERLAP lines each step so adjacent chunks
      share OVERLAP lines of context.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return []

    try:
        # errors="ignore" skips undecodable bytes (e.g. binary content in .md)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return []

    if not lines:
        return []

    repo_name = os.path.basename(repo_root.rstrip("/"))
    lang = _language(ext)
    chunks = []

    # Small files fit in one chunk — no splitting needed
    if len(lines) <= CHUNK_SIZE:
        chunks.append(CodeChunk(
            content="".join(lines),
            file_path=file_path,
            repo=repo_name,
            language=lang,
            start_line=1,
            end_line=len(lines),
        ))
        return chunks

    # Large files: sliding window with overlap
    # e.g. CHUNK_SIZE=60, OVERLAP=10 → stride=50
    # Chunk 1: lines 1-60, Chunk 2: lines 51-110, Chunk 3: lines 101-160 ...
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        chunks.append(CodeChunk(
            content="".join(lines[start:end]),
            file_path=file_path,
            repo=repo_name,
            language=lang,
            start_line=start + 1,  # convert 0-based index to 1-based line number
            end_line=end,
        ))
        start += CHUNK_SIZE - OVERLAP  # advance by stride (50 lines)

    return chunks


def chunk_repo(repo_path: str) -> list[CodeChunk]:
    """
    Walk an entire repository and chunk all supported source files.

    Skips directories in SKIP_DIRS (node_modules, .git, build output, etc.)
    Returns a flat list of all CodeChunk objects across all files.
    """
    all_chunks = []
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to prevent os.walk from descending into skipped dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(root, fname)
                all_chunks.extend(chunk_file(fpath, repo_path))
    return all_chunks
