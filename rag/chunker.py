import os
from dataclasses import dataclass

SUPPORTED_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx",
    ".java", ".py", ".go",
    ".json", ".yaml", ".yml", ".md",
    ".sql", ".sh", ".env.example",
}

SKIP_DIRS = {
    "node_modules", ".git", "target", "build", "dist",
    ".idea", ".vscode", "__pycache__", ".gradle", ".mvn",
    "coverage", ".nyc_output", "out",
}

CHUNK_SIZE = 60   # lines per chunk
OVERLAP = 10      # overlap between chunks


@dataclass
class CodeChunk:
    content: str
    file_path: str
    repo: str
    language: str
    start_line: int
    end_line: int

    @property
    def id(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def _language(ext: str) -> str:
    return {
        ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript",
        ".java": "java", ".py": "python",
        ".go": "go", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml",
        ".md": "markdown", ".sql": "sql",
        ".sh": "bash",
    }.get(ext, "text")


def chunk_file(file_path: str, repo_root: str) -> list[CodeChunk]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return []

    if not lines:
        return []

    repo_name = os.path.basename(repo_root.rstrip("/"))
    lang = _language(ext)
    chunks = []

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

    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        chunks.append(CodeChunk(
            content="".join(lines[start:end]),
            file_path=file_path,
            repo=repo_name,
            language=lang,
            start_line=start + 1,
            end_line=end,
        ))
        start += CHUNK_SIZE - OVERLAP

    return chunks


def chunk_repo(repo_path: str) -> list[CodeChunk]:
    all_chunks = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(root, fname)
                all_chunks.extend(chunk_file(fpath, repo_path))
    return all_chunks
