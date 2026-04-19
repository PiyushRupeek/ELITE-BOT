import json
import logging
import os
from rag.retriever import SearchResult
from llm.llm_router import LLMRouter
from rag.retriever import Retriever
from tools.grafana_tool import GrafanaTool
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

_ENGINEERING_KEYWORDS = {
    "debug", "implement", "explain", "review", "refactor", "deploy", "build",
    "write", "fix", "create", "add", "update", "change", "test", "check",
    "code", "function", "class", "method", "module", "interface", "endpoint",
    "api", "service", "controller", "repository", "model", "schema", "config",
    "middleware", "handler", "hook", "util", "helper", "type", "enum",
    "typescript", "javascript", "java", "python", "spring", "nestjs", "nest",
    "node", "react", "sql", "database", "db", "redis", "kafka", "docker",
    "git", "env", "yaml", "json", "rest", "graphql", "grpc",
    "error", "exception", "bug", "crash", "fail", "broken", "issue",
    "500", "404", "null", "undefined", "timeout", "memory", "leak",
    "log", "trace", "metric", "grafana", "loki", "k8s", "ci", "cd",
    "pipeline", "release", "branch", "pr", "merge",
    "bazaar", "lead", "loan", "image", "rupeek", "customer",
    "how does", "how do", "what is", "where is", "which file", "show me",
    "why is", "why does", "when does", "does it", "can you",
}

OUT_OF_SCOPE = (
    ":no_entry: I'm an engineering assistant for the Rupeek codebase.\n"
    "I can help with *debugging*, *explaining code*, *implementing features*, and *code reviews*.\n"
    "Type `help` to see what I can do."
)


def is_relevant(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _ENGINEERING_KEYWORDS)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS = {
    "debug": [
        "error", "exception", "failing", "failed", "broken", "crash", "500",
        "404", "why is", "not working", "traceback", "stacktrace", "bug",
        "issue", "problem", "wrong", "null pointer", "undefined", "timeout",
    ],
    "explain": [
        "explain", "how does", "what is", "what does", "show me", "walk me",
        "understand", "describe", "overview", "how do", "what are", "tell me",
        "give me an idea", "how is",
    ],
    "implement": [
        "write", "implement", "add", "create", "change", "update", "refactor",
        "generate", "build", "modify", "fix", "make", "code", "develop",
        "new endpoint", "new api", "new function", "new class",
    ],
    "review": [
        "review", "check", "is this correct", "any issues", "code review",
        "look at this", "what do you think", "feedback", "improve",
    ],
}


def _detect_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "You are a search query optimizer. Rewrite user questions into short, "
    "keyword-rich search queries for semantic code search. "
    "Output ONLY the rewritten query — no explanation, no quotes, no punctuation at the end."
)

_REWRITE_EXAMPLES = """Examples:
User: why is checkout returning 500?
Search: checkout payment processing error exception handler

User: explain how image upload works
Search: image upload file storage S3 multipart handler service

User: write a new endpoint to get loan status
Search: loan status REST endpoint controller service method

User: {query}
Search:"""


# ---------------------------------------------------------------------------
# System prompts per intent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "debug": """You are an expert backend debugging assistant for the Rupeek engineering team.
You have access to relevant source code snippets and recent logs.

ALWAYS respond using EXACTLY this structure:

### 🔍 Root Cause
[1-2 sentences identifying the exact cause]

### 📍 Evidence
[Reference specific file:line and/or log lines that prove the root cause]

### 🛠 Fix
```
// File: path/to/file
// exact code change here
```

### ⚠️ Side Effects
[Any related issues or things to watch out for after applying the fix]

Be concise. Use actual file paths from the code snippets provided.""",

    "explain": """You are a senior engineer helping a teammate understand the Rupeek codebase.
You have access to relevant source code snippets.

ALWAYS respond using EXACTLY this structure:

### 📌 Summary
[1-2 sentences: what it does and why it exists]

### 🔄 Flow
[Numbered steps describing the execution path]

### 🗂 Key Files
[Bullet list: `file/path.ts` — what each file is responsible for]

### 💡 Notes
[Any gotchas, design decisions, or things worth knowing]

Reference actual file paths from the code snippets provided.""",

    "implement": """You are a senior engineer helping implement code changes in the Rupeek codebase.
You have access to relevant source code snippets showing existing patterns and conventions.

ALWAYS respond using EXACTLY this structure:

### 🎯 Approach
[Brief description of the implementation strategy]

### 💻 Code
```language
// File: path/to/file.ts
// exact implementation here, following existing patterns
```

### 📋 Steps to Apply
[Numbered list of steps to apply the change]

### 🔗 Dependencies
[Any new imports, packages, or configs needed]

Follow the exact coding style visible in the provided snippets.""",

    "review": """You are a senior code reviewer for the Rupeek engineering team.
You have access to relevant source code snippets for context.

ALWAYS respond using EXACTLY this structure:

### 🔴 Critical
[Must fix before merge — security issues, bugs, data loss risks]

### 🟡 Warnings
[Should fix — performance issues, bad patterns, maintainability]

### 🟢 Suggestions
[Nice to have — style, readability, minor improvements]

### ✅ What's Good
[Acknowledge what's done well]

Reference specific line numbers or patterns where applicable.""",

    "general": """You are a knowledgeable dev assistant for the Rupeek engineering team.
You have access to relevant source code snippets from the codebase.

ALWAYS respond using EXACTLY this structure:

### 💬 Answer
[Direct, concise answer to the question]

### 📂 Relevant Files
[Bullet list: `file/path.ts` — brief description of relevance]

### 🔎 Where to Look Next
[Any related areas or files worth exploring]

If you're unsure, say so — don't hallucinate.""",
}

# ---------------------------------------------------------------------------
# Service extraction (for Grafana log fetching)
# ---------------------------------------------------------------------------

_SERVICE_KEYWORDS = [
    "lead", "lead-service", "bazaar", "rupeek-bazaar",
    "image", "image-service", "quick", "rupeek-quick",
    "customer", "customer-landing",
]


def _extract_service(query: str) -> str | None:
    q = query.lower()
    for kw in _SERVICE_KEYWORDS:
        if kw in q:
            return kw
    return None


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

HELP_TEXT = """:wave: *Rupeek Dev Assistant* — here's what I can do:

*Debug* — find root causes and fixes
> `why is bazaar checkout returning 500?`
> `lead-service is throwing NullPointerException on startup`

*Explain* — understand code and flows
> `explain how the payment flow works in bazaar`
> `how does image upload work?`

*Implement* — write or suggest code changes
> `write a NestJS endpoint to get all active loans`
> `add input validation to the apply endpoint`

*Review* — get feedback on code
> `review this middleware: [paste code]`
> `any issues with this service class?`

*General* — ask anything about the codebase
> `where is S3 upload handled?`
> `which service owns loan creation?`

I detect your intent automatically — just ask naturally.
Reply in the same thread to continue the conversation with full context."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DevAgent:
    def __init__(self):
        self.ollama = LLMRouter()
        self.retriever = Retriever()
        self.grafana = GrafanaTool()
        self._history: dict[str, list[dict]] = self._load_history()

    def _load_history(self) -> dict[str, list[dict]]:
        try:
            with open(config.HISTORY_PATH, "r") as f:
                data = json.load(f)
            logger.info("Loaded thread history: %d threads", len(data))
            return data
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning("Could not load thread history (%s) — starting fresh", e)
            return {}

    def _rewrite_query(self, query: str) -> str:
        try:
            prompt = _REWRITE_EXAMPLES.replace("{query}", query)
            result = self.ollama.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=_REWRITE_SYSTEM,
            )
            rewritten = result.strip().splitlines()[0].strip()
            if rewritten and len(rewritten) > 3:
                logger.info("Query rewritten: '%s' → '%s'", query[:60], rewritten[:80])
                return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed: %s — using original", e)
        return query

    def _n_results(self, query: str, intent: str) -> int:
        word_count = len(query.split())
        if intent in ("implement", "debug") or word_count > 15:
            return 7
        if word_count < 8:
            return 3
        return 5

    def _build_chunk_context(self, chunk: SearchResult) -> str:
        header = (
            f"**File:** `{chunk.file_path}` "
            f"(lines {chunk.start_line}-{chunk.end_line}, repo: {chunk.repo})"
        )
        prefix = ""
        if chunk.start_line > 15:
            try:
                with open(chunk.file_path, "r", errors="ignore") as f:
                    top_lines = "".join(f.readlines()[:15])
                prefix = (
                    f"// --- file header (imports / class declaration) ---\n"
                    f"{top_lines}\n"
                    f"// --- chunk starts at line {chunk.start_line} ---\n"
                )
            except OSError:
                pass
        return f"{header}\n```{chunk.language}\n{prefix}{chunk.content.strip()}\n```"

    def run(self, query: str, thread_id: str | None = None) -> str:
        intent = _detect_intent(query)
        logger.info("Intent: %s | Query: %s", intent, query[:80])

        search_query = self._rewrite_query(query)

        n = self._n_results(query, intent)
        logger.info("Searching codebase (n=%d)...", n)
        raw_results = self.retriever.search(search_query, n_results=n)

        code_results = [r for r in raw_results if r.score >= 0.3] or raw_results[:1]
        logger.info("Found %d relevant chunks (filtered from %d)", len(code_results), len(raw_results))

        logs_section = ""
        if intent == "debug" and config.GRAFANA_ENABLED:
            service = _extract_service(query)
            if service:
                logger.info("Fetching Grafana logs for: %s", service)
                raw_logs = self.grafana.fetch_logs(service, duration_minutes=15, limit=40)
                if raw_logs:
                    logs_section = f"\n\n## Recent Logs ({service}, last 15 min)\n```\n{raw_logs}\n```"

        code_section = "\n\n---\n\n".join(
            self._build_chunk_context(r) for r in code_results
        )
        context_block = f"## Relevant Code\n{code_section}{logs_section}"

        history = self._get_history(thread_id)
        messages = history + [
            {"role": "user", "content": f"{context_block}\n\n## Question\n{query}"}
        ]

        system_prompt = _SYSTEM_PROMPTS[intent]
        response = self.ollama.chat(messages=messages, system_prompt=system_prompt)
        logger.info("LLM responded (%d chars)", len(response))

        self._save_history(thread_id, query, response)

        refs = "\n".join(
            f"• `{r.file_path}:{r.start_line}` ({r.repo})"
            for r in code_results
        )
        intent_label = {
            "debug": "🐛", "explain": "📖", "implement": "⚙️",
            "review": "🔍", "general": "💬",
        }
        return f"{intent_label.get(intent, '')} _{intent.capitalize()} mode_\n\n{response}\n\n---\n_Sources:_\n{refs}"

    def _get_history(self, thread_id: str | None) -> list[dict]:
        if not thread_id:
            return []
        return self._history.get(thread_id, [])[-6:]

    def _save_history(self, thread_id: str | None, query: str, response: str):
        if not thread_id:
            return
        if thread_id not in self._history:
            self._history[thread_id] = []
        self._history[thread_id].append({"role": "user", "content": query})
        self._history[thread_id].append({"role": "assistant", "content": response})
        if len(self._history[thread_id]) > 20:
            self._history[thread_id] = self._history[thread_id][-20:]
        try:
            tmp = config.HISTORY_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._history, f)
            os.replace(tmp, config.HISTORY_PATH)
        except Exception as e:
            logger.warning("Could not save thread history: %s", e)
