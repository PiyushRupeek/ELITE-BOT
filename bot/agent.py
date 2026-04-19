"""
Multi-purpose dev assistant agent.

Intents (auto-detected from query):
  debug     → root cause analysis + log context
  explain   → code explanation, flows, architecture
  implement → write / suggest code changes
  review    → code review with severity ratings
  general   → general codebase Q&A

Response quality improvements:
  - Query rewriting before retrieval (better code chunk matching)
  - Adaptive n_results based on query complexity and intent
  - Import context injection (file header prepended to mid-file chunks)
  - Structured response templates per intent
  - Low-score chunk filtering
"""
import logging
from rag.retriever import SearchResult
from llm.llm_router import LLMRouter
from rag.retriever import Retriever
from tools.grafana_tool import GrafanaTool

logger = logging.getLogger(__name__)

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
# Query rewriting prompt
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
# Structured system prompts per intent
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
# Service keyword extraction (for Grafana log fetch)
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
        self._history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def _rewrite_query(self, query: str) -> str:
        """Rewrite user query into a keyword-rich search query for better retrieval."""
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

    # ------------------------------------------------------------------
    # Adaptive n_results
    # ------------------------------------------------------------------

    def _n_results(self, query: str, intent: str) -> int:
        word_count = len(query.split())
        if intent in ("implement", "debug") or word_count > 15:
            return 7
        if word_count < 8:
            return 3
        return 5

    # ------------------------------------------------------------------
    # Import context injection
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str, thread_id: str | None = None) -> str:
        intent = _detect_intent(query)
        logger.info("Intent: %s | Query: %s", intent, query[:80])

        # 1. Rewrite query for better retrieval
        search_query = self._rewrite_query(query)

        # 2. Retrieve relevant code chunks (adaptive count)
        n = self._n_results(query, intent)
        logger.info("Searching codebase (n=%d)...", n)
        raw_results = self.retriever.search(search_query, n_results=n)

        # Filter low-relevance chunks, keep at least 1
        MIN_SCORE = 0.3
        code_results = [r for r in raw_results if r.score >= MIN_SCORE] or raw_results[:1]
        logger.info("Found %d relevant chunks (filtered from %d)", len(code_results), len(raw_results))

        # 3. Fetch logs only for debug intent (and only if Grafana is enabled)
        logs_section = ""
        if intent == "debug":
            import config as _cfg
            if not _cfg.GRAFANA_ENABLED:
                logger.info("Grafana disabled — skipping log fetch (set GRAFANA_ENABLED=true in .env to enable)")
            else:
                service = _extract_service(query)
                if service:
                    logger.info("Fetching Grafana logs for: %s", service)
                    raw_logs = self.grafana.fetch_logs(service, duration_minutes=15, limit=40)
                    if raw_logs:
                        logs_section = f"\n\n## Recent Logs ({service}, last 15 min)\n```\n{raw_logs}\n```"

        # 4. Build code context with import headers injected
        code_section = "\n\n---\n\n".join(
            self._build_chunk_context(r) for r in code_results
        )

        context_block = f"## Relevant Code\n{code_section}{logs_section}"

        # 5. Build message list (history + current)
        history = self._get_history(thread_id)
        messages = history + [
            {"role": "user", "content": f"{context_block}\n\n## Question\n{query}"}
        ]

        # 6. Call Ollama with structured system prompt
        logger.info("Calling Ollama (%s)...", intent)
        system_prompt = _SYSTEM_PROMPTS[intent]
        response = self.ollama.chat(messages=messages, system_prompt=system_prompt)
        logger.info("Ollama responded (%d chars)", len(response))

        # 7. Save to history
        self._save_history(thread_id, query, response)

        # 8. Append source references
        refs = "\n".join(
            f"• `{r.file_path}:{r.start_line}` ({r.repo})"
            for r in code_results
        )
        intent_label = {
            "debug": "🐛", "explain": "📖", "implement": "⚙️",
            "review": "🔍", "general": "💬",
        }
        return f"{intent_label.get(intent, '')} _{intent.capitalize()} mode_\n\n{response}\n\n---\n_Sources:_\n{refs}"

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

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
