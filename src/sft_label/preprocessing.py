"""
Preprocessing Module

Extracts structural signals from SFT conversations before LLM labeling.
These signals are embedded into the prompt to help the LLM focus on semantic
judgment rather than basic information extraction.

Supports two input formats:
  - ShareGPT: {"conversations": [{"from": "human/gpt/tool", "value": "..."}]}
  - Pangu:    {"data": [{"role": "user/assistant/tool", "content": "..."}]}

Signals extracted:
  - Language detection from code blocks
  - Tool role detection (agentic signals)
  - Code block count and content analysis
  - Turn count and conversation structure
  - Keyword-based hints
  - Token estimation
  - Last turn extraction
"""

import re
import json

from sft_label.config import (
    MAX_CONVERSATION_CHARS, TRUNCATION_HEAD_RATIO,
    TRUNCATION_LAST_RESPONSE_RATIO, TRUNCATION_PER_TURN_RATIO,
)


# ─────────────────────────────────────────────────────────
# Format normalization
# ─────────────────────────────────────────────────────────

# CoT / thinking block patterns
_PANGU_COT_RE = re.compile(r'\[unused16\].*?\[unused17\]', re.DOTALL)
_PANGU_TOKENS_RE = re.compile(r'\[unused(?:9|10|11|12|13|14|15|16|17)\]')
_NO_THINK_RE = re.compile(r'\s*/no_think')
_SHAREGPT_COT_RE = re.compile(r'<(?:think|thinking)>.*?</(?:think|thinking)>', re.DOTALL)

PANGU_ROLE_MAP = {"user": "human", "assistant": "gpt", "tool": "tool"}


def detect_format(sample):
    """Detect whether sample is ShareGPT or Pangu format."""
    if "conversations" in sample:
        return "sharegpt"
    if "data" in sample:
        return "pangu"
    return "unknown"


def strip_cot(text):
    """Remove CoT/thinking blocks from text. Handles both Pangu and ShareGPT patterns."""
    # Pangu: [unused16]thinking content[unused17] → remove entire block
    text = _PANGU_COT_RE.sub('', text)
    # ShareGPT: <think>...</think> or <thinking>...</thinking>
    text = _SHAREGPT_COT_RE.sub('', text)
    return text.strip()


def _strip_pangu_tokens(text):
    """Remove Pangu training tokens and CoT content."""
    text = strip_cot(text)
    text = _NO_THINK_RE.sub('', text)
    text = _PANGU_TOKENS_RE.sub('', text)
    return text.strip()


def slice_multiturn(conversations):
    """Slice multi-turn conversations into training-aligned samples.

    Each assistant reply becomes one sample, with full preceding context.
    Mirrors how SFT training expands multi-turn into pyramid-shaped samples:
      [H1] → A1
      [H1, A1, H2] → A2
      [H1, A1, H2, A2, H3] → A3

    Single-turn (1 human + 1 gpt) returns as-is in a list.
    """
    # Find all assistant reply indices
    reply_indices = [i for i, t in enumerate(conversations) if t["from"] == "gpt"]

    if len(reply_indices) <= 1:
        return [conversations]

    slices = []
    for idx in reply_indices:
        # Context = everything up to and including this assistant reply
        slices.append(conversations[:idx + 1])

    return slices


def normalize_pangu(sample):
    """Convert Pangu format sample to internal ShareGPT format.

    - Strips training tokens ([unused*], /no_think)
    - Maps roles (user→human, assistant→gpt)
    - Detects pseudo multi-turn (already a single training sample)
    - Preserves meta_prompt and tools in metadata
    """
    data = sample.get("data", [])
    conversations = []
    is_pseudo_multiturn = False

    for turn in data:
        role = PANGU_ROLE_MAP.get(turn.get("role", ""), turn.get("role", ""))
        content = turn.get("content", "")

        # Detect pseudo multi-turn (don't expand, just strip tokens)
        if role == "human" and "[unused10]" in content:
            is_pseudo_multiturn = True

        conversations.append({
            "from": role,
            "value": _strip_pangu_tokens(content),
        })

    # Build normalized sample
    normalized = {
        "id": sample.get("id", ""),
        "conversations": conversations,
        "metadata": sample.get("metadata", {}),
    }

    # Preserve Pangu-specific info in metadata
    pangu_meta = {}
    if sample.get("meta_prompt"):
        pangu_meta["system_prompt"] = sample["meta_prompt"]
    if sample.get("tools"):
        pangu_meta["tool_definitions"] = sample["tools"]
    pangu_meta["is_pseudo_multiturn"] = is_pseudo_multiturn
    pangu_meta["original_format"] = "pangu"
    normalized["metadata"] = {**normalized["metadata"], **pangu_meta}

    return normalized


def normalize_and_slice(sample):
    """Auto-detect format, normalize, and slice multi-turn into training samples.

    Returns a list of samples. Single-turn and pseudo multi-turn return [1 sample].
    True multi-turn returns [N samples], one per assistant reply.
    Each slice has id suffixed with turn number (e.g. "id_t1", "id_t2").
    """
    fmt = detect_format(sample)
    if fmt == "pangu":
        normalized = normalize_pangu(sample)
    else:
        normalized = dict(sample)
        # Strip CoT from ShareGPT conversations
        if "conversations" in normalized:
            for turn in normalized["conversations"]:
                if turn.get("from") == "gpt" and turn.get("value"):
                    turn["value"] = strip_cot(turn["value"])

    conversations = normalized.get("conversations", [])
    is_pseudo = normalized.get("metadata", {}).get("is_pseudo_multiturn", False)

    # Pseudo multi-turn: already one training sample, don't slice
    if is_pseudo:
        return [normalized]

    slices = slice_multiturn(conversations)

    if len(slices) == 1:
        return [normalized]

    # Build one sample per slice
    results = []
    base_id = normalized.get("id", "")
    for i, conv_slice in enumerate(slices):
        s = {
            "id": f"{base_id}_t{i+1}",
            "conversations": conv_slice,
            "metadata": {
                **normalized.get("metadata", {}),
                "source_id": base_id,
                "turn_index": i + 1,
                "total_turns": len(slices),
            },
        }
        results.append(s)

    return results


# Keep backward-compatible alias
def normalize_sample(sample):
    """Normalize without slicing (for tools that don't need multi-turn expansion)."""
    fmt = detect_format(sample)
    if fmt == "pangu":
        return normalize_pangu(sample)
    return sample


# ─────────────────────────────────────────────────────────
# Conversation truncation for labeling
# ─────────────────────────────────────────────────────────

_TRUNCATION_MARKER = "\n\n[... content truncated for labeling ...]\n\n"


def _truncate_text(text, max_chars, keep_head_ratio=0.3):
    """Truncate a single text block, keeping head + tail with a marker in between."""
    if len(text) <= max_chars:
        return text
    marker_len = len(_TRUNCATION_MARKER)
    head_chars = int(max_chars * keep_head_ratio)
    tail_chars = max_chars - head_chars - marker_len
    if tail_chars < 200:
        tail_chars = 200
        head_chars = max_chars - tail_chars - marker_len
    if head_chars < 100:
        head_chars = 100
    return text[:head_chars] + _TRUNCATION_MARKER + text[-tail_chars:]


def truncate_conversations_for_labeling(conversations, max_total_chars=None,
                                        head_ratio=None, last_response_ratio=None,
                                        per_turn_ratio=None):
    """Truncate conversations to fit within model context for labeling.

    Budget allocation (configurable in config.py):
    - First human turn (task context): head_ratio of budget
    - Last gpt turn (labeling target): last_response_ratio of budget
    - Remaining turns: share the rest, filled from the end backward

    For ≤2 turns: split budget proportionally between query and response,
    giving the response slightly more room.

    Returns (truncated_conversations, was_truncated).
    """
    if max_total_chars is None:
        max_total_chars = MAX_CONVERSATION_CHARS
    if head_ratio is None:
        head_ratio = TRUNCATION_HEAD_RATIO
    if last_response_ratio is None:
        last_response_ratio = TRUNCATION_LAST_RESPONSE_RATIO
    if per_turn_ratio is None:
        per_turn_ratio = TRUNCATION_PER_TURN_RATIO

    total_chars = sum(len(t.get("value", "")) for t in conversations)
    if total_chars <= max_total_chars:
        return conversations, False

    n = len(conversations)

    # --- Single turn or two turns (query + response) ---
    if n <= 2:
        # Give response 60% of budget, query 40%
        if n == 1:
            val = _truncate_text(conversations[0].get("value", ""), max_total_chars)
            return [{**conversations[0], "value": val}], True
        query_budget = int(max_total_chars * 0.4)
        response_budget = max_total_chars - query_budget
        result = []
        for i, t in enumerate(conversations):
            budget = response_budget if t["from"] == "gpt" else query_budget
            val = t.get("value", "")
            if len(val) > budget:
                val = _truncate_text(val, budget)
            result.append({**t, "value": val})
        return result, True

    # --- Multi-turn ---
    # Identify key turns
    first_human_idx = next((i for i, t in enumerate(conversations) if t["from"] == "human"), 0)
    last_gpt_idx = next((i for i in range(n - 1, -1, -1) if conversations[i]["from"] == "gpt"), n - 1)

    # Reserve budgets for key turns
    first_budget = int(max_total_chars * head_ratio)
    last_resp_budget = int(max_total_chars * last_response_ratio)
    middle_budget = max_total_chars - first_budget - last_resp_budget

    # 1. Truncate first human turn
    first_turn = dict(conversations[first_human_idx])
    first_val = first_turn.get("value", "")
    if len(first_val) > first_budget:
        first_val = _truncate_text(first_val, first_budget)
    first_turn["value"] = first_val

    # 2. Truncate last gpt turn
    last_turn = dict(conversations[last_gpt_idx])
    last_val = last_turn.get("value", "")
    if len(last_val) > last_resp_budget:
        last_val = _truncate_text(last_val, last_resp_budget)
    last_turn["value"] = last_val

    # 3. Fill middle from the end backward (excluding first_human and last_gpt)
    #    This preserves the most recent context leading up to the last response.
    middle_indices = [i for i in range(n) if i != first_human_idx and i != last_gpt_idx]
    per_turn_cap = int(max_total_chars * per_turn_ratio)

    tail_turns = []  # (original_index, turn_dict)
    tail_chars = 0
    for i in reversed(middle_indices):
        val = conversations[i].get("value", "")
        # Cap individual turn
        if len(val) > per_turn_cap:
            val = _truncate_text(val, per_turn_cap)
        if tail_chars + len(val) > middle_budget:
            # Try to fit a truncated version
            remaining = middle_budget - tail_chars
            if remaining > 500:
                val = _truncate_text(conversations[i].get("value", ""), remaining)
                tail_turns.append((i, {**conversations[i], "value": val}))
            break
        tail_turns.append((i, {**conversations[i], "value": val}))
        tail_chars += len(val)

    tail_turns.reverse()

    # 4. Assemble: first_human + [omission marker] + kept middle turns + last_gpt
    result = [first_turn]

    kept_indices = {i for i, _ in tail_turns}
    # Count omitted turns between first_human and the earliest kept middle turn
    omitted = len(middle_indices) - len(tail_turns)
    if omitted > 0:
        result.append({"from": "system", "value": f"[... {omitted} middle turns omitted for labeling ...]"})

    for _, turn in tail_turns:
        result.append(turn)

    # Ensure last_gpt is at the end (it might already be in tail_turns if last_gpt_idx was in middle)
    if not tail_turns or tail_turns[-1][0] != last_gpt_idx:
        result.append(last_turn)

    return result, True


# ─────────────────────────────────────────────────────────
# Language detection patterns
# ─────────────────────────────────────────────────────────

# Map of code fence language identifiers → taxonomy language tag IDs
FENCE_LANG_MAP = {
    "python": "python", "py": "python", "python3": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript", "tsx": "typescript", "jsx": "javascript",
    "java": "java",
    "go": "go", "golang": "go",
    "rust": "rust", "rs": "rust",
    "c": "c", "h": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp", "hpp": "cpp",
    "csharp": "csharp", "cs": "csharp", "c#": "csharp",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin", "kt": "kotlin",
    "scala": "scala",
    "sql": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql", "sqlite": "sql",
    "html": "html",
    "css": "css", "scss": "css", "sass": "css", "less": "css",
    "shell": "shell", "bash": "shell", "sh": "shell", "zsh": "shell",
    "dockerfile": "dockerfile", "docker": "dockerfile",
    "yaml": "yaml", "yml": "yaml",
    "json": "json", "jsonc": "json",
    "toml": "toml",
    "xml": "xml",
    "markdown": "markdown", "md": "markdown",
    "hcl": "hcl", "terraform": "hcl", "tf": "hcl",
    "lua": "lua",
    "r": "r",
    "dart": "dart",
    "elixir": "elixir", "ex": "elixir",
    "erlang": "erlang", "erl": "erlang",
    "haskell": "haskell", "hs": "haskell",
    "ocaml": "ocaml", "ml": "ocaml",
    "perl": "perl", "pl": "perl",
    "solidity": "solidity", "sol": "solidity",
    "verilog": "verilog", "v": "verilog",
    "zig": "zig",
    "makefile": "makefile", "make": "makefile",
    "cmake": "cmake",
    "nginx": "nginx-config",
    "ini": "ini",
    "properties": "properties",
    "latex": "latex", "tex": "latex",
    "matlab": "matlab",
    "julia": "julia", "jl": "julia",
    "clojure": "clojure", "clj": "clojure",
    "fsharp": "fsharp", "fs": "fsharp", "f#": "fsharp",
    "powershell": "powershell", "ps1": "powershell",
    "prolog": "prolog",
    "scheme": "scheme",
    "lisp": "lisp",
    "assembly": "assembly", "asm": "assembly", "nasm": "assembly",
    "fortran": "fortran", "f90": "fortran",
    "cobol": "cobol",
    "groovy": "groovy",
    "ada": "ada",
    "nim": "nim",
    "crystal": "crystal",
    "vyper": "vyper",
}

# Framework → language inference
FRAMEWORK_LANG_MAP = {
    "react": "typescript", "next.js": "typescript", "nextjs": "typescript",
    "vue": "typescript", "nuxt": "typescript", "angular": "typescript",
    "svelte": "typescript", "sveltekit": "typescript",
    "django": "python", "flask": "python", "fastapi": "python", "tornado": "python",
    "spring": "java", "spring boot": "java", "springboot": "java",
    "express": "javascript", "express.js": "javascript", "koa": "javascript",
    "rails": "ruby", "ruby on rails": "ruby", "sinatra": "ruby",
    "laravel": "php", "symfony": "php",
    "gin": "go", "echo": "go", "fiber": "go",
    "actix": "rust", "axum": "rust", "tokio": "rust", "rocket": "rust",
    "swiftui": "swift", "uikit": "swift",
    "jetpack compose": "kotlin", "ktor": "kotlin",
    "flutter": "dart",
    "pytorch": "python", "tensorflow": "python", "keras": "python",
    "pandas": "python", "numpy": "python", "scikit-learn": "python",
    "tailwind": "css", "bootstrap": "css",
    "terraform": "hcl", "ansible": "yaml", "helm": "yaml",
    "docker compose": "yaml", "docker-compose": "yaml",
    "prisma": "typescript", "drizzle": "typescript",
    "sqlalchemy": "python", "alembic": "python",
    "playwright": "typescript", "cypress": "typescript", "selenium": "python",
    "pygame": "python",
}

# Tool name → agentic tag mapping
TOOL_NAME_MAP = {
    # File operations
    "read_file": "file-operations", "write_file": "file-operations",
    "edit_file": "file-operations", "read": "file-operations",
    "write": "file-operations", "edit": "file-operations",
    "glob": "file-operations", "grep": "file-operations",
    "search_files": "file-operations", "list_files": "file-operations",
    "cat": "file-operations", "ls": "file-operations",

    # Shell / bash
    "bash": "bash-execution", "shell": "bash-execution",
    "terminal": "bash-execution", "execute_command": "bash-execution",
    "run_command": "bash-execution",

    # Code execution
    "python": "code-execution", "execute_code": "code-execution",
    "run_code": "code-execution", "jupyter": "code-execution",
    "repl": "code-execution",

    # Git
    "git": "git-operations",

    # Web
    "web_search": "web-search", "search": "web-search",
    "browser": "web-search", "fetch_url": "web-search",

    # Build
    "build": "build-execution", "compile": "build-execution",
    "make": "build-execution",

    # Test
    "test": "test-running", "run_tests": "test-running",
    "pytest": "test-running",

    # DB
    "sql": "database-query", "query": "database-query",

    # Package management
    "install": "dependency-installation", "pip": "dependency-installation",
    "npm": "dependency-installation", "yarn": "dependency-installation",
}


def detect_code_fence_languages(text):
    """Extract language tags from markdown code fences."""
    pattern = r'```(\w[\w+#.-]*)'
    matches = re.findall(pattern, text)
    languages = set()
    for match in matches:
        lang = match.lower().strip()
        if lang in FENCE_LANG_MAP:
            languages.add(FENCE_LANG_MAP[lang])
    return sorted(languages)


def detect_framework_languages(text):
    """Infer languages from framework/library mentions."""
    text_lower = text.lower()
    languages = set()
    for framework, lang in FRAMEWORK_LANG_MAP.items():
        if framework in text_lower:
            languages.add(lang)
    return sorted(languages)


def count_code_blocks(text):
    """Count number of code blocks in text."""
    return len(re.findall(r'```', text)) // 2


def extract_tool_signals(conversations):
    """Extract agentic signals from tool role messages."""
    tool_names = []
    agentic_tags = set()

    for turn in conversations:
        if turn.get("from") == "tool":
            value = turn.get("value", "")

            # Try to detect tool name from common patterns
            # Pattern: "$ command ..." (shell)
            if value.strip().startswith("$") or value.strip().startswith("#"):
                tool_names.append("bash")
                agentic_tags.add("bash-execution")

                # Check specific commands within bash
                cmd = value.strip().lstrip("$ ").split()[0] if value.strip().lstrip("$ ") else ""
                if cmd in ("cat", "ls", "find", "head", "tail", "grep"):
                    agentic_tags.add("file-operations")
                elif cmd in ("git",):
                    agentic_tags.add("git-operations")
                elif cmd in ("npm", "pip", "yarn", "pnpm", "cargo", "go"):
                    subargs = value.strip().lstrip("$ ").split()
                    if len(subargs) > 1 and subargs[1] in ("install", "add", "get"):
                        agentic_tags.add("dependency-installation")
                elif cmd in ("pytest", "jest", "go test", "cargo test"):
                    agentic_tags.add("test-running")
                elif cmd in ("docker", "make", "cargo build", "npm run build"):
                    agentic_tags.add("build-execution")
                elif cmd in ("python", "node", "ruby", "go run"):
                    agentic_tags.add("code-execution")

            # Pattern: structured tool call JSON-like
            if '"name"' in value or '"tool"' in value:
                for tool_key, tag in TOOL_NAME_MAP.items():
                    if tool_key in value.lower():
                        tool_names.append(tool_key)
                        agentic_tags.add(tag)

    return sorted(set(tool_names)), sorted(agentic_tags)


def detect_behavioral_patterns(conversations):
    """Detect agentic behavioral patterns from conversation structure."""
    patterns = set()
    turns = len(conversations)
    gpt_turns = [t for t in conversations if t.get("from") == "gpt"]
    tool_turns = [t for t in conversations if t.get("from") == "tool"]

    # Multi-file coordination: multiple file operations on different paths
    file_paths = set()
    for t in tool_turns:
        val = t.get("value", "")
        paths = re.findall(r'(?:cat|read|write|edit)\s+(\S+\.\w+)', val)
        paths += re.findall(r'(?:>\s*)(\S+\.\w+)', val)
        file_paths.update(paths)
    if len(file_paths) >= 2:
        patterns.add("multi-file-coordination")

    # Iterative refinement: multiple tool calls suggesting try-fix-retry
    if len(tool_turns) >= 3:
        patterns.add("iterative-refinement")

    # Planning: first gpt turn contains plan-like language
    if gpt_turns:
        first_gpt = gpt_turns[0].get("value", "").lower()
        plan_signals = ["let me", "first", "step 1", "plan", "i'll start by",
                        "先", "首先", "步骤", "计划", "我来"]
        if any(s in first_gpt for s in plan_signals):
            patterns.add("planning")

    # Multi-step reasoning: long gpt response with sequential logic
    for t in gpt_turns:
        val = t.get("value", "")
        if len(val) > 500:
            reasoning_signals = ["because", "therefore", "this means", "so we need",
                                 "因为", "所以", "这意味着", "因此", "根本原因"]
            if sum(1 for s in reasoning_signals if s in val.lower()) >= 2:
                patterns.add("multi-step-reasoning")
                break

    # Error recovery: error message followed by a fix attempt
    for i, t in enumerate(conversations):
        val = t.get("value", "").lower()
        if any(w in val for w in ["error", "failed", "exception", "traceback", "panic", "报错"]):
            if i + 1 < len(conversations):
                patterns.add("error-recovery")
                break

    return sorted(patterns)


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for mixed en/zh content."""
    return len(text) // 4


def extract_last_turn(conversations):
    """Extract the last user query and last assistant response."""
    last_human = ""
    last_gpt = ""
    for turn in reversed(conversations):
        if turn.get("from") == "human" and not last_human:
            last_human = turn.get("value", "")
        if turn.get("from") == "gpt" and not last_gpt:
            last_gpt = turn.get("value", "")
        if last_human and last_gpt:
            break
    return last_human, last_gpt


def detect_keywords(text):
    """Detect domain/topic keywords for context hints."""
    text_lower = text.lower()
    hits = []
    keyword_groups = {
        "web": ["react", "vue", "angular", "next.js", "express", "django", "flask",
                "html", "css", "api", "rest", "graphql", "frontend", "backend"],
        "devops": ["docker", "kubernetes", "k8s", "terraform", "ansible", "ci/cd",
                   "github actions", "jenkins", "helm", "nginx", "prometheus", "grafana"],
        "database": ["sql", "postgresql", "mysql", "mongodb", "redis", "sqlite",
                     "database", "query", "index", "migration", "schema"],
        "ml": ["pytorch", "tensorflow", "model", "training", "neural", "dataset",
               "classification", "regression", "epoch", "loss", "optimizer"],
        "security": ["auth", "oauth", "jwt", "xss", "csrf", "injection", "encryption",
                     "password", "token", "vulnerability", "security"],
        "mobile": ["ios", "android", "swift", "kotlin", "flutter", "react native"],
        "systems": ["kernel", "memory", "allocator", "lock-free", "atomic", "assembly",
                    "embedded", "firmware", "rtos"],
    }
    for group, keywords in keyword_groups.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            hits.append((group, matches))
    return hits


def generate_sparse_schedule(n, full_label_count=10, gap_multiplier=1.2, min_gap=2, threshold=12):
    """Front-dense, back-sparse sampling schedule for pyramid slices.

    n <= threshold: label all (return [0..n-1])
    n > threshold: first full_label_count all labeled, then gap grows ~gap_multiplier, last always labeled
    """
    if n <= threshold:
        return list(range(n))

    schedule = list(range(full_label_count))  # first N always labeled
    pos = full_label_count
    gap = float(min_gap)
    while pos < n - 1:
        schedule.append(pos)
        gap *= gap_multiplier
        pos = min(pos + max(int(gap), min_gap), n - 1)

    # Always include the last slice
    if schedule[-1] != n - 1:
        schedule.append(n - 1)

    return schedule


def apply_sparse_sampling(samples, full_label_count=10, gap_multiplier=1.2, min_gap=2, threshold=12):
    """Apply sparse sampling to multi-turn pyramid slices.

    Single-turn samples (no source_id or total_turns=1) are always labeled.
    Multi-turn slices are grouped by source_id, sorted by turn_index,
    and sampled via generate_sparse_schedule.

    Returns:
        label_indices: set[int] — global indices that need actual LLM labeling
        inherit_map: dict[int, int] — {unlabeled_idx: source_idx} for inheritance

    Inheritance strategy: look forward (inherit from next labeled slice).
    Reason: slice_i is a prefix of slice_j (j>i), so j's labels are more complete.
    Last unlabeled slice with no successor inherits backward.
    """
    label_indices = set()
    inherit_map = {}

    # Group by source_id; track global index
    groups = {}  # source_id -> [(global_idx, turn_index)]
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        source_id = meta.get("source_id")
        total_turns = meta.get("total_turns", 1)

        if not source_id or total_turns <= 1:
            # Single-turn: always label
            label_indices.add(i)
        else:
            turn_idx = meta.get("turn_index", 1)
            groups.setdefault(source_id, []).append((i, turn_idx))

    # For each multi-turn group, apply sparse schedule
    for source_id, members in groups.items():
        # Sort by turn_index
        members.sort(key=lambda x: x[1])
        n = len(members)
        schedule = generate_sparse_schedule(n, full_label_count, gap_multiplier, min_gap, threshold)
        schedule_set = set(schedule)

        # Mark labeled indices
        labeled_positions = []
        for pos_in_group, (global_idx, _) in enumerate(members):
            if pos_in_group in schedule_set:
                label_indices.add(global_idx)
                labeled_positions.append(pos_in_group)

        # Build inherit_map: unlabeled → nearest labeled (forward first, backward fallback)
        for pos_in_group, (global_idx, _) in enumerate(members):
            if pos_in_group in schedule_set:
                continue
            # Find next labeled position (forward)
            source_pos = None
            for lp in labeled_positions:
                if lp > pos_in_group:
                    source_pos = lp
                    break
            # Fallback: previous labeled position (backward)
            if source_pos is None:
                for lp in reversed(labeled_positions):
                    if lp < pos_in_group:
                        source_pos = lp
                        break
            if source_pos is not None:
                inherit_map[global_idx] = members[source_pos][0]

    return label_indices, inherit_map


def preprocess(sample):
    """
    Full preprocessing pipeline for one SFT sample.

    Args:
        sample: dict in ShareGPT or Pangu format (auto-detected)

    Returns:
        dict with all extracted signals
    """
    sample = normalize_sample(sample)
    conversations = sample.get("conversations", [])
    full_text = " ".join(t.get("value", "") for t in conversations)

    # Language detection
    fence_langs = detect_code_fence_languages(full_text)
    framework_langs = detect_framework_languages(full_text)
    all_detected_langs = sorted(set(fence_langs + framework_langs))

    # Tool / agentic detection
    has_tool_roles = any(t.get("from") == "tool" for t in conversations)
    tool_names, tool_agentic_tags = extract_tool_signals(conversations)
    behavioral_patterns = detect_behavioral_patterns(conversations)

    # Conversation structure
    total_turns = len(conversations)
    code_block_count = count_code_blocks(full_text)
    last_query, last_response = extract_last_turn(conversations)

    # Keywords
    keyword_hits = detect_keywords(full_text)

    # Token estimation
    est_tokens = estimate_tokens(full_text)

    signals = {
        "detected_languages": all_detected_langs,
        "fence_languages": fence_langs,
        "framework_languages": framework_langs,
        "has_tool_roles": has_tool_roles,
        "tool_names": tool_names,
        "tool_agentic_tags": tool_agentic_tags,
        "behavioral_patterns": behavioral_patterns,
        "total_turns": total_turns,
        "code_block_count": code_block_count,
        "est_tokens": est_tokens,
        "keyword_hits": keyword_hits,
        "last_query_preview": last_query[:200] if last_query else "",
        "last_response_length": len(last_response),
    }

    return signals


def format_signals_for_prompt(signals):
    """Format preprocessed signals as a string to embed in the LLM prompt."""
    lines = []
    if signals["detected_languages"]:
        lines.append(f"detected_languages: {signals['detected_languages']}")
    lines.append(f"has_tool_roles: {str(signals['has_tool_roles']).lower()}")
    if signals["tool_names"]:
        lines.append(f"tool_names: {signals['tool_names']}")
    if signals["tool_agentic_tags"]:
        lines.append(f"tool_agentic_tags_detected: {signals['tool_agentic_tags']}")
    if signals["behavioral_patterns"]:
        lines.append(f"behavioral_patterns_detected: {signals['behavioral_patterns']}")
    lines.append(f"code_block_count: {signals['code_block_count']}")
    lines.append(f"total_turns: {signals['total_turns']}")
    lines.append(f"est_tokens: {signals['est_tokens']}")
    if signals["keyword_hits"]:
        kw_str = "; ".join(f"{g}: {', '.join(kws)}" for g, kws in signals["keyword_hits"])
        lines.append(f"keyword_hints: {kw_str}")
    return "\n".join(lines)
