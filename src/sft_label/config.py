"""
Labeling Pipeline Configuration

All production settings extracted here for easy tuning.
"""

import os
from dataclasses import dataclass, field

# ─── LLM API ────────────────────────────────────────────
LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://localhost:4000/v1")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "")

# ─── Shared Defaults ────────────────────────────────────
MAX_RETRIES = 3
SAMPLE_MAX_RETRIES = 3         # sample-level retry on call failure
REQUEST_TIMEOUT = 60           # seconds per LLM call (first attempt)
REQUEST_TIMEOUT_ESCALATION = [60, 90, 120]  # per-attempt timeout escalation
DEFAULT_RPS_LIMIT = 80         # max LLM requests/sec (0 = unlimited)
DEFAULT_RPS_WARMUP = 20        # seconds to ramp from 1 rps to full rps (0 = no warmup)

# ═══════════════════════════════════════════════════════════
# Pass 1: Tag Labeling
# ═══════════════════════════════════════════════════════════

DEFAULT_LABELING_MODEL = "gpt-4o-mini"
DEFAULT_CONCURRENCY = 600
CONFIDENCE_THRESHOLD = 0.60

# ─── Conversation Truncation (Pass 1) ───────────────────
MAX_CONVERSATION_CHARS = 20000   # total budget (~5K tokens); aggressive for fast labeling
COMPACT_CONVERSATION_CHARS = 8000  # compact mode budget; keeps CALL2 payload < 25KB
TRUNCATION_HEAD_RATIO = 0.35     # fraction of budget for first human turn (task context)
TRUNCATION_LAST_RESPONSE_RATIO = 0.30  # fraction of budget for last gpt turn (labeling target)
TRUNCATION_PER_TURN_RATIO = 0.35 # max fraction of budget for any single turn

# ─── Directory Pipeline ────────────────────────────────
DIR_PIPELINE_WATERMARK = 2.0   # load next file when in-flight < concurrency * watermark
DIR_PIPELINE_MAX_FILES = 40    # max files loaded in memory simultaneously

# ─── Chunked JSONL Pipeline ──────────────────────────
CHUNK_SIZE = 5000              # raw JSONL lines per chunk
MAX_ACTIVE_CHUNKS = 3          # max chunks in memory simultaneously

# ─── Sparse Sampling (multi-turn slices) ──────────────
SPARSE_FULL_LABEL_COUNT = 8   # first N slices always labeled
SPARSE_GAP_MULTIPLIER = 1.3   # gap between labeled slices grows by this factor
SPARSE_MIN_GAP = 2            # minimum gap between labeled slices
SPARSE_MAX_GAP = 8            # maximum gap between labeled slices (bounds inheritance distance)
SPARSE_THRESHOLD = 12         # slices <= this: label all, no sparse sampling

# ─── Consistency Rules ──────────────────────────────────
CONSISTENCY_RULES = [
    ("intent == 'learn' and len(agentic) > 3",
     "Intent=learn but many agentic tags"),
    ("intent == 'build' and 'feature-implementation' not in task "
     "and 'configuration' not in task and 'schema-design' not in task "
     "and 'deployment' not in task and 'api-design' not in task "
     "and 'testing-task' not in task and 'documentation' not in task "
     "and 'logging' not in task and 'monitoring' not in task "
     "and len(task) > 0",
     "Intent=build but no build-relevant task"),
    ("difficulty == 'beginner' and len(concept) > 3",
     "Difficulty=beginner but many concepts"),
    ("len(constraint) > 0 and difficulty == 'beginner'",
     "Has constraints but difficulty=beginner"),
    ("intent == 'debug' and 'bug-fixing' not in task and len(task) > 0",
     "Intent=debug but no bug-fixing in task"),
    ("difficulty == 'expert' and len(concept) == 0",
     "Difficulty=expert but no concepts tagged"),
    ("'code-translation' in task and len(language) < 2",
     "Task=code-translation but fewer than 2 languages"),
    ("'ownership' in concept and 'rust' not in language",
     "Concept=ownership (Rust-specific) but Rust not in language"),
    ("context == 'snippet' and 'multi-file-coordination' in agentic",
     "Context=snippet contradicts agentic=multi-file-coordination"),
    ("intent == 'review' and 'code-review-task' not in task and len(task) > 0",
     "Intent=review but no code-review-task in task"),
    ("intent == 'modify' and 'code-refactoring' not in task "
     "and 'code-optimization' not in task and 'migration' not in task "
     "and 'code-translation' not in task "
     "and 'feature-implementation' not in task "
     "and 'bug-fixing' not in task "
     "and 'error-handling-task' not in task "
     "and 'configuration' not in task "
     "and 'documentation' not in task "
     "and 'testing-task' not in task "
     "and 'schema-design' not in task "
     "and 'dependency-management' not in task "
     "and len(task) > 0",
     "Intent=modify but no modify-relevant task"),
]

# ═══════════════════════════════════════════════════════════
# Pass 2: Value Scoring
# ═══════════════════════════════════════════════════════════

DEFAULT_SCORING_MODEL = "gpt-4o-mini"
DEFAULT_SCORING_CONCURRENCY = 500

VALUE_WEIGHTS = {
    "complexity": 0.25,
    "quality": 0.40,
    "reasoning": 0.20,
    "rarity": 0.15,
}
RARITY_WEIGHTS = {
    "intent": 0.15,
    "difficulty": 0.3,
    "context": 0.5,
    "language": 1.0,
    "domain": 1.5,
    "task": 1.0,
    "concept": 2.0,
    "agentic": 1.5,
    "constraint": 1.0,
}
RARITY_COMBO_ALPHA = 0.7            # weight for tag IDF vs combo IDF

# ─── COT-Preserving Truncation (Pass 2) ─────────────────
VALUE_TRUNCATION_BUDGET = 20000     # total chars for scoring truncation
COMPACT_VALUE_TRUNCATION_BUDGET = 14000  # compact mode; keeps scoring payload < 25KB
VALUE_TRUNCATION_INSTRUCTION_RATIO = 0.15
VALUE_TRUNCATION_COT_RATIO = 0.45
VALUE_TRUNCATION_RESPONSE_RATIO = 0.35
VALUE_TRUNCATION_META_RATIO = 0.05
VALUE_TRUNCATION_FRAGMENT_COUNT = 3  # middle fragments for COT sampling

KNOWN_FLAGS_POSITIVE = frozenset({
    "excellent-explanation", "clean-code", "creative-solution",
    "good-error-handling", "comprehensive-testing",
})
KNOWN_FLAGS_NEGATIVE = frozenset({
    "has-bug", "security-issue", "outdated-practice",
    "incomplete", "over-engineered", "incorrect-output", "poor-explanation",
    "hallucination",
})
KNOWN_FLAGS = KNOWN_FLAGS_POSITIVE | KNOWN_FLAGS_NEGATIVE

# ─── Selection Score (Post-scoring, no LLM) ──────────
SELECTION_INTRA_WEIGHT = 0.55          # weight for intra-class percentile ranking
SELECTION_QUALITY_WEIGHT = 0.20        # weight for absolute quality (pure_quality)
SELECTION_RARITY_WEIGHT = 0.25         # weight for global rarity
SELECTION_MIN_GROUP_SIZE = 5       # min samples per tag to compute intra-class percentile
SELECTION_SMOOTHING_PRIOR = 30     # Bayesian shrinkage prior: blend toward global at n < prior

# ─── Conversation-Level Aggregation (Post-scoring, no LLM) ──
CONV_CONFIDENCE_INHERITED = 0.7           # confidence weight for inherited slices
CONV_QUALITY_PENALTIES = {3: 0.5, 5: 0.8} # quality_floor < key → penalty multiplier
CONV_QUALITY_PENALTY_DEFAULT = 1.0        # penalty when floor >= max(keys)
CONV_FLAG_PENALTY_BASE = 0.95             # 0.95 ^ len(negative_flags)
CONV_AGENTIC_QUALITY_PERCENTILE = 0.1     # use p10 instead of min for agentic conversations

# ─── Rationale (exploratory, default off) ──────────────
ENABLE_RATIONALE = False                  # when True, prompt asks for rationale field (~30% more tokens)


# ═══════════════════════════════════════════════════════════
# Runtime-Overridable Config
# ═══════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Runtime-overridable pipeline configuration.

    Module-level constants above remain unchanged for CLI mode and
    direct imports from other modules (e.g. preprocessing.py).
    This dataclass lets library callers override any setting.
    """
    # Shared
    litellm_base: str = LITELLM_BASE
    litellm_key: str = LITELLM_KEY
    max_retries: int = MAX_RETRIES
    sample_max_retries: int = SAMPLE_MAX_RETRIES
    request_timeout: int = REQUEST_TIMEOUT
    request_timeout_escalation: list = None  # defaults to REQUEST_TIMEOUT_ESCALATION
    rps_limit: float = DEFAULT_RPS_LIMIT
    rps_warmup: float = DEFAULT_RPS_WARMUP

    # Pass 1: Tag Labeling
    labeling_model: str = DEFAULT_LABELING_MODEL
    concurrency: int = DEFAULT_CONCURRENCY
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    max_conversation_chars: int = MAX_CONVERSATION_CHARS
    truncation_head_ratio: float = TRUNCATION_HEAD_RATIO
    truncation_last_response_ratio: float = TRUNCATION_LAST_RESPONSE_RATIO
    truncation_per_turn_ratio: float = TRUNCATION_PER_TURN_RATIO
    dir_pipeline_watermark: float = DIR_PIPELINE_WATERMARK
    dir_pipeline_max_files: int = DIR_PIPELINE_MAX_FILES
    chunk_size: int = CHUNK_SIZE
    max_active_chunks: int = MAX_ACTIVE_CHUNKS
    sparse_full_label_count: int = SPARSE_FULL_LABEL_COUNT
    sparse_gap_multiplier: float = SPARSE_GAP_MULTIPLIER
    sparse_min_gap: int = SPARSE_MIN_GAP
    sparse_max_gap: int = SPARSE_MAX_GAP
    sparse_threshold: int = SPARSE_THRESHOLD

    # Pass 2: Value Scoring
    scoring_model: str = DEFAULT_SCORING_MODEL
    scoring_concurrency: int = DEFAULT_SCORING_CONCURRENCY
    value_weights: dict = None  # defaults to VALUE_WEIGHTS
    rarity_weights: dict = None  # defaults to RARITY_WEIGHTS
    rarity_combo_alpha: float = RARITY_COMBO_ALPHA
    value_truncation_budget: int = VALUE_TRUNCATION_BUDGET
    selection_intra_weight: float = SELECTION_INTRA_WEIGHT
    selection_quality_weight: float = SELECTION_QUALITY_WEIGHT
    selection_min_group_size: int = SELECTION_MIN_GROUP_SIZE
    selection_smoothing_prior: int = SELECTION_SMOOTHING_PRIOR
    enable_rationale: bool = ENABLE_RATIONALE
    prompt_mode: str = "full"  # "full" or "compact" (compact reduces few-shot count)
