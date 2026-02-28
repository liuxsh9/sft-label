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
REQUEST_TIMEOUT = 120          # seconds per LLM call (gpt-4o-mini is fast)

# ═══════════════════════════════════════════════════════════
# Pass 1: Tag Labeling
# ═══════════════════════════════════════════════════════════

DEFAULT_LABELING_MODEL = "gpt-4o-mini"
DEFAULT_CONCURRENCY = 200
CONFIDENCE_THRESHOLD = 0.60

# ─── Conversation Truncation (Pass 1) ───────────────────
MAX_CONVERSATION_CHARS = 20000   # total budget (~5K tokens); aggressive for fast labeling
TRUNCATION_HEAD_RATIO = 0.35     # fraction of budget for first human turn (task context)
TRUNCATION_LAST_RESPONSE_RATIO = 0.30  # fraction of budget for last gpt turn (labeling target)
TRUNCATION_PER_TURN_RATIO = 0.35 # max fraction of budget for any single turn

# ─── Directory Pipeline ────────────────────────────────
DIR_PIPELINE_WATERMARK = 2.0   # load next file when in-flight < concurrency * watermark
DIR_PIPELINE_MAX_FILES = 5     # max files loaded in memory simultaneously

# ─── Sparse Sampling (multi-turn slices) ──────────────
SPARSE_FULL_LABEL_COUNT = 10   # first N slices always labeled
SPARSE_GAP_MULTIPLIER = 1.4   # gap between labeled slices grows by this factor
SPARSE_MIN_GAP = 2            # minimum gap between labeled slices
SPARSE_THRESHOLD = 12         # slices <= this: label all, no sparse sampling

# ─── Consistency Rules ──────────────────────────────────
CONSISTENCY_RULES = [
    ("intent == 'learn' and len(agentic) > 3",
     "Intent=learn but many agentic tags"),
    ("intent == 'build' and 'feature-implementation' not in task and len(task) > 0",
     "Intent=build but no feature-implementation in task"),
    ("difficulty == 'beginner' and len(concept) > 3",
     "Difficulty=beginner but many concepts"),
    ("len(constraint) > 0 and difficulty == 'beginner'",
     "Has constraints but difficulty=beginner"),
]

# ═══════════════════════════════════════════════════════════
# Pass 2: Value Scoring
# ═══════════════════════════════════════════════════════════

DEFAULT_SCORING_MODEL = "gpt-4o-mini"
DEFAULT_SCORING_CONCURRENCY = 200

VALUE_WEIGHTS = {
    "complexity": 0.25,
    "quality": 0.35,
    "reasoning": 0.15,
    "rarity": 0.25,
}
RARITY_WEIGHTS = {
    "intent": 0.3,
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
})
KNOWN_FLAGS = KNOWN_FLAGS_POSITIVE | KNOWN_FLAGS_NEGATIVE


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
    sparse_full_label_count: int = SPARSE_FULL_LABEL_COUNT
    sparse_gap_multiplier: float = SPARSE_GAP_MULTIPLIER
    sparse_min_gap: int = SPARSE_MIN_GAP
    sparse_threshold: int = SPARSE_THRESHOLD

    # Pass 2: Value Scoring
    scoring_model: str = DEFAULT_SCORING_MODEL
    scoring_concurrency: int = DEFAULT_SCORING_CONCURRENCY
    value_weights: dict = None  # defaults to VALUE_WEIGHTS
    rarity_weights: dict = None  # defaults to RARITY_WEIGHTS
    rarity_combo_alpha: float = RARITY_COMBO_ALPHA
    value_truncation_budget: int = VALUE_TRUNCATION_BUDGET
