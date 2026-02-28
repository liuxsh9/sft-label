"""
Labeling Pipeline Configuration

All production settings extracted here for easy tuning.
"""

import os
from dataclasses import dataclass

# ─── LLM API ────────────────────────────────────────────
LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://localhost:4000/v1")
LITELLM_KEY = os.environ.get("LITELLM_KEY", "")

# ─── Pipeline Defaults ──────────────────────────────────
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONCURRENCY = 300
CONFIDENCE_THRESHOLD = 0.60
MAX_RETRIES = 3
SAMPLE_MAX_RETRIES = 3         # sample-level retry on call failure
REQUEST_TIMEOUT = 120          # seconds per LLM call (gpt-4o-mini is fast)

# ─── Conversation Truncation ──────────────────────────
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

# ─── Runtime-Overridable Config ──────────────────────
@dataclass
class PipelineConfig:
    """Runtime-overridable pipeline configuration.

    Module-level constants above remain unchanged for CLI mode and
    direct imports from other modules (e.g. preprocessing.py).
    This dataclass lets library callers override any setting.
    """
    litellm_base: str = LITELLM_BASE
    litellm_key: str = LITELLM_KEY
    model: str = DEFAULT_MODEL
    concurrency: int = DEFAULT_CONCURRENCY
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    max_retries: int = MAX_RETRIES
    sample_max_retries: int = SAMPLE_MAX_RETRIES
    request_timeout: int = REQUEST_TIMEOUT
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
