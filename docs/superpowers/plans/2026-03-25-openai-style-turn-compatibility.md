# OpenAI-style Turn Compatibility Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the labeling pipeline natively accept `conversations[].role/content` and `messages[].role/content` inputs, preserve single-turn and multi-turn slicing semantics, and keep inline provenance accurate.

**Architecture:** Centralize schema detection, routing, provenance, and turn normalization inside `src/sft_label/preprocessing.py`, then keep downstream consumers on the existing internal `conversations[].from/value` contract. Add regression coverage at the preprocessing, JSONL row-ingestion, file-reader, and inline merge/provenance layers so old ShareGPT/Pangu behavior stays stable while new OpenAI-style inputs become first-class.

**Tech Stack:** Python, pytest, existing preprocessing / inline merge / JSONL ingestion helpers.

---

## Chunk 1: Normalization and detection core

### Task 1: Lock the routing and normalization contract in preprocessing tests

**Files:**
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing format-detection tests**
  - Add direct `detect_format()` coverage for:
    - `conversations[].role/content` routing to `sharegpt`
    - `messages[].role/content` routing to `openai_messages`
    - `data` beating lower-priority keys when legal and non-empty
    - empty `data`/`conversations` falling back to a lower-priority non-empty legal source
    - illegal `conversations` falling back to valid non-empty `messages`
    - illegal high-priority sources raising instead of silently downgrading when no non-empty legal fallback exists.

- [ ] **Step 2: Write failing normalization/slicing tests**
  - Add tests for:
    - `conversations[].role/content` single-turn normalization to `human/gpt`
    - `None` selected text fields normalizing to `""`
    - turns missing both field pairs normalizing to `{from: "", value: ""}`
    - `conversations[].role/content` multi-turn slicing with preserved context
    - leading `system` and inline `tool` turns surviving normalization and slicing
    - `messages[].role/content` single-turn and multi-turn normalization
    - mixed `conversations` provenance choosing `sharegpt` vs `openai_conversations` from the first non-empty, fully-formed legal turn while ignoring leading empty/incomplete turns
    - unknown roles lowercasing and preserving rather than remapping/dropping
    - an OpenAI-style long trajectory preserving sample count plus `turn_index` / `total_turns` stability
    - zero-assistant rows staying as one unsliced sample
    - a winning empty source producing exactly one unsliced sample with `conversations == []`
    - `metadata.original_format` being set/preserved for `sharegpt`, `openai_conversations`, `openai_messages`, and `pangu`
    - a `conversations` turn carrying both `from/value` and `role/content`, asserting `from/value` wins and extra `role/content` is ignored
    - invalid selected text fields (`list`/`dict`) raising `ValueError` with `source_file`, `source_row`, offending field name, turn index, and type clues.

- [ ] **Step 3: Write failing no-slice normalization tests**
  - Add `normalize_sample()` / `preprocess()` tests proving OpenAI-style schemas normalize even when the caller does not use `normalize_and_slice()`.
  - Include parity checks for invalid-content `ValueError` behavior and `metadata.original_format` / provenance behavior on the no-slice path.

- [ ] **Step 4: Run focused preprocessing tests to verify RED**

Run:
`uv run pytest tests/test_preprocessing.py -q`

Expected:
FAIL on the new detection/normalization assertions.

### Task 2: Implement centralized routing, provenance, and turn normalization

**Files:**
- Modify: `src/sft_label/preprocessing.py`

- [ ] **Step 1: Add a single routing/provenance decision helper**
  - Introduce one focused helper that resolves:
    - routing format
    - provenance/original format
    - normalized internal turns
  - Keep `data`, `conversations`, and `messages` precedence in one place.

- [ ] **Step 2: Add turn-shape normalization helpers**
  - Support:
    - internal `from/value`
    - `conversations[].role/content`
    - `messages[].role/content`
  - Only `conversations` may mix `from/value` and `role/content` turns; `messages`/`data` stay on their declared shape.
  - For mixed `conversations`, derive provenance from the first non-empty, fully-formed legal turn and ignore leading empty/incomplete turns for provenance bookkeeping.
  - If a turn already has internal `from/value`, use those fields and ignore extra `role/content` fields on the same turn.

- [ ] **Step 3: Enforce the failure and fallback contract**
  - Non-string selected text fields (`list`/`dict`) must raise `ValueError` unless a lower-priority non-empty legal fallback source is available.
  - Empty sources win only when no later non-empty legal source exists.

- [ ] **Step 4: Thread the helper through both normalization entry points**
  - Update `detect_format()` to return stable routing values.
  - Update `normalize_and_slice()` to normalize before COT extraction and slicing.
  - Update `normalize_sample()` so no-slice callers share the same behavior and provenance.

- [ ] **Step 5: Re-run preprocessing tests to verify GREEN**

Run:
`uv run pytest tests/test_preprocessing.py -q`

Expected:
PASS

## Chunk 2: JSONL/file-reader and inline provenance integration

### Task 3: Lock row-ingestion and mixed-file behavior with failing tests

**Files:**
- Modify: `tests/test_inline_rows.py`

- [ ] **Step 1: Write failing JSONL bundle tests**
  - Add row-bundle coverage for:
    - `conversations[].role/content` single-turn and multi-turn rows
    - `messages[].role/content` rows
    - a long-trajectory JSONL row containing `system`/`tool` turns and 3+ assistant replies
    - an accepted JSONL row with `content: None`
    - a mixed-schema JSONL file read through `iter_row_sample_bundles_from_jsonl()` / `iter_samples_from_file(..., return_row_bundles=True)`
    - fallback from invalid/empty higher-priority sources to lower-priority non-empty legal sources.

- [ ] **Step 2: Write failing invalid-input file-reader tests**
  - Add an explicit JSONL row whose selected `content`/text field is a `list` or `dict` and assert it raises `ValueError` through the row-ingestion/file-reader path because no non-empty legal fallback exists.
  - Add a 0-assistant JSONL row and assert file-reader / flattening preserves the single-sample contract end-to-end.
  - In the accepted row-bundle cases, assert stable sample IDs and turn metadata remain correct after flattening/file-reader expansion.

- [ ] **Step 3: Run row-ingestion tests to verify RED**

Run:
`uv run pytest tests/test_inline_rows.py -q`

Expected:
FAIL on the new bundle / mixed-file assertions.

### Task 4: Lock inline provenance overwrite behavior with failing tests

**Files:**
- Modify: `tests/test_inline_pass1.py`

- [ ] **Step 1: Write failing source-format tests for new inputs**
  - Add merge coverage showing `conversations[].role/content` writes `openai_conversations` and `messages[].role/content` writes `openai_messages` into `data_label.meta.source_format`.

- [ ] **Step 2: Write a failing incremental-merge regression**
  - Start from an existing row whose `data_label.meta.source_format` is stale (`sharegpt`), then run an incremental merge on a newly-normalized `openai_conversations` row with `use_existing_data_label=True` and assert the stale provenance is overwritten.
  - Add the matching refresh-path regression so both refresh and incremental merges rewrite newly inferred provenance.

- [ ] **Step 3: Run inline-merge tests to verify RED**

Run:
`uv run pytest tests/test_inline_pass1.py -q -k 'source_format or openai or incremental'`

Expected:
FAIL on the new provenance assertions.

### Task 5: Implement row-ingestion and inline provenance integration

**Files:**
- Modify: `src/sft_label/inline_pass1.py`
- Verify only: `src/sft_label/inline_rows.py`
- Verify only: `src/sft_label/pipeline.py`

- [ ] **Step 1: Update source-format inference**
  - Make `_infer_source_format()` prefer normalized sample provenance over raw top-level key heuristics.

- [ ] **Step 2: Overwrite stale inline provenance on merge**
  - Ensure both refresh and incremental / `use_existing_data_label=True` paths write the newly inferred `source_format` back into `data_label.meta.source_format`.

- [ ] **Step 3: Re-run row-ingestion and inline-merge tests to verify GREEN**

Run:
`uv run pytest tests/test_inline_rows.py tests/test_inline_pass1.py -q -k "openai or source_format or iter_samples_from_file or row_sample_bundle"`

Expected:
PASS

- [ ] **Step 4: Run the full row-ingestion + inline-merge subset**

Run:
`uv run pytest tests/test_inline_rows.py tests/test_inline_pass1.py -q`

Expected:
PASS

## Chunk 3: Full verification and real-data compatibility probe

### Task 6: Run focused regression suite

**Files:**
- Verify only

- [ ] **Step 1: Run the focused regression set**

Run:
`uv run pytest tests/test_preprocessing.py tests/test_inline_rows.py tests/test_inline_pass1.py -q`

Expected:
PASS

- [ ] **Step 2: Run additional affected reader/integration tests only if Step 1 exposes failures outside the core targeted suite**

Run:
`uv run pytest tests/test_conversation.py tests/test_filter.py tests/test_inline_scoring.py -q`

Expected:
PASS

### Task 7: Probe the real target dataset with the new normalization path

**Files:**
- Create: `scripts/probe_openai_turn_compat.py`

- [ ] **Step 1: Add a repeatable local compatibility probe script**
  - Create `scripts/probe_openai_turn_compat.py` that:
    - walks `/Volumes/MOVESPEED/datasets/Step-3.5-Flash-SFT-code-sft-jsonl-qwen3-coder-plus`
    - samples representative rows across single-turn, 2+ assistant multi-turn, and 3+ assistant `system`/`tool` long trajectories
    - runs them through the real file-reader / `iter_samples_from_file()` normalization path
    - prints per-case summaries of detected route, provenance, role order, and sample counts.

- [ ] **Step 2: Run the probe script against the target dataset**

Run:
`uv run python scripts/probe_openai_turn_compat.py --input /Volumes/MOVESPEED/datasets/Step-3.5-Flash-SFT-code-sft-jsonl-qwen3-coder-plus`

Expected:
The script reports successful coverage for:
- single-turn `user/assistant` rows
- 2+ assistant multi-turn rows
- 3+ assistant long trajectories with `system`/`tool` turns
- role normalization correctness and preserved turn order
- sample counts matching assistant replies (or the zero-assistant/empty-source single-sample contract).

- [ ] **Step 3: Record observed coverage and any limitations in the final handoff**
  - Include which sample shapes were successfully probed, any unsupported schema discovered, and whether the directory-reading path (not just helper-level normalization) was exercised end-to-end.
