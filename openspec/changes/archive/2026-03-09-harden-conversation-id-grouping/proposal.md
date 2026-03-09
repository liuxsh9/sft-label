## Why

Conversation aggregation currently groups slices by `source_id` alone. In multi-source corpora, different files can reuse the same `source_id`, causing unrelated conversations to be merged and corrupting `conv_value`/`conv_selection` filtering decisions.

## What Changes

- Introduce a canonical conversation identity key that incorporates both `source_file` and `source_id` when available.
- Update conversation aggregation to group and score by the canonical key.
- Keep backward compatibility for legacy data lacking `source_file`.
- Update conversation-level filtering lookup to resolve canonical IDs with fallback to legacy IDs.
- Add collision-focused tests for aggregation and filter behavior.

## Capabilities

### New Capabilities
- `conversation-identity-grouping`: Conversation scoring and filtering use collision-safe IDs across files.

### Modified Capabilities
- None.

## Impact

- Affected code: `src/sft_label/conversation.py`, `src/sft_label/tools/filter_value.py`.
- Affected tests: conversation aggregation and conversation-level filter lookup.
- Data quality impact: prevents cross-source contamination of conversation scores.
