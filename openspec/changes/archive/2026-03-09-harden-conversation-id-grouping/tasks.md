## 1. Canonical Identity Implementation

- [x] 1.1 Add shared helper to build canonical conversation key from sample metadata
- [x] 1.2 Update conversation grouping to use canonical key
- [x] 1.3 Ensure `conversation_scores.json` emits canonical `conversation_id`

## 2. Filter Alignment

- [x] 2.1 Update conversation-level filter lookup to resolve canonical IDs
- [x] 2.2 Add legacy fallback from canonical key to plain `source_id`
- [x] 2.3 Add diagnostics for fallback usage and missing identity metadata

## 3. Verification

- [x] 3.1 Add aggregation regression test for cross-file `source_id` collision
- [x] 3.2 Add filter regression test for canonical ID matching
- [x] 3.3 Add backward-compat test for legacy `conversation_scores.json`
