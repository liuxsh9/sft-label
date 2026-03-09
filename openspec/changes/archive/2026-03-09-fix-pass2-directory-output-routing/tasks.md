## 1. Output Root Routing

- [x] 1.1 Normalize and create directory-mode `output_dir` at scoring start
- [x] 1.2 Route `_flush_scoring_file` outputs to mirrored directories under `output_dir`
- [x] 1.3 Keep legacy behavior unchanged when `output_dir` is omitted

## 2. Global Artifact Consistency

- [x] 2.1 Keep summary/global dashboard emission rooted at `output_dir`
- [x] 2.2 Make global conversation aggregation recursively discover scored artifacts
- [x] 2.3 Deduplicate mixed `scored.json` and `scored.jsonl` inputs deterministically

## 3. Verification

- [x] 3.1 Add regression test for non-existent `output_dir`
- [x] 3.2 Add regression test that per-file outputs are written under mirrored output tree
- [x] 3.3 Add regression test for nested global conversation aggregation
