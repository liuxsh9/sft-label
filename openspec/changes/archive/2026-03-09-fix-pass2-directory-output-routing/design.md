## Context

`run_scoring(..., output_dir=...)` in directory mode currently resolves global outputs in the requested root, but per-file outputs are still written to `labeled_path.parent`. In addition, if `output_dir` does not exist, writing the summary can fail even though scoring completed.

## Goals / Non-Goals

**Goals:**
- Make `output_dir` authoritative for all directory-mode Pass 2 artifacts.
- Preserve relative input structure under `output_dir` for per-file outputs.
- Keep existing behavior unchanged when `output_dir` is omitted.
- Ensure global conversation aggregation includes nested scored outputs.

**Non-Goals:**
- No changes to scoring model prompts or value score formulas.
- No changes to Pass 1 output layout.

## Decisions

1. Normalize and create `output_dir` at the start of `_run_scoring_directory`.
   - Rationale: remove write-time failures and keep behavior deterministic.
   - Alternative considered: lazily create per write call. Rejected due to repeated path checks and inconsistent failure surface.

2. Compute per-file output path as `output_dir / relative_parent_of_labeled_file`.
   - Rationale: preserves dataset structure while honoring explicit output root.
   - Alternative considered: flatten all outputs into one directory. Rejected because of name collisions and poorer traceability.

3. Keep summary and global dashboard outputs at the output root.
   - Rationale: consistent single location for run-level artifacts.

4. Replace one-level global conversation scan with recursive discovery of `scored.json` and `scored.jsonl`.
   - Rationale: directory trees often contain nested folders such as `code/...` and `multi_turn/...`.

## Risks / Trade-offs

- [Risk] Existing consumers may read per-file artifacts from input folders.
  - Mitigation: preserve legacy behavior only when `output_dir` is not set; document behavior in release notes.

- [Risk] Recursive scan may read duplicate files if both JSON and JSONL exist.
  - Mitigation: use deterministic preference (`scored.json` over `scored.jsonl`) per directory.

- [Risk] Path bugs on unusual relative layouts.
  - Mitigation: add regression tests for flat and nested paths.
