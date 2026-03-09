## Why

Directory mode scoring currently violates user expectations for `output_dir`: it may fail when the directory does not exist, and per-file artifacts are written back to input folders instead of the requested output root. This breaks reproducible data curation workflows and multi-session review handoff.

## What Changes

- Ensure directory-mode scoring always creates and validates `output_dir` before writing summary artifacts.
- Route per-file scoring artifacts to a mirrored path under `output_dir` instead of writing into input folders.
- Keep global artifacts (`summary_stats_scoring.json`, global dashboard, global conversation scores) consistently under `output_dir`.
- Make global conversation aggregation scan scored outputs recursively under the output root.
- Add regression tests for non-existent output roots, nested input trees, and output path consistency.

## Capabilities

### New Capabilities
- `scoring-directory-output-routing`: Directory scoring writes all per-file and global artifacts under the configured output root with deterministic mirrored paths.

### Modified Capabilities
- None.

## Impact

- Affected code: `src/sft_label/scoring.py` (directory-mode output path resolution and global aggregation scan).
- Affected tests: scoring directory-mode path behavior and nested aggregation coverage.
- Operational impact: safer batch scoring for large corpora and multi-session pipelines.
