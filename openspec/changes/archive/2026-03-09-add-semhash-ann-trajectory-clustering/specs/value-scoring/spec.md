## MODIFIED Requirements

### Requirement: Output file structure
Pass 2 SHALL write output files alongside Pass 1 outputs. Per-file outputs: `scored[_prefix].json`, `scored[_prefix].jsonl`, `stats_value[_prefix].json`, `monitor_value[_prefix].jsonl`, `dashboard_value[_prefix].html`, `failed_value[_prefix].jsonl`. Global outputs: `summary_stats_value.json`, `dashboard_value_<name>.html`.

Per-sample value records SHALL remain backward compatible and SHALL preserve stable linkage metadata needed by trajectory-level semantic clustering (`id`, `metadata.source_id` when available, and turn/window positional metadata when present). Value-scoring output SHALL be consumable as-is by downstream representative selection pipelines without requiring schema transformation.

#### Scenario: Per-sample output structure
- **WHEN** a sample is scored
- **THEN** the output SHALL contain the original `labels` field (unchanged) plus a new `value` field with complexity, quality, reasoning, rarity (with stats_ref), flags, thinking_mode, value_score, and confidence

#### Scenario: Directory mode output
- **WHEN** scoring a directory of labeled files
- **THEN** each file SHALL produce prefixed output files, plus global `summary_stats_value.json` and `dashboard_value_<name>.html`

#### Scenario: Downstream trajectory selection interoperability
- **WHEN** semantic clustering representative selection consumes scored samples
- **THEN** the required linkage metadata and `value.value_score` SHALL be available in output without additional conversion
