## MODIFIED Requirements

### Requirement: Per-file value dashboard
The system SHALL generate a per-file value dashboard for inline-labeled datasets using embedded `data_label` values and Pass 2 stats produced under `meta_label_data/`. The dashboard SHALL continue to expose value overview cards, score distributions, sub-score breakdowns, value-by-tag analysis, thinking-mode analysis, and flag analysis for the corresponding mirrored file.

#### Scenario: Dashboard from inline-labeled mirrored file
- **WHEN** dashboard generation targets a mirrored inline-labeled file
- **THEN** the system SHALL derive the dashboard data from the file’s embedded `data_label` records or from rebuildable flattened caches generated from those records

### Requirement: Global value dashboard
The system SHALL generate a global value dashboard at the run root for directory-mode inline-labeled runs. The dashboard SHALL summarize the mirrored dataset tree using run-level Pass 2 stats and conversation aggregates derived from embedded annotations.

#### Scenario: Global dashboard over mirrored dataset tree
- **WHEN** dashboard generation targets a run root containing a mirrored inline-labeled dataset tree
- **THEN** the system SHALL aggregate per-file and run-level metrics from embedded `data_label` records and render a global dashboard at the run root

#### Scenario: File ranking table
- **WHEN** the global dashboard is generated
- **THEN** it SHALL display one row per mirrored dataset file with counts and mean score metrics derived from the embedded annotations for that file

### Requirement: stats_value.json structure
The system SHALL produce a Pass 2 stats artifact under `meta_label_data/` that contains score distributions, sub-score means, value-by-tag summaries, thinking-mode stats, flag summaries, selection thresholds, coverage summaries, and rarity configuration derived from embedded annotations.

#### Scenario: Per-file stats from embedded annotations
- **WHEN** per-file scoring completes for an inline-labeled file
- **THEN** the per-file Pass 2 stats artifact under `meta_label_data/` SHALL contain the same metric families as before, computed from the embedded `data_label` values for that file

#### Scenario: Global summary with per-file breakdown
- **WHEN** directory-mode scoring completes for an inline-labeled run
- **THEN** the run-level Pass 2 summary artifact under `meta_label_data/` SHALL contain aggregated stats plus a per-file summary derived from the mirrored dataset tree
