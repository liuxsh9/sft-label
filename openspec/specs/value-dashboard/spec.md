## ADDED Requirements

### Requirement: Per-file value dashboard
The system SHALL generate `dashboard_value.html` for each scored file, containing 6 sections beyond what Pass 1's dashboard shows: Value Overview Cards, Score Distributions, Sub-score Breakdown, Value×Tag Cross-Analysis, Thinking Mode Analysis, and Flag Analysis.

#### Scenario: Value Overview Cards
- **WHEN** dashboard is generated
- **THEN** it SHALL display cards for: Total Scored, Mean Value Score, Mean Complexity, Mean Quality, Median Rarity, Top 10% count (and threshold), Bottom 10% count, and top negative flag count

#### Scenario: Score Distributions
- **WHEN** dashboard is generated
- **THEN** it SHALL show histograms (1-10 bins) for value_score, complexity, quality, reasoning, and rarity side by side

#### Scenario: Value×Tag Cross-Analysis
- **WHEN** dashboard is generated
- **THEN** it SHALL show: quality mean by difficulty level, value score mean by domain (top 15), complexity vs quality scatter plot representation, and rarity vs quality quadrant analysis with per-quadrant sample counts

#### Scenario: Thinking Mode Analysis
- **WHEN** dashboard is generated
- **THEN** it SHALL show slow vs fast thinking comparison table (count, mean value, mean quality, mean reasoning) and COT length vs reasoning score breakdown for slow-thinking samples

#### Scenario: Flag Analysis
- **WHEN** dashboard is generated
- **THEN** it SHALL show flag frequency bar chart and mean value score per flag

### Requirement: Global value dashboard
The system SHALL generate a global `dashboard_value_<name>.html` for directory-mode scoring with cross-file analysis capabilities.

#### Scenario: File Ranking Table
- **WHEN** global dashboard is generated
- **THEN** it SHALL display a sortable table with one row per file, columns: file name, sample count, mean value, mean complexity, mean quality, mean rarity, top 10% count, negative flag count

#### Scenario: Coverage Impact Analysis
- **WHEN** global dashboard is generated
- **THEN** it SHALL show for each of several pre-computed value thresholds: how many samples retained, percentage, which tags would be lost, and overall tag coverage rate

#### Scenario: Data Selection Simulator
- **WHEN** global dashboard is generated
- **THEN** it SHALL show pre-computed threshold points (top 10%, 25%, 50%, 75%) with retained count, tag coverage, mean quality uplift, and files most affected

#### Scenario: Global Score Distributions
- **WHEN** global dashboard is generated from summary_stats_value.json
- **THEN** it SHALL show aggregated score distributions and value-by-tag breakdowns across all files

### Requirement: Self-contained HTML
All dashboard files SHALL be self-contained HTML with embedded CSS and JavaScript, requiring no external dependencies. Data SHALL be injected as a JSON variable in a script tag.

#### Scenario: Offline viewing
- **WHEN** a dashboard HTML file is opened in any modern browser
- **THEN** it SHALL render completely without network access

### Requirement: stats_value.json structure
The system SHALL output `stats_value.json` containing: score distributions (mean/std/percentiles for each dimension), sub-score means, value_by_tag (mean value per tag per dimension), thinking_mode_stats, flag_counts, flag_value_impact, selection_thresholds, coverage_at_thresholds, weights_used, and rarity_config.

#### Scenario: Per-file stats
- **WHEN** per-file scoring completes
- **THEN** `stats_value.json` SHALL contain all distribution data plus the weights and rarity config used

#### Scenario: Global summary with per-file breakdown
- **WHEN** directory-mode scoring completes
- **THEN** `summary_stats_value.json` SHALL contain aggregated stats plus a `per_file_summary` array with per-file mean scores and counts
