## ADDED Requirements

### Requirement: Stage-aware slice classification
The system SHALL infer a deterministic `trajectory_stage` for each scored slice using local features already available in preprocessing or scoring. The inferred stage SHALL distinguish at least opener, exploration, implementation, verification, and final-summary behaviors without requiring larger prompts or additional LLM calls.

#### Scenario: Opener slice detection
- **WHEN** a slice is in the earliest trajectory position and the assistant response mostly announces a plan or immediately starts navigation/tool actions without technical conclusions
- **THEN** the system SHALL classify the slice as `opener`

#### Scenario: Final-summary slice detection
- **WHEN** the assistant response contains final-summary style markers such as summary/final review/finish language and appears near the end of the trajectory
- **THEN** the system SHALL classify the slice as `final-summary`

### Requirement: Tool-action penalty for low-information slices
The system SHALL apply a bounded penalty to value/selection aggregation when the final assistant response is predominantly tool actions or file-navigation actions and lacks technical evidence of diagnosis, implementation, or verification.

#### Scenario: Tool-call-only view action
- **WHEN** the final assistant response is mostly a `view`, `grep`, `ls`, or similar tool action with little or no explanatory content
- **THEN** the system SHALL reduce the slice's post-LLM aggregate score relative to a contentful implementation or verification slice

#### Scenario: Tool-assisted but evidence-rich slice
- **WHEN** the final assistant response includes tool actions together with concrete diagnosis, code-level conclusions, or verification results
- **THEN** the system SHALL NOT apply the full low-information penalty

### Requirement: Summary-evidence guard
The system SHALL require concrete technical evidence before a final-summary style slice can receive the highest selection band. Evidence MAY include file paths, function/class references, root-cause statements, tests/commands run, verification outcomes, or explicit fix descriptions.

#### Scenario: Summary with evidence remains high
- **WHEN** a final-summary slice includes concrete technical evidence and strong quality signals
- **THEN** the system SHALL allow the slice to remain in a top selection bucket

#### Scenario: Summary without evidence is discounted
- **WHEN** a final-summary slice is polished but lacks concrete technical evidence
- **THEN** the system SHALL reduce its selection score relative to evidence-backed summaries of similar base quality

### Requirement: Audited ranking regression pack
The repository SHALL include an audited regression pack for selection stability using real reviewed trajectory slices from the Nemotron SWE sample set.

#### Scenario: Regression coverage includes representative failure modes
- **WHEN** the regression pack is built
- **THEN** it SHALL include opener/incomplete slices, tool-action exploration slices, common high-quality bug-fix finals, and summary-heavy high-selection finals

#### Scenario: Regression assertions use relative ordering
- **WHEN** regression tests run
- **THEN** they SHALL assert ordering or bucket movement between curated slices instead of exact floating-point score equality
