## ADDED Requirements

### Requirement: Prompt-size non-increase for scoring changes
Any Pass 2 prompt revision introduced by this change SHALL be token-budget-neutral or smaller than the current production prompt configuration.

#### Scenario: Prompt edit replaces existing text
- **WHEN** new scoring guidance is added for tool-only slices, summary evidence, or trajectory-stage interpretation
- **THEN** the implementation SHALL replace or compress existing wording instead of appending net-new long instructions or examples

#### Scenario: No new few-shot payload growth
- **WHEN** the scoring prompt is revised
- **THEN** it SHALL NOT add new few-shot examples, larger meta blocks, or larger conversation wrappers that increase request payload size

### Requirement: Concise guardrails for low-information slices
The scoring prompt SHALL include concise wording that tells the model to avoid over-rewarding low-information tool-action slices.

#### Scenario: Tool-only navigation response
- **WHEN** the model sees a slice whose final assistant message is mostly file navigation or tool execution setup
- **THEN** the prompt SHALL direct the model to avoid assigning top quality/completeness scores unless technical progress is visible

### Requirement: Concise guardrails for unsupported summaries
The scoring prompt SHALL include concise wording that tells the model to avoid giving top-end quality credit to polished final summaries that lack concrete technical evidence.

#### Scenario: Summary without implementation evidence
- **WHEN** the model sees a polished final summary without concrete files, fixes, causes, or verification evidence
- **THEN** the prompt SHALL direct the model to avoid top-end completeness/explanation scores based on polish alone
