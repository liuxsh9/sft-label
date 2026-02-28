## ADDED Requirements

### Requirement: Scoring system prompt
The system SHALL provide a system prompt for Pass 2's LLM call that evaluates three dimensions: complexity (1-10), quality (1-10), and reasoning (1-10). The prompt SHALL include detailed scoring anchors at 2-point intervals with concrete code examples for each level.

#### Scenario: Complexity scoring rubric
- **WHEN** the LLM evaluates complexity
- **THEN** the prompt SHALL define anchors: 1-2 (trivial: print hello world), 3-4 (basic: simple CRUD), 5-6 (intermediate: design patterns, moderate algorithms), 7-8 (advanced: system design, concurrency), 9-10 (expert: novel algorithms, deep system internals)

#### Scenario: Quality scoring rubric
- **WHEN** the LLM evaluates quality
- **THEN** the prompt SHALL require sub-scores for correctness, code_quality, explanation, and completeness, with correctness weighted highest and a `correctness_assessable` boolean flag

#### Scenario: Reasoning adaptation for thinking mode
- **WHEN** `thinking_mode` is `"slow"`
- **THEN** the prompt SHALL instruct evaluation of COT clarity, consistency with final answer, and self-correction behavior
- **WHEN** `thinking_mode` is `"fast"`
- **THEN** the prompt SHALL instruct evaluation of reasoning integration within response and explanation depth

### Requirement: Scoring few-shot examples
The prompt SHALL include 2-3 few-shot examples covering diverse scenarios: a high-complexity/high-quality sample, a low-complexity/high-quality sample, and a high-complexity/low-quality sample (with bugs or incomplete response).

#### Scenario: Few-shot variety
- **WHEN** the LLM processes a scoring request
- **THEN** it SHALL have seen examples demonstrating the full range of scores (2-3 through 8-9) across dimensions

### Requirement: Structured JSON output
The prompt SHALL instruct the LLM to return a JSON object with exact field names matching the value schema: `complexity` (object with instruction/reasoning/implementation/overall), `quality` (object with correctness/code_quality/explanation/completeness/overall), `reasoning` (object adapting to thinking_mode), `flags` (array of strings), and `confidence` (float 0-1).

#### Scenario: Output validation
- **WHEN** the LLM returns a response
- **THEN** the system SHALL validate that all required fields are present, all scores are integers 1-10, confidence is a float 0.0-1.0, and flags are from the known flag vocabulary

### Requirement: Flag vocabulary
The prompt SHALL define a controlled vocabulary of flags. Positive: `excellent-explanation`, `clean-code`, `creative-solution`, `good-error-handling`, `comprehensive-testing`. Negative: `has-bug`, `security-issue`, `outdated-practice`, `incomplete`, `over-engineered`, `incorrect-output`, `poor-explanation`.

#### Scenario: Unknown flags
- **WHEN** the LLM returns a flag not in the controlled vocabulary
- **THEN** the system SHALL accept it but track it separately as an unmapped flag in monitoring

### Requirement: User message construction
The scoring prompt's user message SHALL contain: a `<meta>` block with thinking_mode, original char lengths (COT, response), turn count, code block count, and Pass 1 tags; followed by a `<conversation>` block with the smart-truncated conversation including COT content.

#### Scenario: Meta block content
- **WHEN** building the user message
- **THEN** `<meta>` SHALL include `thinking_mode`, `original_cot_chars`, `original_response_chars`, `total_turns`, `code_block_count`, and `tags` (the full Pass 1 labels dict)

#### Scenario: Truncated conversation with COT
- **WHEN** a slow-thinking sample is truncated
- **THEN** COT content SHALL appear within `[COT]...[/COT]` markers inside the conversation, with fragment position annotations
