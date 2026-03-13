## ADDED Requirements

### Requirement: Evidence-aware post-LLM aggregation
Pass 2 SHALL support deterministic post-LLM adjustments that use local slice structure to refine `value_score` and `selection_score` after the model returns its JSON result.

#### Scenario: Low-information exploration slice
- **WHEN** Pass 2 identifies a slice as low-information exploration dominated by navigation/tool actions
- **THEN** the system SHALL apply a bounded downward adjustment before persisting final aggregate scores

#### Scenario: Evidence-backed implementation slice
- **WHEN** Pass 2 identifies a slice with concrete implementation or verification evidence
- **THEN** the system SHALL preserve the LLM-derived quality signal aside from any configured global normalization

### Requirement: Diversity-aware selection composition
The system SHALL compute `selection_score` from a blend of sample quality/value, intra-class rank, rarity/diversity contribution, and stage/evidence modifiers, rather than allowing intra-class rank to dominate the full selection delta.

#### Scenario: Two equally strong slices with different rarity
- **WHEN** two slices have similar value and quality but one belongs to a rarer capability/domain pattern
- **THEN** the rarer slice SHALL receive a higher selection score by a measurable margin

#### Scenario: Common high-quality slice remains selectable
- **WHEN** a slice is high quality but comes from a common bug-fix pattern
- **THEN** the system SHALL keep it high quality while giving it less selection bonus than an equally strong but more diverse slice

### Requirement: Stable output schema during selection rebalance
Selection rebalance SHALL NOT change persisted field names or primary JSON structure for scored samples and dashboards.

#### Scenario: Existing consumers continue to read outputs
- **WHEN** a run completes with the new selection logic enabled
- **THEN** persisted rows SHALL still expose `value_score`, `selection_score`, `intra_class_rank`, `rarity`, `quality`, `reasoning`, and existing aggregate structures under their current field names

### Requirement: Conservative domain backfill for repo-repair slices
The scoring pipeline SHALL allow conservative domain backfill using preprocessing or deterministic heuristics when Pass 1 leaves repo-repair slices without a useful domain label.

#### Scenario: Obvious domain inferred from issue and file context
- **WHEN** repo metadata or issue context clearly indicates an existing domain such as compiler-development, machine-learning, cloud-computing, or api-development
- **THEN** the system SHALL backfill the matching existing domain label without requiring a larger prompt

#### Scenario: Ambiguous domain remains empty
- **WHEN** deterministic heuristics cannot infer a domain confidently
- **THEN** the system SHALL preserve the empty domain rather than guessing
