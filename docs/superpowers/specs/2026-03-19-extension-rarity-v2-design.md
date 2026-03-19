# Extension Rarity V2 Design

## Background

`sft-label` currently computes rarity, value, and selection entirely from the core Pass 1 9-dimension taxonomy:

- `intent`
- `language`
- `domain`
- `task`
- `difficulty`
- `concept`
- `agentic`
- `constraint`
- `context`

User-defined Pass 1 extensions already exist as a parallel labeling layer. They produce per-spec payloads under `label_extensions.<spec_id>` and aggregate into `extension_stats`, but they do **not** participate in:

- core rarity
- combo rarity
- value scoring
- selection grouping
- conversation rarity/selection

That separation is currently the right default and is explicitly documented in the existing extension-labeling design.

The new goal is to support a future where extension labels can influence rarity in a **stable, versioned, opt-in** way, without breaking current semantics or historical runs.

---

## Design Goal

Add a complete **Extension Rarity V2** architecture that can be implemented in one pass, while preserving the current core scoring contract.

The design must allow:

1. computing rarity signals from extension labels,
2. storing those signals reproducibly,
3. exposing them in dashboards / exports / recompute flows,
4. optionally blending them into new V2 score fields,
5. while leaving all current score fields unchanged by default.

---

## Non-Goals

1. Do **not** change the meaning of current `value.rarity.score`.
2. Do **not** fold extension labels into core `tag_distributions` or core `combo_distributions`.
3. Do **not** let extension labels silently affect current `value_score`, `selection_score`, `conv_rarity`, or `conv_selection`.
4. Do **not** add extension fields to selection percentile grouping dimensions.
5. Do **not** introduce extension combo rarity in the first version.
6. Do **not** enable extension rarity by default.
7. Do **not** let `label_extensions` leak into the Pass 2 prompt unless that is explicitly versioned in a separate scoring-prompt change.

---

## Invariants (Must Never Break)

### 1. Core rarity semantics remain canonical

Current fields continue to mean what they mean today:

- `value.rarity.score`
- `value.value_score`
- `value.selection_score`
- existing conversation V2 shadow metrics such as the current `conv_selection_v2`

Any new semantics must be placed in **new fields**.

### 2. Core rarity baseline stays core-only

The current core rarity pipeline is built from core label distributions only:

- `tag_distributions`
- `combo_distributions`
- `distribution_total_samples`

These structures must not be repurposed to include extension label counts.

### 3. Extension stats remain parallel

`extension_stats` continues to be stored as a separate parallel structure, not merged into core label stats.

### 4. Historical runs remain readable

Old runs without extension rarity must continue to:

- load without migration,
- render dashboards,
- support `refresh-rarity`,
- preserve old thresholds and score expectations.

### 5. Recompute must be reproducible

Any V2 rarity result must be reproducible offline from persisted stats/config metadata.

---

## Recommended Architecture

## High-level model

Introduce a **dual-channel rarity architecture**:

1. **Core channel** — unchanged, canonical.
2. **Extension channel** — parallel, per-spec rarity.
3. **V2 blended channel** — optional, versioned, opt-in.

### Output families

#### Legacy / canonical
- `value.rarity.score`
- `value.value_score`
- `value.selection_score`

#### New V2 fields
- `value.rarity_core`
- `value.rarity_extension`
- `value.rarity_v2`
- `value.value_score_v2`
- `value.selection_score_v2`
- optional conversation analogs:
  - `conv_rarity_extension_v2`
  - `conv_selection_extension_v2`

Legacy fields remain the default source of truth. V2 fields are additive.

---

## User-facing Modes

Add a new config/CLI enum:

```text
extension_rarity_mode = off | preview | bonus_only
```

### `off` (default)
- no extension rarity computation
- no score impact
- current behavior preserved

### `preview`
- compute extension rarity
- expose it in stats / dashboard / export
- do **not** affect any score fields
- is the recommended first user-facing rollout mode

### `bonus_only`
- compute extension rarity
- compute `rarity_v2`
- compute `value_score_v2`
- compute `selection_score_v2`
- keep legacy score fields unchanged
- should ship as experimental/advanced until preview validation is clean

### Explicitly excluded for v1
- `replace`
- `penalty_only`
- `full_reweight`

The first stable rollout should support only `off`, `preview`, and `bonus_only`.

---

## Data Model

## Sample-level `value`

Recommended structure:

```json
{
  "rarity": { "score": 6.8 },
  "rarity_core": {
    "score": 6.8,
    "tag_rarity": 6.2,
    "combo_rarity": 7.1,
    "stats_ref": {"path": "stats_labeling.json", "mode": "absolute"}
  },
  "rarity_extension": {
    "mode": "preview",
    "score": 7.4,
    "confidence": 0.79,
    "matched_specs": 1,
    "specs": {
      "ui_web_analysis_example": {
        "spec_version": "v1",
        "spec_hash": "sha256:...",
        "status": "success",
        "matched": true,
        "baseline_total": 1240,
        "score": 7.4,
        "confidence": 0.79
      }
    }
  },
  "rarity_v2": {
    "score": 7.1,
    "core_score": 6.8,
    "extension_bonus": 0.3,
    "blend_mode": "bonus_only"
  },
  "value_score": 7.55,
  "value_score_v2": 7.60,
  "selection_score": 7.2,
  "selection_score_v2": 7.3,
  "intra_class_rank": 6.9,
  "intra_class_rank_v2": 6.9
}
```

### Notes

- `rarity` remains for backward compatibility.
- `rarity_core` makes the canonical old result explicit.
- `rarity_extension` contains the aggregate extension rarity plus per-spec details.
- `rarity_v2` is only present when `extension_rarity_mode == bonus_only`.
- `intra_class_rank_v2` can equal the legacy `intra_class_rank` because extension labels are not used for grouping.
- preserve the current `stats_ref` contract keys for legacy/core rarity; add new keys only, never replace old provenance fields.

---

## Extension Rarity Computation

## Per-spec isolation

Extension rarity is computed **within each spec**, never across heterogeneous specs.

Each spec is keyed by:

- `spec_id`
- `spec_version`
- `spec_hash`

A different `spec_hash` is treated as a different rarity space.

### Why

User-defined extension schemas are not globally comparable. Different prompt wording, field design, or option sets can radically change the distribution. The baseline must therefore remain spec-local.

---

## Baseline population

For a spec `S`, define:

- `eligible_total`: samples where `matched == true`
- `baseline_total`: samples where `matched == true && status == "success"`

Only `baseline_total` participates in rarity math.

### Excluded from baseline
- `skipped`
- `failed`
- `invalid`
- missing/partial extension outputs

### Why

These statuses are routing/operational outcomes, not evidence-bearing extension labels. Including them would contaminate rarity semantics.

---

## Per-field rarity

For each successful matched sample and for each field in the spec:

### `enum`
Compute field value IDF:

```text
idf(value) = max(0, log2(N / (count + 1)))
```

where:
- `N = baseline_total`

### `multi_enum`
Count each selected item independently.

At sample time, aggregate selected option rarity with the same conservative style already used in core multi-select rarity:

- a bounded `max + mean` blend
- no all-or-nothing winner-take-all

### Confidence shrinkage
For each field:

```text
field_rarity = conf * observed_idf + (1 - conf) * field_prior
```

where:
- `conf` = field confidence from extension output,
- `field_prior` = prior mean rarity for that field’s distribution.

### Spec confidence
Aggregate field confidence into one `spec_confidence` using the mean of populated-field confidences, then apply a missing-field penalty:

```text
spec_confidence = mean(populated_field_confidences) * populated_field_ratio
```

If no field confidence is present, fall back to a conservative floor rather than assuming certainty.

### Missing / empty fields
Missing field values do not contribute positive rarity. They instead reduce `populated_field_ratio`, which lowers `spec_confidence`.

---

## Spec-level sample rarity

For one sample under one spec:

```text
spec_rarity = weighted_mean(field_rarity_i)
```

### Default weighting
Use **uniform field weights** in the initial implementation.

### Future expansion (not required now)
Spec-local rarity weights may be added later, but they are not part of this design’s first implementation.

---

## Extension combo rarity

### Decision
**Do not implement extension combo rarity in V2 initial rollout.**

### Reasoning
- It is highly sparse.
- It is unstable under schema edits.
- It is not comparable across spec versions.
- It increases complexity without proven signal gain.

### Explicit prohibition
- do not mix extension fields into core combo keys
- do not add spec-local combo rarity in the first release

---

## Multi-spec aggregation

If a sample has multiple successful matched extension payloads:

```text
rarity_extension.score = weighted_mean(normalized_spec_rarity_i)
```

### Normalization contract
Each spec’s raw rarity must be normalized onto the shared 1–10 rarity scale **before** multi-spec aggregation.
Use the same normalization mode family as core rarity (`absolute` or `percentile`), but computed inside the spec-local baseline.

#### Absolute mode
Use the same mapping contract as core rarity, but with the spec-local `baseline_total` as the sample count ceiling:

```text
raw_ceiling = log2(max(baseline_total, 2))
normalized_score = 1 + clamp(raw_score, 0, raw_ceiling) / raw_ceiling * 9
```

#### Percentile mode
Rank the sample’s raw spec rarity within the successful-matched population of that spec and map percentile to 1–10, exactly like core percentile mode.

Recommended weight per spec:

```text
spec_weight_i = spec_confidence_i * support_gate_i * config_weight_i
```

where:
- `spec_confidence_i` is the sample-level extension rarity confidence for that spec,
- `support_gate_i` becomes 0 when the spec baseline is too small,
- `config_weight_i` defaults to `1.0`.

A spec with insufficient support contributes zero weight.

---

## Blending Rule

## Bonus-only blend

When `extension_rarity_mode == bonus_only`:

```text
rarity_v2 = clamp(core_rarity + extension_bonus, 1, 10)
```

Where:

```text
extension_bonus
= λ * extension_gate * max(0, rarity_extension.score - 5)
```

### Recommended defaults
- `λ = 0.10`
- `bonus_cap = 0.50`
- `min_extension_baseline_total = 200`

### Support gate
Use a single support gate derived from the spec-local `baseline_total` (which already means `matched && success`).
Do not introduce a second threshold over the same population.

### Extension gate
`extension_gate` is the final aggregate gate derived once from:
- aggregate support sufficiency
- aggregate extension confidence
- baseline source eligibility

Do not multiply support/confidence gates twice at both per-spec weighting time and final bonus time.

### Important restrictions
- extension rarity can only add a **bounded bonus**
- extension rarity cannot apply a negative penalty
- no extension bonus if support is insufficient
- no extension bonus if the spec did not match successfully
- `bonus_only` may only apply a non-zero bonus when the baseline source is eligible for ranking use

### Why bonus-only is safer
This prevents a sample from being unfairly downgraded just because it did not match a user-defined extension or because the extension’s trigger is narrow.

---

## Value and Selection V2

## `value_score_v2`

When `bonus_only` is enabled:

```text
value_score_v2 = compute_value_score(score_result, rarity_v2)
```

Legacy `value_score` remains untouched.

---

## `selection_score_v2`

### Key rule
Do **not** use extension labels for percentile grouping.

### What remains unchanged
- `_SELECTION_DIMS`
- `RARITY_WEIGHTS`
- intra-class rank grouping over core labels

### V2 behavior
Reuse the core intra-class rank, but let the final selection composition consume `rarity_v2` instead of legacy rarity:

```text
selection_score_v2 = compose_selection_score(
  intra_class_rank_core,
  pure_quality,
  rarity_v2
)
```

This preserves grouping stability while allowing extension rarity to act as a capped diversity nudge.

---

## Conversation-level V2

Add parallel conversation fields with names that do not collide with the existing trajectory-scoring `conv_selection_v2` semantics:

- `conv_rarity_extension_v2`
- `conv_selection_extension_v2`

### Rule
Conversation extension-V2 fields should be computed exactly like current conversation rarity/selection, but sourcing rarity from sample-level `rarity_v2` instead of legacy `rarity.score`.

Legacy conversation fields remain unchanged.

---

## Pass 2 Prompt Isolation

## Required prerequisite

Before any extension rarity implementation ships, ensure that `label_extensions` is not silently serialized into the current Pass 2 scoring prompt.

### Reason
If extension labels already reach the scoring LLM prompt, then adding deterministic extension rarity would cause a double influence path:

1. prompt-conditioned effect
2. deterministic rarity effect

That would make the rollout statistically ambiguous and difficult to validate.

### Rule
- current Pass 2 prompt version must remain **extension-blind** by default
- any future extension-aware scoring prompt must be a separate, explicit, versioned change

---

## Persisted Stats

## Pass 1 stats additions

Store reproducible extension rarity baselines under:

```text
extension_stats.specs.<spec_id>.baselines.<spec_hash>
```

The existing `extension_stats.specs.<spec_id>` summary may remain for dashboard readability, but reproducible rarity baselines must be fingerprinted by `spec_hash` (and include `spec_version`). Mixed-hash summaries must either preserve separate baseline buckets or be explicitly marked non-recomputable for V2.

Recommended structure:

```json
{
  "spec_version": "v1",
  "spec_hash": "sha256:...",
  "config": {...},
  "total": 3000,
  "matched": 1200,
  "success": 1100,
  "invalid": 30,
  "failed": 20,
  "skipped": 1800,
  "baseline_total": 1100,
  "field_value_distributions": {...},
  "field_presence_counts": {...},
  "confidence_stats": {...},
  "rarity_config": {
    "field_weight_mode": "uniform",
    "multi_value_aggregation": "max_mean_blend",
    "confidence_shrinkage": true,
    "normalization_mode": "absolute"
  }
}
```

### Notes
Current `field_distributions` may be retained for dashboard compatibility, but `field_value_distributions` / `field_presence_counts` should be explicit enough for deterministic offline recomputation.

### Baseline source precedence
For extension rarity:
1. explicit external/shared extension-rarity stats
2. embedded pass1 summary stats with hash-consistent baselines
3. local fallback built from the current input (preview only)

`bonus_only` must not apply a non-zero bonus from a local fallback baseline. If only a local baseline is available, the run should downgrade to diagnostic preview behavior for extension rarity.

---

## Pass 2 stats additions

Persist the exact V2 scoring contract used for recompute reproducibility:

```json
{
  "extension_rarity_config": {
    "enabled": true,
    "mode": "bonus_only",
    "blend_weight": 0.10,
    "bonus_cap": 0.50,
    "min_extension_baseline_total": 200,
    "affects": ["rarity_v2", "value_score_v2", "selection_score_v2"],
    "specs_used": [
      {
        "spec_id": "ui_web_analysis_example",
        "spec_version": "v1",
        "spec_hash": "sha256:..."
      }
    ]
  }
}
```

---

## Dashboard / Export / UX

## Dashboard

Always distinguish:

- **Core rarity**
- **Extension rarity (preview)**
- **Rarity V2**

Never label extension rarity simply as “rarity”.

### Required provenance display
For extension rarity cards/columns, show:
- spec id
- spec version/hash
- matched sample count
- baseline total
- normalization mode
- baseline source

### Small-baseline warning
If support is weak, show a visible warning:

> Diagnostic only — this preview is built from a low-volume or local baseline and is not safely comparable across runs.

---

## Review export

Add a separate opt-in switch for V2 rarity/export columns. Suggested columns:

- `rarity_core_score`
- `extension_rarity_preview_score`
- `extension_rarity_preview_confidence`
- `rarity_v2_score`
- `value_score_v2`
- `selection_score_v2`
- per-spec extension rarity columns where useful

This should not silently piggyback on `--include-extensions` without an explicit user decision.

---

## Launcher / CLI

### New CLI flag

```text
--extension-rarity-mode off|preview|bonus_only
```

User-facing launcher wording for `bonus_only` should mark it as experimental, e.g. `Experimental V2 bonus mode (writes new V2 scores only; legacy ranking unchanged)`.

Default:

```text
off
```

### Optional future explicit baseline flag

```text
--extension-rarity-stats <stats.json>
```

This keeps baseline provenance explicit instead of silently inferring behavior.

### Launcher behavior
Expose extension rarity as an advanced / opt-in scoring option with explanatory wording:
- preview only
- does not change current value/selection scores unless `bonus_only`
- `bonus_only` writes new V2 fields only
- local/inferred baselines are diagnostic-only
- if extension rarity is computed but gated out, say explicitly that it was not applied to V2 scores

---

## Refresh / Recompute Semantics

`refresh-rarity` must support rebuilding:

- `rarity_core`
- `rarity_extension`
- `rarity_v2`
- `value_score_v2`
- `selection_score_v2`

without re-running Pass 1 or Pass 2 LLM calls.

### Important compatibility rule
Legacy score fields are not rewritten unless the user explicitly requests a legacy refresh path. V2 refresh should remain additive. In `off` mode, no new V2 fields should be written at all, so the output schema remains byte-for-byte compatible for legacy-only runs.

---

## Migration and Compatibility Strategy

## One-pass implementation, non-breaking activation

The architecture can be implemented in one pass, provided activation remains conservative:

1. ship the full data model,
2. ship preview + bonus_only modes,
3. keep default mode `off`,
4. keep legacy score fields canonical,
5. keep V2 fields parallel and opt-in.

This is “one-pass architecture completion” without semantic breakage.

---

## Known Failure Modes to Guard Against

1. **Silent overwrite of core semantics**  
   Accidentally reusing current rarity/value/selection fields.

2. **Spec baseline contamination**  
   Mixing different `spec_hash` / `spec_version` distributions into one rarity space.

3. **Prompt leakage + deterministic double counting**  
   Allowing extension labels into Pass 2 prompt while also blending extension rarity.

4. **Selection fragmentation**  
   Letting extension labels participate in percentile grouping.

5. **Non-reproducible recompute**  
   Failing to persist spec hash, baseline counts, and blend config.

---

## Final Recommendation

Implement **Extension Rarity V2** as:

- a fully versioned parallel scoring architecture,
- default-off,
- preview-first,
- bonus-only when enabled,
- no replacement of current semantics,
- no extension combo rarity,
- no extension grouping in selection,
- and with Pass 2 prompt isolation as a prerequisite.

This gives the project the “one-pass final architecture” the user wants, while preserving the current scoring contract and avoiding migration hazards.
