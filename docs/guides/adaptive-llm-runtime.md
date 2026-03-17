# Adaptive LLM Runtime

`sft-label` can run LLM calls in an adaptive mode that is designed for unstable
or overloaded LLM endpoints.

In adaptive mode, the pipeline monitors recent request outcomes (timeouts,
overload responses, malformed responses) and automatically adjusts pressure:

- lowers effective concurrency and RPS when the endpoint degrades
- pauses new inference briefly during outages, then probes for recovery
- gradually ramps back up when the endpoint becomes healthy again

This complements per-request retry logic by adding system-level backpressure so
short outages do not amplify into large batches of failures.

## CLI Flags

Adaptive runtime is enabled by default. You can override per run:

- `--adaptive-runtime` / `--no-adaptive-runtime`
- `--recovery-sweep` / `--no-recovery-sweep`

When adaptive runtime is enabled, `--concurrency` and `--rps-limit` are treated
as caps (maximums). The runtime may run below those caps to protect the endpoint
and improve overall completion rate.

## End-of-Phase Recovery Sweep

Recovery sweep is enabled by default. At the end of Pass 1 and Pass 2, the
pipeline can optionally retry samples that failed for infra-retryable reasons
(for example: timeouts, 429/503, transient network failures).

The intent is to improve labeling/scoring completion rate without re-running the
entire dataset.

## Configuration (Library Mode)

If you are using the library interface, advanced adaptive thresholds are
available via `PipelineConfig`. The interactive launcher keeps the common path
simple and only exposes the boolean toggles.

