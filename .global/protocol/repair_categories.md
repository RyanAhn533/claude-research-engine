# Protocol: Repair Categories (v2)

> Three repair categories. Each has its own retry budget and escalation rule.
> Authority: `ARCHITECTURE.md` §5. Failure rows go to `engine/schemas/paper_tried_entry.schema.json` with `failure_category`.

## Why categorize

v1 retried any failure up to 3 times with the same strategy. This wastes compute on `dynamics` failures — same hyperparameters → same gradient explosion. v2 splits failures by what the **fix** looks like, not by what the **error** looks like.

## Categories

### Syntactic
- **Examples**: `ImportError`, `ShapeMismatch`, `TypeError`, `IndexError`, missing CLI arg
- **Why retry**: the design is fine; the code is broken locally
- **Retry budget**: 3
- **Strategy**: `code_diff_patch` — Repair Planner reads the traceback, proposes a minimal diff, Gate B reruns
- **Escalation if exhausted**: → `POSTMORTEM` → `METHOD_SEARCH`

### Semantic
- **Examples**: wrong loss function, incorrect metric aggregation, off-by-one in indexing, `do_sample=False` with `temperature`, wrong tokenizer
- **Why retry**: design.md is correct but implementation diverges from design intent
- **Retry budget**: 2
- **Strategy**: `design_review_then_patch` — Repair Planner is **required** to re-read `design.md` before proposing a fix. Diffs are validated against the design's pseudocode.
- **Escalation if exhausted**: → `POSTMORTEM` → `METHOD_SEARCH`

### Dynamics
- **Examples**: gradient explosion, NaN loss, non-convergence, mode collapse, ICL prompt instability across seeds
- **Why retry only once**: rerunning the same training dynamics with the same seed/config will give the same result. New information requires a config or architectural change.
- **Retry budget**: **1**
- **Strategy**: `hyperparameter_or_architectural_change` — Failure Analyzer **must** be invoked first; the proposed change must touch at least one of {learning rate, init scheme, warmup, gradient clipping, layer count, regularizer}.
- **Escalation if exhausted**: → directly `METHOD_SEARCH` (skip POSTMORTEM-only path; the postmortem is written as part of the escalation).

### Protocol (terminal — no retry)
- **Examples**: data leakage detected by Leakage Auditor, baseline drift, metric_name changed mid-iter, seed count below minimum
- **Retry budget**: **0**
- **Strategy**: none. Gate A halts the engine.
- **Escalation**: `HALT`. Requires human resolution before any restart.

## Logging

Each repair attempt appends a row to `paper_tried.jsonl` with:
- `failure_category` ∈ {syntactic, semantic, dynamics, protocol}
- `retryable_with_changes` listing the config knobs that, if changed, justify a future retry
- `blocked_configs` listing exact `(method_id, config_fingerprint)` pairs that must not be re-tried

`method_globally_blocked: true` is set **only by humans** — never by the Repair Planner.

## Implementation note

`engine/core/retry_repair_router.py` reads the Gate B failure record, classifies into one of the four categories, looks up the budget, and decides:
- retry the same phase with a patched run.py, OR
- escalate to METHOD_SEARCH with a `paper_tried` row pre-filled, OR
- HALT.
