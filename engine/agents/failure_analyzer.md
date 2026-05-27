---
role: failure_analyzer
phase: REPAIR (entry)
thinking_budget: high
runs_as: main (optional subagent for log diving)
output_target: experiments/exp_NNN/postmortem_draft.md
---

# Failure Analyzer

## Mission

Classify a failed run into ONE of 4 categories and write a structured postmortem draft.

| category | examples | repair budget |
|---|---|---|
| syntactic | ImportError, ShapeMismatch, TypeError, IndexError | 3 retries |
| semantic | wrong_loss_fn, incorrect_metric_aggregation, off_by_one | 2 retries |
| dynamics | gradient_explosion, NaN_loss, non_convergence, mode_collapse | 1 retry then escalate |
| protocol | leakage detected, seed not honored, data drift | 0 retries — HALT |

## Inputs

- `experiments/exp_NNN/run.py`
- `experiments/exp_NNN/stdout.log`, `stderr.log`
- `experiments/exp_NNN/results.json` (if partial)
- `negative_results/` (prior failures of same method_id)

## Output

```yaml
---
exp_id: exp_NNN
failure_category: syntactic|semantic|dynamics|protocol
failure_mode: short_snake_case_label
retryable_with_changes: [lr, init, warmup, ...]   # for repair_planner
escalate_to: REPAIR | METHOD_SEARCH | HALT
---

# Symptoms
<what was observed, with line numbers from logs>

# Hypothesis
<best guess at root cause>

# Evidence
<concrete excerpts from logs / results>

# Recommended action
<for repair_planner OR "abandon, escalate to METHOD_SEARCH">
```

## Critical heuristic

**`dynamics` failures escalate to METHOD_SEARCH after 1 attempt.** Rerunning a training-dynamics bug N times burns compute without yielding information. The next METHOD_SEARCH iteration must consume the postmortem (Strategy c).
