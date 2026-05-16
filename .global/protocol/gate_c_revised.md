# Protocol: Gate C — Statistical Rigor (v2)

> Two stages. Prototype iteration is fast and directional. Paper-submission window is rigorous.
> Authority: `ARCHITECTURE.md` §3.1. Companion: `engine/schemas/gate_result.schema.json` and `experiment.schema.json::statistical_evidence`.

## Stages

### `prototype` (default for day-to-day iteration)

```yaml
seed_min: 3
required_metrics: [mean, std]
decision: directional_only
```

- No statistical claim. `baseline_candidate: true` is allowed but does **not** trigger baseline replacement.
- Output verdict: `pass` / `neutral` / `fail` based on `mean ± std` ordering only.
- Cheap. Used while exploring methods.

### `paper_ready` (escalated near submission)

```yaml
seed_min: 7
required_metrics: [mean, std, bootstrap_95_CI, cohen_d]
statistical_tests:
  - paired_comparison_with_baseline
  - wilcoxon_signed_rank
equivalence_test:
  framework: TOST
  delta_threshold: pre_registered    # from hypothesis_registry
decision_rule: |
  improvement_candidate IFF:
    bootstrap_CI_lower > baseline_mean
    AND cohen_d > 0.5
    AND wilcoxon_p < 0.05
```

- Required before any leaderboard row can be promoted to `baseline_candidate: true` AND replace the working baseline.
- `delta_threshold` comes from the **pre-registered** `hypothesis_registry.jsonl` entry. Post-hoc threshold tuning is `protocol_violation`.

## Switching stages

A project switches stage via `rules/gate_c_stage.txt` containing one line (`prototype` or `paper_ready`). Changing it after experiments have started is allowed but logged. Re-running prior experiments under the new stage is encouraged before submission.

## Why TOST, not just t-test

The interesting question for "did this improvement hold up?" is often **"is the effect at least as large as our pre-registered threshold?"** — an equivalence question, not a difference question. TOST (two one-sided tests) bounds it directly. Required `delta_threshold` forces honest pre-registration.

## Why Wilcoxon

Paired emotion-recognition metrics are often heavy-tailed across seeds (seed variance is bimodal). Wilcoxon signed-rank is robust where the paired t-test inflates Type I.

## Reporting

`experiment.schema.json::statistical_evidence` carries the numbers:

```json
{
  "bootstrap_95_CI": [0.412, 0.456],
  "cohen_d": 0.83,
  "wilcoxon_p": 0.018,
  "tost_passed": true,
  "delta_threshold": 0.02
}
```

Reviewer Simulator can read these directly.

## Failure modes (handled by Gate A, not C)

Changing metric mid-iter, dropping a seed because it looks like an outlier, swapping baselines without superseding rows — these are **Gate A protocol violations**, not Gate C neutral verdicts. Gate C never sees them; Gate A halts first.
