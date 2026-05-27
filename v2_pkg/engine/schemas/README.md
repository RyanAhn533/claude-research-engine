# engine/schemas — frozen v2.0

JSON Schema draft-07. **Once committed, do not edit in place. Migrate via versioned new file.**

## 8 schemas

| File | Purpose | Append target |
|------|---------|---------------|
| `state.schema.json` | Engine FSM state (current phase, budget, kill switch) | `state/engine_state.jsonl` |
| `task.schema.json` | DAG node within a phase | `state/task_log.jsonl` |
| `experiment.schema.json` | Leaderboard row. Gate C paired-delta fields. | `state/leaderboard.jsonl` |
| `hypothesis.schema.json` | Pre-registration (Phase 1.5) | `state/hypothesis_registry.jsonl` |
| `reproducibility_manifest.schema.json` | git sha + data hash + env lock | `reproducibility_manifests/exp_NNN.yaml` |
| `paper_tried_entry.schema.json` | Config-aware dedup (NOT method-name dedup) | `state/paper_tried.jsonl` |
| `claim.schema.json` | Paper claim ↔ exp linkage (NEW v2.0) | `state/claim_registry.jsonl` |
| `permission.schema.json` | Auto mode vs HUMAN_APPROVAL policy (NEW v2.0) | `state/permission_policy.jsonl` |

## Key invariants enforced by schemas

1. **Append-only** — every schema has `supersedes` and `prev_hash`. In-place edits are out of contract.
2. **Config-aware dedup** — `paper_tried_entry` requires `config_fingerprint`. Same `method_id` with different config is NOT a duplicate.
3. **Human-only fields** — `paper_tried_entry.method_globally_blocked=true` requires `blocked_by` (human name).
4. **Stage gating** — `experiment.stage=paper_ready` conditionally requires n_seeds ≥ 7 + CI + Cohen's d + Wilcoxon + linked_claims.
5. **Pre-registration** — `hypothesis.falsifiability_check=FAIL` is meant to block HUMAN_APPROVAL at runtime.
6. **Submission-grade claims** — `claim.required_for_submission=true` forces `self_attack_required=true` and a `venue`.
7. **Hard blocks** — `permission.default_policy=hard_block` cannot have `override_allowed=true`.

## Validation

```bash
python tests/validate_schemas.py
```

24 checks: 8 schemas (draft-07 self) + 8 positive fixtures + 8 negative fixtures.
All must pass. CI hook (Tier 1) will block commit on failure.

## Frozen at

Date: 2026-05-22
Adopted from: ARCHITECTURE.md §3–§6 + external review §5.2, §7.2 patches
Hash chain: enforced by engine/core/append_only_logger.py (next deliverable)
