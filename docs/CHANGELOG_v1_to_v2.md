# CHANGELOG — v1 → v2

**Adopted**: 2026-05-22  
**Trigger**: external review (52/70) identifying v1 architecture's enforcement gap

---

## Why this migration was necessary

v1 had a 5-phase loop documented in prose. The external review identified 4 critical failure modes that v1 could not prevent because **the protocol lived in prompts, not in code**:

1. **Auto mode ↔ HUMAN_APPROVAL conflict** — HANDOFF said "auto mode, don't ask permission"; METHODOLOGY said "destructive actions need consent". Both true, both unenforced.
2. **Gate C used unpaired comparison** — `bootstrap_CI_lower > baseline_mean` is statistically the wrong test for paired experiments. Reviewers would catch this in seconds.
3. **No claim ↔ exp linkage** — paper narrative ("LoRA universal, ICL modality-gated") was prose in HANDOFF, not data. Drift was undetectable.
4. **SELF_ATTACK was unconditional** — running 3 max-thinking adversaries on every experiment was wasteful for exploration runs.

v2 solves all four through schemas + hooks + paired statistics + claim registry.

---

## Schema changes

### NEW schemas (2)

#### `claim.schema.json`
Links experiments to specific paper claims. Required for paper_ready experiments. Solves *narrative drift*.

```json
{
  "claim_id": "C1_lora_universal",
  "statement": "LoRA produces universal gains; ICL is modality-gated.",
  "status": "active",
  "supporting_exps": ["exp_024_iemocap_lora_multiseed", ...],
  "contradicting_exps": [],
  "required_for_submission": true,
  "venue": "NeurIPS 2026",
  "self_attack_required": true
}
```

If `required_for_submission=true`, schema forces `self_attack_required=true` AND a non-null `venue`. No more orphan claims.

#### `permission.schema.json`
22 default action classes mapped to one of `auto_safe` / `human_gate_required` / `hard_block`. Solves *auto mode conflict*.

Default seeded by `engine.core.permission_policy.seed_default_policies()` on first bootstrap.

| auto_safe (8) | human_gate_required (9) | hard_block (5) |
|---|---|---|
| read, analysis, design_draft, local_experiment_run, local_result_parse, local_report_generation, jsonl_append, schema_validate | git_push, branch_delete, rm, overwrite, baseline_replacement, paper_claim_update, claim_status_change, public_release, expensive_full_run | git_force_push, global_method_block, in_place_jsonl_edit, schema_modification, kill_switch_remove |

`hard_block` is unoverridable even by JY.

### MODIFIED schemas (3)

#### `experiment.schema.json`
**Added** paired-statistics fields (Gate C):
- `paired_delta_mean`
- `paired_delta_ci_95` (bootstrap 95% CI)
- `cohen_d_paired`
- `wilcoxon_p`
- `tost_passed`
- `linked_claims` (array of claim_id)
- `manifest_ref` (required)
- `self_attack_run`, `self_attack_concerns`

**Conditional requirement**:
```
if stage == "paper_ready":
  required: n_seeds≥7 + paired_delta_ci_95 + cohen_d_paired + wilcoxon_p + linked_claims
if stage == "prototype":
  required: n_seeds≥3
```

**Removed concept**: `bootstrap_CI_lower > baseline_mean` decision rule. v2 uses paired delta CI lower > 0.

#### `paper_tried_entry.schema.json`
**Was**: deduplication by method name string. *"AU 8-node graph" failed once → permanently blocked* even if a different config might have worked.

**Now**:
```
dedup key = (method_id, config_fingerprint)
config_fingerprint = sha256(canonical(config_dict))
```

Same `method_id` + different config = retryable (with justification).  
`method_globally_blocked: true` is a separate, human-only field.

#### `hypothesis.schema.json`
**Was**: optional yaml block in `design.md`.

**Now**: separate schema. `falsifiability_check ∈ {PASS, FAIL}`. FAIL blocks HUMAN_APPROVAL. Outcome recorded as new row with `supersedes`, not in-place edit.

### Unchanged (3)

`state.schema.json`, `task.schema.json`, `reproducibility_manifest.schema.json` — same semantics, added `supersedes`/`prev_hash`/`row_id` for hash chain.

---

## Phase flow changes

### NEW phases (4)

```
STATUS_SYNC                  — file-system state beats HANDOFF.md prose
CLAIM_LINKING                — exp → paper claim id (mandatory for paper_ready)
CONDITIONAL_SELF_ATTACK      — was SELF_ATTACK (unconditional). Now triggered by:
                               required_for_submission OR baseline_candidate OR
                               novelty_claim OR reviewer_risk≥medium OR
                               contradicts_previous_narrative
PAPER_QUEUE_UPDATE           — bump claim_registry status after LOGGING
```

### Loop comparison

```
v1                            v2
──                            ──
BOOTSTRAP                     BOOTSTRAP
METHOD_SEARCH                 STATUS_SYNC               [NEW]
HYPOTHESIS_REGISTRATION       METHOD_SEARCH
HUMAN_APPROVAL                HYPOTHESIS_REGISTRATION
IMPLEMENTATION                HUMAN_APPROVAL
EVALUATION                    IMPLEMENTATION
SELF_ATTACK     (always)      EVALUATION
LOGGING                       CLAIM_LINKING             [NEW]
INSIGHT_UPDATE                CONDITIONAL_SELF_ATTACK   [CHANGED]
                              LOGGING
                              PAPER_QUEUE_UPDATE        [NEW]
                              INSIGHT_UPDATE
```

---

## New runtime components

### `engine/core/append_only_logger.py`
The spine. Every jsonl write goes through `AppendOnlyLog`:
- schema-validates before write
- assigns `row_id` + `prev_hash` (sha256 chain)
- detects in-place edits via size + head-bytes hash
- serializes concurrent writers via `fcntl.flock`
- `verify_chain()` walks the file and checks every link

Raises `AppendOnlyViolation` on any breach. The engine halts on this exception.

### `engine/core/permission_policy.py`
Reads `permission_policy.jsonl`, resolves action classes through `supersedes`, exposes:

```python
is_action_allowed(
  policy_log_path,
  action_class,
  *, auto_mode, human_consent_given
) -> (allowed, reason)
```

Bootstrap seeds 22 defaults. JY can amend by appending new rows.

### `engine/core/hashing.py`
- `config_fingerprint(config_dict)` — deterministic sha256 for paper_tried dedup
- `row_hash(row)` — for the chain
- `data_hash(bytes)` — for reproducibility_manifest

### `engine/cli/jy.py`
Single CLI entry. Subcommands:
- `bootstrap --project <NAME> --prefix <ABC>`
- `status --project <NAME>`
- `validate [--project <NAME>]`
- `verify-chain --project <NAME> --log <leaderboard|paper_tried|...>`
- `seed-policy --project <NAME>`

### `.claude/` — Claude Code integration
- `settings.json` — hooks registered
- `commands/` — `/status`, `/validate`, `/iter`, `/attack`, `/log`
- `hooks/block_jsonl_in_place.sh` — PreToolUse hook on Edit/Write
- `hooks/require_human_gate.sh` — PreToolUse hook on Bash, blocks destructive

---

## What still needs implementing (after this commit)

### Tier 2 — closed-loop runtime
- `state_machine.py` (FSM with budget pre-check, kill switch poll, supersedes for state rows)
- `task_dag.py` (intra-phase DAG)
- `compute_budget.py` (GPU/cost/wall-clock cap)
- `reproducibility_manifest.py` (writer, env_lock hash, git status)
- `leakage_auditor.py` (6 checks from ARCHITECTURE.md §6.5)
- `engine/gates/{a,b,c}.py` (gate logic with paired-delta Gate C)
- `engine/agents/*.md` (6 agent prompts: method_planner, failure_analyzer, repair_planner, novelty_critic, reviewer_simulator, insight_summarizer)

### Tier 3 — differentiation
- Reviewer Simulator full implementation
- `analysis/dpo_pair_extractor.py`
- `analysis/hindsight_scoring.py`
- Gate D research-value advisory output

---

## Migration safety

- v1 jsonl rows remain valid against v2 schemas **if** they include the new required fields. For older rows that don't, the validator will fail — fix by writing a `supersedes` row that adds the missing fields. No in-place edits.
- v1 `paper_tried.jsonl` rows without `config_fingerprint` will fail v2 schema. The migration option is either (a) backfill via `git log` of the run scripts, or (b) accept the file as v1-frozen and start v2's paper_tried_v2.jsonl fresh. The former is honest, the latter is faster.
- v1 `directions.jsonl` and `logic_chains.jsonl` are unchanged. Hindsight scoring continues.
- v1 backups land in `.v1_backup/` after `install_v2.sh`.

---

## Acceptance criteria (this commit)

Run:
```bash
python engine/tests/validate_schemas.py
```

Expected: **24/24 pass** (8 schemas self + 8 valid fixtures + 8 invalid fixtures rejected).

Then:
```bash
python -m engine.cli.jy bootstrap --project 99_test --prefix TST
python -m engine.cli.jy status --project 99_test
python -m engine.cli.jy validate --project 99_test
rm -rf projects/99_test
```

Expected: bootstrap creates state files, status shows phase=BOOTSTRAP iter=0, validate passes chain on the freshly-seeded permission policy + engine_state.

If both pass, v2 spine is live. Tier 2 work can proceed without breaking Tier 1.

---

## Appendix: Tier 2 commit (2026-05-22)

Tier 2 ships the **closed-loop runtime** modules. Schemas already frozen in Tier 1 stay untouched.

### Added Python modules

| Module | What it does |
|---|---|
| `engine/core/state_machine.py` | Phase FSM. 16 phases. `LEGAL_TRANSITIONS` dict + per-phase pre-conditions. `can_transition(target)` returns (ok, reason) so Claude can't cheat. |
| `engine/core/compute_budget.py` | `check()`, `record_usage()`, `per_experiment_check()`. Budget is dict on `state.budget` field. |
| `engine/core/reproducibility_manifest.py` | Auto-collects git_sha + git_dirty + nvidia-smi + nvcc + env_lock hash. `data_hash` is the only required caller input. |
| `engine/core/leakage_auditor.py` | 3 auto checks (set intersection, CV grouping, MI upper bound) + 3 declarations parsed from `design.md`'s `leakage_audit:` block. |
| `engine/gates/gate_c.py` | **External review §3 problem 4 fix.** `paired_delta_bootstrap_CI_lower > 0 AND cohen_d > 0.5 AND wilcoxon_p < 0.05` for paper_ready. Prototype = directional only. |

### Added agent prompts (`engine/agents/`)

`method_planner.md`, `failure_analyzer.md`, `repair_planner.md`, `novelty_critic.md`, `reviewer_simulator.md`, `method_skeptic.md`, `insight_summarizer.md`

Each is ≤2K-token contract spec with `runs_as`, `thinking_budget`, `output_target` in YAML frontmatter.

### Acceptance test

```bash
python engine/tests/test_tier2.py
```

7 steps. All must pass:
1. `jy bootstrap` creates project state cleanly
2. `state_machine` enforces legal transitions and refuses skips
3. `compute_budget` exhausts at 110%, accumulates immutably
4. `reproducibility_manifest` generates schema-valid output, refuses empty data_hash
5. `leakage_auditor` passes clean audit, fails dirty audit with correct failure list
6. `gate_c` prototype directional / paper_ready statistical (scipy required)
7. cleanup

### What still needs implementing (Tier 3 → on demand)

- `engine/gates/gate_a.py`, `gate_b.py` (Gate A = protocol consistency, Gate B = leakage_auditor + smoke). Stubs exist; full logic comes when projects first need them.
- `engine/core/task_dag.py` — intra-phase DAG. So far each phase has been a single linear sequence in practice. DAG abstraction added when a phase actually needs parallel tasks beyond what subagent fan-out covers.
- DPO pair extractor, hindsight scoring CLI — wait for the first 50+ Direction rows.
