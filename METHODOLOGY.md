# JY Research Engine — Methodology (v2)

> Operating manual. Load first when a new Claude session opens this repo.
>
> Architecture spec → `ARCHITECTURE.md` · Tool-layer rules → `.global/protocol/claude_leverage_2026.md`
>
> v1 (5-phase loop) is preserved at `.global/protocol/engine_loop.md` for historical trace. Active engine is v2.

---

## 0. Core invariants (non-negotiable)

1. **LLM is a worker, not an orchestrator.** Phase transitions are decided by code (`engine/core/state_machine.py`), never by an LLM.
2. **Protocol lives in code and files.** If a rule cannot be expressed as a Gate check, it is a guideline, not a protocol.
3. **All records are append-only.** Corrections are new rows with `supersedes`. In-place edits to `*.jsonl` are detected and halt the engine.
4. **Non-reproducible runs are not experiments.** Every experiment has a `reproducibility_manifest`; missing manifest blocks LOGGING.
5. **Humans enter via named gates.** `HUMAN_APPROVAL` is the only point where Claude waits for explicit consent. No silent deletion.
6. **All rejects are advisory. Kills happen only on protocol violation.** Gate D (research-value) can flag but never block.
7. **Falsifiability is pre-registered.** Hypothesis with `falsifiability_check: FAIL` is rejected before HUMAN_APPROVAL.
8. **JY approval before destructive / shared-state actions.** Never push, never delete branches, never rm -rf without explicit consent.
9. **Prior-research search before proposing a new method / metric.** WebSearch first. Pattern: "X et al. 2024" beats "I propose".
10. **Claude does not score its own directions.** `hindsight_score` is JY-only. Self-censoring is also banned.

---

## 1. Engine loop (v2 phase flow)

The State Machine walks the phases below. Each phase has fixed entry preconditions, an intra-phase Task DAG, and exit postconditions. Spec: `.global/protocol/state_machine.md`.

```
BOOTSTRAP
  → METHOD_SEARCH
  → HYPOTHESIS_REGISTRATION       [pre-register falsifiable claim]
  → HUMAN_APPROVAL                [JY gate]
  → IMPLEMENTATION                [Gate B, repair router on fail]
  → EVALUATION                    [Gate C, prototype/paper_ready]
  → LOGGING                       [Gate D advisory, append-only writes]
  → INSIGHT_UPDATE (every 5 iter)
  → METHOD_SEARCH (loop)
```

### Phase 0 — BOOTSTRAP
Load `engine_state.json`. Check budget. Validate protocol files. Refuse to proceed if any check fails.

### Phase 1 — METHOD_SEARCH
Five strategies. At least one of each per 4 iterations (diversity is enforced by code, not by prompt):
- **a** Recent paper survey (WebSearch — required)
- **b** Adjacent-domain transplant
- **c** Failure-case analysis (reads `negative_results/`, not just `paper_tried.jsonl`)
- **d** Top-k hybrid / ensemble
- **e** First-principles redesign (new in v2)

Dedup is by `(method_id, config_fingerprint)` — not by method name alone. If the same `method_id` reappears with a new config, the Method Planner must justify why a different outcome is expected.

Output: 3 candidates → `experiments/exp_NNN/design.md` (draft).

### Phase 1.5 — HYPOTHESIS_REGISTRATION
For each candidate, write the `hypothesis` block in `design.md`:
- primary claim (falsifiable)
- null hypothesis
- `success_criterion.delta_threshold` (the TOST equivalence margin — pre-registered)
- failure implication

Append the row to `state/hypothesis_registry.jsonl`. The `falsifiability_check` field is computed automatically; `FAIL` blocks the candidate from HUMAN_APPROVAL.

### Phase 2 — HUMAN_APPROVAL
JY reviews 3 candidates + their hypotheses. Picks one, or rejects all → re-enter METHOD_SEARCH. No silent advance.

### Phase 3 — IMPLEMENTATION
1. Generate `reproducibility_manifest` first (git sha, data hash, env lock, random state).
2. Write `run.py`.
3. Run leakage audit (`engine/core/leakage_auditor.py`).
4. Unit / dry-run / smoke test.
5. Gate B verdict.
6. On fail → repair router classifies the failure into `syntactic` / `semantic` / `dynamics` / `protocol` and applies the matching budget (see `.global/protocol/repair_categories.md`).

### Phase 4 — EVALUATION
- `prototype` stage: ≥ 3 seeds, mean ± std, directional decision only.
- `paper_ready` stage: ≥ 7 seeds, bootstrap 95% CI, Cohen's d, Wilcoxon signed-rank, TOST against the pre-registered `delta_threshold`. Decision rule in `.global/protocol/gate_c_revised.md`.
- Project chooses stage via `rules/gate_c_stage.txt`.

### Phase 5 — LOGGING
- Append to `leaderboard.jsonl` (`engine/schemas/experiment.schema.json`).
- Append to `paper_tried.jsonl` (`engine/schemas/paper_tried_entry.schema.json` — config-aware).
- Update `hypothesis_registry.jsonl::outcome`.
- On failure → archive run artifacts to `negative_results/exp_NNN_failed/`.
- Reviewer Simulator runs → Gate D verdict (advisory).
- Direction IDs + Logic Chains appended to local + global (`methodology/directions.jsonl`, `methodology/logic_chains.jsonl`).

### Phase 6 — INSIGHT_UPDATE (every 5 iter)
Read the last 5 iterations. Extract works / doesn't-work patterns. Update `state/insights.md`. Bias next METHOD_SEARCH (Strategy c reads this).

---

## 2. Quality Gates

| Gate | Fires at | Authority | On fail |
|------|----------|-----------|---------|
| **A** Protocol | METHOD_SEARCH end, EVALUATION end | metric / split / seed / baseline comparability, leakage | `protocol_violation` → HALT |
| **B** Implementation | IMPLEMENTATION end | code runs, output well-formed, leakage audit | repair router (3 categories) |
| **C** Evaluation | EVALUATION end | statistical rigor at current stage | `neutral` or `failed` verdict |
| **D** Research Value | before LOGGING | novelty, ablation depth, reviewer defense | **advisory only — cannot block** |

Spec: `ARCHITECTURE.md` §3, `.global/protocol/gate_c_revised.md`.

---

## 3. Direction ID + Logic Chain (RLHF accumulation)

Unchanged in mechanism. v2 only adds: Direction rows include the `phase` and `gate` context, so DPO pairs can condition on phase. See `ARCHITECTURE.md` §11 Tier 3.

```json
{"id": "EMA-D042", "date": "2026-05-16", "phase": "METHOD_SEARCH",
 "context": "...", "direction": "...", "rationale": "...", "risk": "...",
 "accepted": true, "hindsight_score": null, "outcome": null}
```

`hindsight_score` is set by JY ≥ 7 days after. Claude never sets it (bias).

---

## 4. Experiment report format

`design.md` (one per experiment) follows `.global/protocol/report_format.md` 8-section template, **plus** the v2-required `hypothesis` block (Phase 1.5).

---

## 5. Numerical trust rules (kept from v1)

| ❌ Don't | ✅ Do |
|---------|-----|
| Silhouette / cos-sim alone for NO-GO | Linear probe accuracy vs random baseline (%p) |
| Raw high-dim distance metrics | L2 norm + PCA + Standardize first |
| Absolute values alone | Relative to random / prior-work baseline |
| "Looks weird but moving on" | Intuition mismatch → re-examine metric |

---

## 6. Communication style (JY preference)

- 짧고 직설. 한 문장으로 될 걸 세 문장 X.
- 바로 실행. "할 수 있습니다" 대신 코드.
- 불필요한 요약 X (JY는 diff 읽음).
- 이모지 / 시간 예측 X.
- "ㄱ" = 진행. "ㄱㄱㄱ" = 빠르게.

---

## 7. Debate framework (for major decisions)

```
Round 1: 3 personas (Method Skeptic / Data-Centric / Strategic) — 2-3 sentences each
Round 2: cross-rebuttal with logic + evidence
Round 3: revised stances + consensus
Final review (Claude main):
  - my call (no both-sides-ism)
  - agreed
  - unresolved
  - 3 recommended actions
  - blind spots
```

---

## 8. New project bootstrap

```bash
scripts/init_project.sh NN_new_project
```

Generates:
```
projects/NN_xxx/
├── README.md
├── rules/{target_metrics.md, constraints.md, gate_c_stage.txt}
├── state/{leaderboard.jsonl, paper_tried.jsonl, hypothesis_registry.jsonl,
│         gate_log.jsonl, engine_state.json, insights.md}
├── experiments/
├── negative_results/
├── reproducibility_manifests/
└── research_log/{sessions/, references/, claude_evaluation/}
```

Pick a Direction ID prefix (≤ 5 uppercase letters, e.g. `EMA`, `SPC`).

---

## 9. Kill switch

```bash
touch scripts/kill_switch
```

State Machine HALTs on next pre-transition check. Resume: `rm scripts/kill_switch` then `jy bootstrap`.

---

## 10. Hard nos (Claude must not)

1. Score its own directions (`hindsight_score` is JY-only).
2. Hide its own weaknesses via self-censoring.
3. Delete rejected directions (blind-spot data).
4. Change data split / metric / seed count mid-iteration (Gate A violation).
5. Re-try a `(method_id, config_fingerprint)` pair already in `paper_tried.jsonl`.
6. Proceed past suspected data leakage — flag and HALT.
7. Run destructive / shared-state actions (push, rm, force) without JY's explicit consent for that specific action.
8. Propose a method without prior-research search (when triggered).
9. Set `method_globally_blocked: true` — that is human-only.
10. Edit any `*.jsonl` in place. Only append, only `supersedes`.

---

## 11. v1 → v2 migration status (2026-05-16)

Adopted (this commit):
- ARCHITECTURE.md
- engine/schemas/*.json (6 schemas)
- .global/protocol/{state_machine, repair_categories, gate_c_revised}.md
- METHODOLOGY v2 (this file)

Pending code implementation (Tier 1 → 3, see `ARCHITECTURE.md` §11):
- engine/core/{state_machine, task_dag, append_only_logger, compute_budget, reproducibility_manifest, leakage_auditor}.py
- engine/gates/{a, b, c}.py
- engine/agents/*.md (LLM worker prompts)
- engine/cli/jy.py

Until the code lands, projects can opt into v2 schemas voluntarily — new `paper_tried.jsonl` rows include `config_fingerprint`, new experiments register hypotheses, etc. v1 rows remain valid.
