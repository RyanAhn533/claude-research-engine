# JY Research Engine — Architecture v2.0

> Protocol-enforced, append-only research execution engine.
> LLM is a worker, not an orchestrator. Protocol lives in code, not in prompts.

**Status**: v2.0 design adopted 2026-05-16. Current code = v1 (5-phase loop, README §2). Implementation roadmap: §11.

---

## 0. Design Invariants

1. LLM is a worker, not an orchestrator.
2. Protocol is enforced by code and the file system — never delegated to the LLM.
3. All records are append-only. Corrections are new rows, not edits.
4. Non-reproducible runs are not experiments.
5. The human enters via **explicit named gates**, not via deletion.
6. **All rejects are advisory. Kills happen only on protocol violation.**

---

## 1. 4-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│ Layer 1: Control Layer (Deterministic)          │
│  • State Machine (phase-level FSM)              │
│  • Task DAG (intra-phase orchestration)         │
│  • Quality Gates A / B / C / D                  │
│  • Protocol Validator                           │
│  • Compute Budget Tracker         [NEW]         │
│  • Reproducibility Manifest       [NEW]         │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│ Layer 2: LLM Reasoning Layer (Non-deterministic)│
│  Agents (6 simultaneous personas):              │
│  • Method Planner (5 strategies)                │
│  • Failure Analyzer                             │
│  • Repair Planner (3 categories)  [REVISED]     │
│  • Novelty Critic (advisory)      [REVISED]     │
│  • Reviewer Simulator             [NEW]         │
│  • Insight Summarizer                           │
│  Claude Toolbox (operational surface) [§13]:    │
│  • Subagent (context isolation, parallel)       │
│  • Plan mode (read-only pre-analysis)           │
│  • Adaptive thinking (low/medium/high/max)      │
│  • Prompt caching (stable context)              │
│  • Skills / Hooks (codified workflows)          │
│  • Memory bridge (~/.claude ↔ engine)           │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│ Layer 3: Execution Layer                        │
│  • Experiment Runner                            │
│  • Unit / Integration Tester                    │
│  • Leakage Auditor                [NEW]         │
│  • Statistical Evaluator          [REVISED]     │
│  • Artifact Generator                           │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│ Layer 4: Memory Layer (Append-only)             │
│  • leaderboard.jsonl                            │
│  • paper_tried.jsonl (+ config_fingerprint)     │
│  • hypothesis_registry.jsonl      [NEW]         │
│  • negative_result_archive/       [NEW]         │
│  • insights.md / insights_global.md             │
│  • reproducibility_manifests/     [NEW]         │
└─────────────────────────────────────────────────┘
```

**Hierarchy rule**: State Machine is top. State Machine selects the phase → that phase runs its Task DAG. DAGs do not cross phase boundaries.

---

## 2. State Machine ⊃ Task DAG

### 2.1 Phase-level FSM

```
[BOOTSTRAP]
  → [METHOD_SEARCH]
  → [HYPOTHESIS_REGISTRATION]        [NEW]
  → [HUMAN_APPROVAL]
  → [IMPLEMENTATION]
  → [EVALUATION]
  → [SELF_ATTACK]                    [NEW — adversarial Claude vs Claude]
  → [LOGGING]
  → [INSIGHT_UPDATE] (every 5 iter)
  → [METHOD_SEARCH] (loop)

Failure transitions:
  [IMPLEMENTATION] → [REPAIR] → [IMPLEMENTATION]   (budget-capped)
  [REPAIR_EXHAUSTED] → [POSTMORTEM] → [METHOD_SEARCH]
  [PROTOCOL_VIOLATION] → [HALT]                    (human only)
  [BUDGET_EXHAUSTED] → [HALT]
```

### 2.2 Task DAG (intra-phase, IMPLEMENTATION example)

```
T_impl_1  load_design_md
T_impl_2  generate_reproducibility_manifest   [NEW]
T_impl_3  write_run_py
T_impl_4  import_check
T_impl_5  dry_run
T_impl_6  unit_test
T_impl_7  leakage_audit                       [NEW]
T_impl_8  small_batch_smoke_test
T_impl_9  gate_B_check
            pass → return IMPL_COMPLETE
            fail → repair_router (3 categories)
```

DAG cannot transition phases. Phase transition is the State Machine's authority alone.

---

## 3. Quality Gates — Phase Mapping

| Gate | Fires at | Responsibility | On fail |
|------|----------|----------------|---------|
| **A** Protocol | end of METHOD_SEARCH, end of EVALUATION | metric / split / seed / baseline comparability | `protocol_violation` → HALT |
| **B** Implementation | end of IMPLEMENTATION | code runs, leakage audit, output well-formed | Repair Router (3 categories) |
| **C** Evaluation | end of EVALUATION | statistical rigor, effect size, CI | `neutral` or `failed` |
| **D** Research Value | before LOGGING | novelty, ablation depth, reviewer defense | **advisory only — no reject power** |

### 3.1 Gate C (revised — statistical rigor)

```yaml
gate_c:
  prototype_stage:
    seed_min: 3
    required_metrics: [mean, std]
    decision: directional_only         # no statistical claim
  paper_ready_stage:
    seed_min: 7
    required_metrics: [mean, std, bootstrap_95_CI, cohen_d]
    statistical_tests:
      - paired_comparison_with_baseline
      - wilcoxon_signed_rank
    equivalence_test:
      framework: TOST
      delta_threshold: pre_registered  # from design.md
    decision_rule: |
      improvement_candidate IFF:
        bootstrap_CI_lower > baseline_mean
        AND cohen_d > 0.5
        AND wilcoxon_p < 0.05
```

Day-to-day iteration runs `prototype`. Paper-submission window escalates to `paper_ready`.

### 3.2 Gate D (advisory only)

Gate D output is one of:
- `strong_accept` — Reviewer Simulator also passes
- `accept_with_concerns` — auto-attaches concern list
- `flag_for_human_review` — human decides
- `reject` — **removed in v2**. Gate D cannot block. Only protocol violations block.

---

## 4. `paper_tried` Redesign — config_fingerprint

**v1 problem**: "AU 8-node graph" failed once → permanently blocked. False negatives accumulate.

**v2 schema**:
```json
{
  "method_id": "au_8node_graph",
  "config_fingerprint": "sha256(lr=1e-4,bs=32,init=xavier,...)",
  "result": "failed",
  "failure_mode": "gradient_explosion_at_epoch_3",
  "failure_category": "dynamics",
  "retryable_with_changes": ["lr", "init", "warmup"],
  "blocked_configs": ["lr=1e-4,bs=32,init=xavier"],
  "method_globally_blocked": false
}
```

**Dedup rules**:
- exact `(method_id, config_fingerprint)` match → reject
- same `method_id`, different config → Method Planner must justify "why expect different outcome"
- `method_globally_blocked: true` is **human-only**

---

## 5. Repair Planner — 3 Categories

```yaml
repair_categories:
  syntactic:
    examples: [ImportError, ShapeMismatch, TypeError, IndexError]
    retry_budget: 3
    strategy: code_diff_patch

  semantic:
    examples: [wrong_loss_fn, incorrect_metric_aggregation, off_by_one]
    retry_budget: 2
    strategy: design_review_then_patch
    requires: re_read_design_md

  dynamics:
    examples: [gradient_explosion, NaN_loss, non_convergence, mode_collapse]
    retry_budget: 1                    # rerunning the same bug is waste
    strategy: hyperparameter_or_architectural_change
    requires: failure_analyzer_invocation
    escalate_to_method_search_if_failed: true
```

**Core insight**: `dynamics` failures escalate to METHOD_SEARCH after 1 attempt. Rerunning the same training-dynamics bug N times burns compute without information.

---

## 6. New Components

### 6.1 Hypothesis Registry (pre-registration)

Each experiment's `design.md` must include:

```yaml
hypothesis:
  primary: "AU 8-node graph lowers within_class_distance vs 7-node"
  null: "no difference"
  success_criterion:
    metric: within_class_distance
    direction: lower_is_better
    delta_threshold: 0.5             # equivalence margin for TOST
  failure_implication: "node count is not the critical factor"
falsifiability_check: PASS           # validated automatically
```

**Purpose**: prevents post-hoc narrative shopping (changing hypothesis after seeing results).

### 6.2 Reviewer Simulator

Input: `design.md`. Output (ICLR/NeurIPS reviewer persona LLM):

```yaml
expected_critiques:
  - severity: high
    point: "Missing baseline [X et al. 2024]"
    suggested_action: "add baseline or justify omission"
estimated_review_score: 5.5 / 10
weak_points: [...]
strong_points: [...]
```

Gate D reads it; **does not block**. Output goes to humans.

### 6.3 Compute Budget Tracker

```json
{
  "session_id": "2026-05-16-001",
  "budget": {
    "gpu_hours_total": 100, "gpu_hours_used": 47.3,
    "cost_usd_total": 500, "cost_usd_used": 234.50,
    "wall_clock_hours_total": 168, "wall_clock_hours_used": 89.2
  },
  "alerts": { "warn_at_80_percent": true, "halt_at_95_percent": true },
  "per_experiment_cap": { "gpu_hours": 8, "halt_if_exceeded": true }
}
```

State Machine checks budget before every phase transition.

### 6.4 Reproducibility Manifest (one per experiment)

```yaml
exp_id: exp_039
created_at: 2026-05-16T14:23:01Z
git_sha: a3f8c92...
data_hash: sha256:b91a...
env_lock: env.lock                     # poetry.lock or uv.lock
random_state:
  python: 42
  numpy: 42
  torch: 42
  cuda_deterministic: true
hardware:
  gpu: A6000 48GB
  driver: 535.104
  cuda: 12.1
container_digest: sha256:c8d2...
```

Quarterly: sample one manifest, rebuild env, verify results.

### 6.5 Leakage Auditor

Runs in Gate B:

```python
checks = [
    "test_set_not_in_train_indices",
    "preprocessing_stats_fit_on_train_only",
    "no_future_information_in_features",
    "cv_folds_respect_grouping_variable",
    "augmentation_only_on_train_split",
    "label_not_derivable_from_other_features",
]
```

Any fail → `protocol_violation` → HALT.

### 6.6 Negative Result Archive

`paper_tried.jsonl` is flat. Failure analysis needs richer data:

```
negative_results/
└── exp_037_failed/
    ├── loss_curve.json
    ├── grad_norm_trace.json
    ├── attention_maps/
    ├── final_predictions_sample.json
    ├── postmortem.md
    └── manifest.yaml
```

Failure Analyzer reads from here, not `paper_tried.jsonl`.

---

## 7. Phase Flow (v2)

```
Phase 0: BOOTSTRAP
  - load state
  - check budget remaining
  - validate protocol files exist

Phase 1: METHOD_SEARCH
  strategies (≥1 of each per 4 iterations):
    a. Recent paper survey (WebSearch)
    b. Adjacent-domain transplant
    c. Failure-case analysis (reads negative_result_archive)
    d. Top-k hybrid
    e. First-principles redesign       [NEW]
  - Novelty Critic: advisory tagging only
  - dedup vs paper_tried by config_fingerprint
  - output: 3 candidates

Phase 1.5: HYPOTHESIS_REGISTRATION     [NEW]
  - each candidate writes hypothesis section
  - falsifiability check (automated)
  - pre-register success/failure criteria

Phase 2: HUMAN_APPROVAL
  - JY reviews 3 candidates + hypotheses
  - selects 1 (or rejects all → back to Phase 1)

Phase 3: IMPLEMENTATION
  - generate reproducibility_manifest first
  - write run.py
  - leakage audit
  - Gate B
  - repair router (3 categories) on fail

Phase 4: EVALUATION
  - N seeds (prototype: 3, paper: 7+)
  - bootstrap CI + Cohen's d + Wilcoxon
  - pre-registered criterion (TOST)
  - Gate C

Phase 4.5: SELF_ATTACK                 [NEW]
  - 3 adversarial personas, parallel subagents:
      • Method Skeptic   — attack design choices
      • Reviewer Simulator — ICLR/NeurIPS critique
      • Novelty Critic   — challenge novelty claim
  - all attacks written to gate_log.jsonl with gate="D", is_advisory=true
  - concerns auto-attached to leaderboard row
  - phase cannot block — purely surfaces blind spots before LOGGING
  - thinking budget: max  (this is where Claude depth matters most)

Phase 5: LOGGING
  - append leaderboard.jsonl
  - append paper_tried.jsonl (with config_fingerprint)
  - update hypothesis_registry with outcome
  - if failed: archive to negative_results/
  - Reviewer Simulator → Gate D (advisory)

Phase 6: INSIGHT_UPDATE (every 5 iter)
  - read last 5 iterations
  - extract works/doesn't_work patterns
  - update insights.md
  - bias next Phase 1 search
```

---

## 8. Repository Layout (v2 target)

```
claude-research-engine/
├── ARCHITECTURE.md              this file
├── METHODOLOGY.md               v2 operating manual
├── README.md
├── CLAUDE.md
├── HANDOFF.md
│
├── engine/                      [NEW — code layer]
│   ├── core/
│   │   ├── state_machine.py
│   │   ├── task_dag.py
│   │   ├── append_only_logger.py
│   │   ├── compute_budget.py
│   │   ├── reproducibility_manifest.py
│   │   └── leakage_auditor.py
│   ├── gates/
│   │   ├── gate_a_protocol.py
│   │   ├── gate_b_implementation.py
│   │   ├── gate_c_evaluation.py
│   │   └── gate_d_research_value.py
│   ├── agents/                  Layer 2 LLM workers (prompts)
│   │   ├── method_planner.md
│   │   ├── failure_analyzer.md
│   │   ├── repair_planner.md
│   │   ├── novelty_critic.md
│   │   ├── reviewer_simulator.md
│   │   └── insight_summarizer.md
│   ├── schemas/                 JSON Schema (frozen first)
│   │   ├── state.schema.json
│   │   ├── experiment.schema.json
│   │   ├── hypothesis.schema.json
│   │   ├── reproducibility_manifest.schema.json
│   │   ├── gate_result.schema.json
│   │   └── paper_tried_entry.schema.json
│   └── cli/
│       └── jy.py                jy bootstrap / plan / approve / ...
│
├── .global/
│   ├── protocol/
│   │   ├── engine_loop.md       v1 (kept for trace)
│   │   ├── state_machine.md     v2 phase FSM spec
│   │   ├── repair_categories.md v2 repair spec
│   │   ├── gate_c_revised.md    v2 statistical spec
│   │   ├── claude_leverage_2026.md
│   │   ├── gpu_sharing.md
│   │   └── report_format.md
│   └── templates/
│
├── methodology/                 append-only, cross-project
│   ├── directions.jsonl
│   ├── logic_chains.jsonl
│   └── insights_global.md
│
└── projects/
    └── 02_emotion_agent/
        ├── state/
        │   ├── leaderboard.jsonl
        │   ├── paper_tried.jsonl
        │   ├── hypothesis_registry.jsonl   [NEW]
        │   └── insights.md
        ├── experiments/
        ├── negative_results/               [NEW]
        ├── reproducibility_manifests/      [NEW]
        ├── research_log/
        └── ...
```

---

## 9. CLI (v2 target)

```bash
# Bootstrap / status
jy bootstrap --project 02_emotion_agent
jy status    --project 02_emotion_agent     # phase, budget, current best
jy validate  --project 02_emotion_agent     # protocol pre-check

# One iteration, phase by phase
jy plan      --project 02_emotion_agent
jy approve   --project 02_emotion_agent --exp exp_039
jy implement --project 02_emotion_agent --exp exp_039
jy evaluate  --project 02_emotion_agent --exp exp_039
jy log       --project 02_emotion_agent --exp exp_039

# Auto (halts at human gates)
jy run-next  --project 02_emotion_agent
jy run-until --project 02_emotion_agent --stop-at human_approval

# Periodic
jy insight   --project 02_emotion_agent             # every 5 iter
jy reproduce --project 02_emotion_agent --exp exp_033   # quarterly audit

# Budget / violations
jy budget     --project 02_emotion_agent
jy violations --project 02_emotion_agent
```

---

## 10. Identity (final)

**EN**: A protocol-enforced, append-only research execution engine that constrains LLM-based hypothesis generation within deterministic experiment control, statistical rigor, and human-validated quality gates.

**KO**: 결정적 실험 제어, 통계적 엄밀성, 사람 검증 게이트 안에 LLM 가설 생성을 가둔, 프로토콜 강제·append-only 연구 실행 엔진.

**Differentiator (one sentence)**: While other autonomous research systems focus on making the LLM smarter, JY Research Engine focuses on making research integrity unbreakable even when the LLM is dumb.

---

## 11. Implementation Priorities

### Tier 1 — engine spine (without these, no closed loop, no reproducibility)
```
engine/core/state_machine.py
engine/core/task_dag.py
engine/core/append_only_logger.py
engine/core/reproducibility_manifest.py
engine/core/compute_budget.py
engine/core/leakage_auditor.py
engine/gates/{gate_a, gate_b, gate_c}.py
engine/schemas/*.json
memory: leaderboard.jsonl, paper_tried.jsonl (with config_fingerprint), hypothesis_registry.jsonl
```

### Tier 2 — closed-loop completion
```
engine/agents/{method_planner, failure_analyzer, repair_planner, insight_summarizer}.md
engine/core/retry_repair_router.py
engine/core/protocol_validator.py
engine/core/postmortem_generator.py
memory: negative_result_archive/
```

### Tier 3 — differentiation / IP layer
```
engine/agents/{novelty_critic, reviewer_simulator}.md
engine/analysis/{hindsight_scoring, dpo_pair_extractor, insights_global_summarizer}.py
engine/gates/gate_d_research_value.py
```

**Hard prerequisite for Tier 1**: freeze all `engine/schemas/*.json` first. Schema migration cost > implementation cost.

---

## 12. Two-week bootstrap (target)

| Day | Deliverable |
|-----|-------------|
| 1–2 | All `engine/schemas/*.json` frozen (state, task, gate, experiment, hypothesis, manifest, paper_tried) |
| 3–5 | `state_machine.py` + `append_only_logger.py` + phase transition smoke test |
| 6–8 | `reproducibility_manifest.py` + `compute_budget.py` + `leakage_auditor.py` |
| 9–11 | Gate A/B/C implemented; dummy experiment runs end-to-end |
| 12–14 | Method Planner agent + Hypothesis Registry + one real experiment full iteration |

**Acceptance**: on day 14, a single `jy run-next` walks method_search → human_approval → implementation → evaluation → self_attack → logging, leaves append-only artifacts, and the manifest reproduces.

---

## 13. Claude Toolbox Layer

> Layer 2 lists *which agents* exist. This section lists *how Claude is invoked* — the operational surface of Claude itself.
> Spec detail: `.global/protocol/claude_toolbox.md`. Heritage: `.global/protocol/claude_leverage_2026.md`.

### 13.1 Operational tools (six surfaces)

| Tool | What it does | When to use |
|------|--------------|-------------|
| **Subagent** | Spawns isolated Claude with own context window; returns summary only | Heavy reads (paper survey, leaderboard stats), parallel attacks, side-tasks that would pollute main context |
| **Plan mode** | Read-only Claude — no file mutation possible | Phase 1 method search; any large code change preview; satisfies "JY approval before destructive" by construction |
| **Adaptive thinking** | Effort budget {low / medium / high / max} → Claude self-titrates reasoning depth | Per-phase, per-agent rule (see matrix below) |
| **Prompt caching** | Stable context (CLAUDE.md, METHODOLOGY.md, ARCHITECTURE.md, leaderboard tail) cached server-side | Every long-running phase. Caches ~10% the cost of re-tokenizing |
| **Skills + Hooks** | Codified workflows (`/distill`, `/debate`) and lifecycle scripts (pre-commit validate, post-experiment log) | Repetitive workflows; deterministic guard-rails |
| **Memory bridge** | `~/.claude/projects/.../memory/` ↔ engine state | Cross-session facts (user profile, feedback rules); never stores live state |

### 13.2 Agent × Tool matrix

| Agent | Subagent? | Plan mode? | Thinking | Cached context | Skill / Hook |
|-------|-----------|-----------|----------|----------------|--------------|
| **Method Planner** | ✅ for paper survey (Strategy a) | ✅ recommended | **high** | CLAUDE.md + insights_global.md | — |
| **Failure Analyzer** | optional | — | high | last experiment artifacts | — |
| **Repair Planner — syntactic** | ❌ | — | low | run.py + traceback | — |
| **Repair Planner — semantic** | ❌ | — | medium | run.py + design.md | — |
| **Repair Planner — dynamics** | ✅ (long failure analysis) | — | high | run.py + loss curve | — |
| **Novelty Critic** | ✅ (parallel with Reviewer) | — | high | design.md + recent literature | — |
| **Reviewer Simulator** | ✅ (always — adversarial isolation) | — | **max** | design.md + paper claims | — |
| **Insight Summarizer** | ✅ (large window: 5 iters of leaderboard) | — | high | last 5 iter rows | — |
| **3-persona debate** | ✅ × 3 parallel | — | **max** | CLAUDE.md only (force first-principles) | `/debate` skill |
| **Direction logging** | ❌ | — | low | — | append-only hook |

### 13.3 Phase × Toolbox routing

| Phase | Primary tools | Why |
|-------|---------------|-----|
| `BOOTSTRAP` | hook (pre-flight checks) | Deterministic, no Claude needed |
| `METHOD_SEARCH` | Plan mode + Subagent × 5 strategies | Strategies a/b/c/d/e run as parallel subagents → main gets summary |
| `HYPOTHESIS_REGISTRATION` | Main + low thinking | Mechanical fill of pre-registered fields |
| `HUMAN_APPROVAL` | — | Human only. Claude waits |
| `IMPLEMENTATION` | Main + medium/high thinking, **no plan mode** | Mutation needed. Plan mode used only for the diff preview, not the patch |
| `EVALUATION` | Hook (run script) → Main reads numbers | No reasoning during the run |
| `SELF_ATTACK` | Subagent × 3 (Skeptic / Reviewer / Novelty) + **max thinking** | Where Claude attacks Claude. Parallel adversaries. Highest depth |
| `LOGGING` | Hook (validate schema, append) + Main writes Direction row | Deterministic file writes; one prose row from Main |
| `INSIGHT_UPDATE` | Subagent (reads 5-iter leaderboard) + high thinking | Heavy read → summary returned |

### 13.4 Hard rules

1. **Mutation phases (IMPLEMENTATION, LOGGING) cannot use Plan mode** — Plan mode is read-only by design.
2. **`SELF_ATTACK` always runs subagents** — adversarial personas must not share context with the design they attack.
3. **Adaptive thinking budget is per-call, not per-session** — `low` for a syntactic patch even if the iteration is on `max` overall.
4. **Subagent results are summaries, not blobs** — if a subagent returns >2K tokens, it has failed its contract.
5. **Direction IDs are appended via hook, not by Claude in prose** — prompt-level rule was unreliable; v2 has a post-response hook that scans for `EMA-D###` claims and appends to `directions.jsonl`.
6. **Memory bridge is read-mostly** — engine never writes user memory; user memory never writes engine state. They cross-reference via slugs.

### 13.5 What this layer is *not*

- Not a wrapper around Anthropic SDK — Claude Code is the runtime.
- Not a router that decides "which model" — model selection is left to Claude Code's defaults.
- Not a place to put method-level prompts — agent prompts live in `engine/agents/*.md`.

This layer specifies **how to invoke Claude**, not **what to ask Claude**.

---

## 14. Why this is the actual differentiator

Other autonomous-research systems use Claude as a single role: "generate the next idea." This engine uses Claude as:

- **6 concurrent personas** (Method Planner, Failure Analyzer, Repair Planner, Novelty Critic, Reviewer Simulator, Insight Summarizer)
- **3 adversaries against itself** in `SELF_ATTACK` (Method Skeptic, Reviewer Simulator, Novelty Critic — parallel subagents)
- **A data source** — every Direction and Logic Chain Claude emits is logged to `directions.jsonl` / `logic_chains.jsonl`, hindsight-scored, and extractable as DPO pairs
- **A bounded worker** — code-level invariants (append-only, config_fingerprint dedup, pre-registered hypotheses) prevent Claude's typical failure modes (narrative shopping, false termination, self-overwrite)

The leverage isn't "smarter Claude." The leverage is **simultaneously running Claude in many roles, with Claude attacking Claude, while every output becomes training data — and none of it can corrupt the research record.**
