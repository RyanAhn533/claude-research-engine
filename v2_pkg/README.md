# claude-research-engine

> Protocol-enforced, append-only research execution engine.
> LLM is a worker, not an orchestrator. Protocol lives in **code**, not prompts.

```
status            : v2.0 schema-frozen
operating model   : human-in-the-loop, plan-mode-aware, hook-enforced
active projects   : 2 (01_au_regionformer_q2, 02_emotion_agent)
hardware          : RTX A6000 48GB × 1
visibility        : private
```

---

## tl;dr — what changed in v2

v1 was a 5-phase loop documented in prose. v2 is the **same loop with deterministic enforcement**: schemas, hash-chained append-only logs, machine-readable permission policy, pre-registered hypotheses, paired-delta statistics.

The headline claim:

> While other autonomous-research systems focus on making the LLM smarter, this engine focuses on making **research integrity unbreakable even when the LLM is dumb**.

If a v1 invariant could be violated by a bad LLM turn, v2 makes that violation *physically impossible* via code + JSON Schema + hooks.

---

## 0. The five failure modes this prevents

| Failure mode | v1 defense | v2 defense |
|---|---|---|
| Context drift across sessions | HANDOFF.md (prose) | `jy status` reads actual jsonl chain; HANDOFF is advisory |
| Unfalsifiable progress | "fixed protocol" rule | `falsifiability_check` field, schema-enforced before HUMAN_APPROVAL |
| Re-trying failed methods | `paper_tried.jsonl` by method name | `(method_id, config_fingerprint)` pair dedup |
| Narrative drift | implicit in prose | `claim_registry.jsonl` + `experiment.linked_claims` (paper_ready: required) |
| Wasted reasoning | RLHF logs | + DPO-extractable Direction/Logic-Chain pairs across phases |

---

## 1. Architecture (v2)

Four layers. Top dictates, bottom serves.

```
┌─────────────────────────────────────────────────┐
│ 1. CONTROL LAYER (deterministic)                │
│  State Machine · Task DAG · Quality Gates A/B/C │
│  Permission Policy · Compute Budget Tracker     │
│  Reproducibility Manifest · Leakage Auditor     │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│ 2. LLM REASONING (non-deterministic)            │
│  6 agents: Method Planner / Failure Analyzer /  │
│   Repair Planner / Novelty Critic /             │
│   Reviewer Simulator / Insight Summarizer       │
│  Claude toolbox: subagent / plan mode /         │
│   adaptive thinking / prompt cache / hooks      │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│ 3. EXECUTION                                    │
│  Experiment Runner · Tester · Stat Evaluator    │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│ 4. MEMORY (append-only)                         │
│  leaderboard · paper_tried · hypothesis_registry│
│  claim_registry · negative_results              │
│  reproducibility_manifests · insights           │
└─────────────────────────────────────────────────┘
```

### Phase flow (v2)

```
BOOTSTRAP
  → STATUS_SYNC
  → METHOD_SEARCH                    [plan mode + 5 subagent strategies, high]
  → HYPOTHESIS_REGISTRATION          [falsifiability_check]
  → HUMAN_APPROVAL                   [JY gate]
  → IMPLEMENTATION                   [Gate B, repair router on fail]
  → EVALUATION                       [Gate C, prototype/paper_ready]
  → CLAIM_LINKING                    [link to paper claim_id]
  → CONDITIONAL_SELF_ATTACK          [3 adversaries × max thinking, IF triggered]
  → LOGGING                          [hook-driven appends, Gate D advisory]
  → PAPER_QUEUE_UPDATE
  → INSIGHT_UPDATE (every 5 iter)
  → METHOD_SEARCH (loop)
```

`STATUS_SYNC`, `CLAIM_LINKING`, `CONDITIONAL_SELF_ATTACK`, `PAPER_QUEUE_UPDATE` are new in v2 (see [CHANGELOG_v1_to_v2.md](docs/CHANGELOG_v1_to_v2.md)).

---

## 2. Quick start

### 2.1 First-time install

```bash
git clone git@github.com:RyanAhn533/claude-research-engine.git
cd claude-research-engine
bash install_v2.sh        # backs up v1 files, lays down v2 layer
```

The install script:
1. Backs up `CLAUDE.md`, `METHODOLOGY.md`, `README.md` to `.v1_backup/`
2. Lays down `engine/`, `.claude/`, new `README.md`, new `CLAUDE.md`, `CHANGELOG_v1_to_v2.md`
3. Makes hooks executable
4. Runs `python engine/tests/validate_schemas.py` — 24 checks pass means schemas are frozen-ready

### 2.2 Bootstrap a project (or migrate an existing one)

```bash
python -m engine.cli.jy bootstrap --project 02_emotion_agent --prefix EMA --gpu-hours 200
```

This creates (or amends without overwriting):
```
projects/02_emotion_agent/
├── state/
│   ├── engine_state.jsonl           [seeded with one BOOTSTRAP row]
│   └── permission_policy.jsonl      [seeded with 22 default policies]
├── experiments/
├── negative_results/
├── reproducibility_manifests/
└── rules/direction_prefix.txt       [contains: EMA]
```

### 2.3 Start a Claude Code session

```bash
cd ~/claude-research-engine
claude
```

Inside the session:

```
/status 02_emotion_agent
/validate 02_emotion_agent
/iter 02_emotion_agent 040       # METHOD_SEARCH → HUMAN_APPROVAL
# JY reviews 3 candidates, picks one or rejects all
# (engine implements, evaluates, etc.)
/attack 02_emotion_agent 040     # CONDITIONAL_SELF_ATTACK
/log 02_emotion_agent 040        # LOGGING
```

---

## 3. The 8 frozen schemas

All in `engine/schemas/`. Draft-07 JSON Schema. Validated by `engine/tests/validate_schemas.py` (24 checks: 8 self + 8 positive + 8 negative).

| Schema | Append target | Critical invariant |
|---|---|---|
| `state` | `state/engine_state.jsonl` | one phase at a time; kill_switch_present halts |
| `task` | `state/task_log.jsonl`, `state/gate_log.jsonl` | DAG nodes don't cross phase boundaries |
| `experiment` | `state/leaderboard.jsonl` | paper_ready ⇒ n_seeds ≥ 7 + paired-delta CI + Cohen's d + Wilcoxon + linked_claims |
| `hypothesis` | `state/hypothesis_registry.jsonl` | falsifiability_check=PASS required before HUMAN_APPROVAL |
| `reproducibility_manifest` | `reproducibility_manifests/*.yaml` | missing manifest blocks LOGGING |
| `paper_tried_entry` | `state/paper_tried.jsonl` | dedup by (method_id, config_fingerprint); method_globally_blocked=true is human-only |
| `claim` | `state/claim_registry.jsonl` | required_for_submission=true ⇒ self_attack_required=true + venue |
| `permission` | `state/permission_policy.jsonl` | hard_block ⇒ override_allowed=false |

---

## 4. Active projects

| # | Project | Stage | Iter | Venue | Key finding |
|---|---|---|---|---|---|
| 01 | `01_au_regionformer_q2` | draft-ready, handed to 후배 | 15 | ESWA (IF 7.5) | Mouth > Eyes under Yonsei 298-person consensus |
| 02 | `02_emotion_agent` | Phase 6 active | 36+ | NeurIPS 2026 / TAFFC | **LoRA universal, ICL modality-gated** (3-tier hierarchy: prompt 29 → ICL 42 → LoRA 55) |

Latest: `projects/<NN>/state/leaderboard.jsonl | tail`. Snapshot: `projects/<NN>/SNAPSHOT.md`.

---

## 5. Hard invariants (`engine` refuses to proceed if any breaks)

1. LLM does not choose phase transitions. Code does.
2. All records are append-only. In-place edits are detected via size+hash and the engine halts.
3. Non-reproducible runs are not experiments. Missing `reproducibility_manifest` blocks LOGGING.
4. Humans enter via named gates. `HUMAN_APPROVAL` + permission policy together resolve auto/manual conflicts.
5. All rejects are advisory. Kills happen only on protocol violation.
6. Falsifiability is pre-registered. `falsifiability_check=FAIL` blocks the candidate.
7. JY consent required before destructive / shared-state actions. Hooks enforce.
8. Prior-research search before proposing a new method/metric (Strategy a of METHOD_SEARCH).
9. Claude never scores its own directions. `hindsight_score` is JY-only.
10. `method_globally_blocked: true` is human-only. The engine cannot set this.

`engine/core/append_only_logger.py` + `permission_policy.py` enforce #1, #2, #3, #5, #7, #10. Schemas enforce #6, #9.

---

## 6. RLHF data pipeline (carried over from v1)

```
Per-iteration                         Cross-project (append-only)
─────────────                         ────────────────────────────
projects/NN/research_log/             methodology/
  claude_evaluation/                    directions.jsonl       (21+ entries)
    directions.jsonl   ────────────►    logic_chains.jsonl     (7+ entries)
    logic_chains.jsonl ────────────►    insights_global.md
                                      
                                        +1 week later
                                        ────────────
                                        JY scores hindsight (1–10)
                                        → DPO pair extraction
```

v2 addition: Direction rows now include `phase` and `gate` context, so DPO pairs can condition on phase.

---

## 7. Repository layout

```
claude-research-engine/
├── README.md                          this file
├── CHANGELOG_v1_to_v2.md              v1 → v2 migration record
├── CLAUDE.md                          loaded by every Claude Code session
├── ARCHITECTURE.md                    layer + phase + agent spec
├── METHODOLOGY.md                     operating manual
├── HANDOFF.md                         human-readable session handoff
├── install_v2.sh                      one-shot installer
│
├── .claude/                           [NEW v2]
│   ├── settings.json                  hooks registered here
│   ├── commands/                      slash commands (/iter, /log, /attack, ...)
│   └── hooks/                         block_jsonl_in_place.sh, require_human_gate.sh
│
├── engine/                            [NEW v2 — code spine]
│   ├── core/
│   │   ├── append_only_logger.py
│   │   ├── permission_policy.py
│   │   └── hashing.py
│   ├── cli/
│   │   └── jy.py                      `python -m engine.cli.jy <cmd>`
│   ├── schemas/                       8 frozen JSON Schemas
│   ├── tests/
│   │   ├── validate_schemas.py
│   │   └── fixtures/                  16 fixtures (8 valid + 8 invalid)
│   └── agents/                        (Tier 2: 6 LLM agent prompts — TBD)
│
├── methodology/                       cross-project, append-only
│   ├── directions.jsonl
│   ├── logic_chains.jsonl
│   └── insights_global.md
│
├── projects/
│   ├── 01_au_regionformer_q2/...
│   └── 02_emotion_agent/...
│
└── scripts/
    ├── init_project.sh
    └── kill_switch_info.md
```

---

## 8. Implementation status

✅ **Tier 1 — schema + spine** (frozen)
- 8 schemas frozen + 24-check validator
- AppendOnlyLog with hash chain + tamper detection
- Permission policy (22 action classes seeded)
- `jy` CLI: bootstrap / status / validate / verify-chain / seed-policy
- Claude Code integration: 5 slash commands + 2 hooks + settings.json

✅ **Tier 2 — closed-loop runtime** (this commit)
- `state_machine.py` — FSM with 16 phases, kill-switch poll, budget pre-check, pre-conditions per phase
- `compute_budget.py` — GPU / cost / wall-clock accounting + per-experiment caps
- `reproducibility_manifest.py` — git_sha + git_dirty + data_hash + env_lock + random_state + hardware auto-collected
- `leakage_auditor.py` — 6 checks (3 automatic, 3 required declarations parsed from design.md)
- `engine/gates/gate_c.py` — paired-delta CI + Cohen's d + Wilcoxon + TOST (paper_ready) / mean±std (prototype)
- 6 agent prompts in `engine/agents/`: method_planner, failure_analyzer, repair_planner, novelty_critic, reviewer_simulator, method_skeptic, insight_summarizer
- `test_tier2.py` — 7-step integration smoke test (FSM + budget + manifest + audit + gate_c)

⬜ **Tier 3 — differentiation** (after first NeurIPS submission)
- DPO pair extractor
- hindsight scoring UI
- Gate D research-value advisory output
- Reviewer Simulator full implementation

---

## 9. License / IP

Private. Includes JY personal IP (POPR engine) and Yonsei lab assets (298-person consensus — external distribution prohibited). Model weights gitignored.

---

## 10. Citation (placeholder)

```bibtex
@misc{ahn2026claude_research_engine,
  author = {Ahn, JY},
  title  = {claude-research-engine: a protocol-enforced research execution engine
            producing Q1 papers and RLHF data simultaneously},
  year   = {2026},
  note   = {v2.0 — schema-frozen, hash-chained, hook-enforced. Private framework.}
}
```

---

## References (engine design grounding)

- arxiv 2601.03315 — *Why LLMs Aren't Scientists Yet* (3/4 fully-autonomous attempts fail at impl/eval; motivates human-in-the-loop)
- arxiv 2501.04227 — *Agent Laboratory* (we replace synthetic feedback with retrospective hindsight)
- Anthropic, *Building Effective Agents* (2025) — orchestrator/worker
- Anthropic, *2026 Agentic Coding Trends Report* — Plan mode, subagents, skills, hooks
- Claude API docs, *Adaptive thinking* — per-call effort budget
