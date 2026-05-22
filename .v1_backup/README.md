# claude-research-engine

A semi-autonomous research workspace where a human PI (JY) and Claude run a structured 5-phase iteration loop, producing (a) Q1-target papers and (b) retrospectively-scored RLHF data, simultaneously, from the same execution trace.

```
status            : active
operating model   : human-in-the-loop, plan-mode-aware
active projects   : 2 (au_regionformer_q2, emotion_agent)
exp count         : 51 across projects (15 + 36)
direction IDs     : 21 (methodology/directions.jsonl)
logic chains      : 7 (methodology/logic_chains.jsonl)
hardware          : RTX A6000 48GB × 1 (single-host, MPS-shareable)
visibility        : private
```

---

## 1. Problem this solves

Three failure modes of typical LLM-assisted research:

1. **Context drift** — chat sessions lose track of what's been tried, what failed, why. By iter 10 the assistant repeats methods from iter 3.
2. **Unfalsifiable progress** — without a fixed evaluation protocol, every "improvement" is suspect. Baseline drift, seed cherry-picking, metric swapping.
3. **Wasted reasoning** — the dialogue, debates, and rejected proposals (which contain the actual signal about *how* the LLM thinks under pressure) are thrown away after each session.

This engine addresses all three by enforcing:

- A **5-phase loop** with append-only state (`leaderboard.jsonl`, `paper_tried.jsonl`) — Phase 4 mechanically prevents method re-tries.
- A **fixed evaluation protocol** per project (`rules/target_metrics.md`, `rules/constraints.md`) — any deviation is logged as protocol violation.
- A **Direction ID + Logic Chain** record of every Claude proposal, scored retrospectively by JY 1 week later — turning the dialogue into DPO-extractable preference data anchored to a real outcome (paper acceptance).

The engine deliberately does **not** aim for full autonomy (cf. AutoGPT, Agent Laboratory). Empirically (arxiv 2601.03315, *"Why LLMs Aren't Scientists Yet"*), 3 of 4 fully-autonomous research attempts fail at implementation/evaluation. Keeping the human in the loop on protocol decisions is a deliberate design choice, not a limitation.

---

## 2. Architecture

### 2.1 Iteration loop (5 phases)

```
                       ┌─────────────────────────────────┐
                       │  rules/target_metrics.md        │  ← fixed at project start
                       │  rules/constraints.md           │
                       └────────────────┬────────────────┘
                                        │
  ┌─── state/leaderboard.jsonl ─────────┼────────── state/paper_tried.jsonl ───┐
  │  (append-only, per-iter row)        │           (method + status, dedupe) │
  └────────────────┬────────────────────┘─────────────────────────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Phase 1  Method search                                  │
   │   strategies a/b/c/d (forced diversity ≥ 1× per 4 iter): │
   │     a. Recent paper survey (WebSearch)                   │
   │     b. Adjacent-domain transplant                        │
   │     c. Failure-case analysis of current best             │
   │     d. Top 2-5 hybrid / ensemble                         │
   │   →  experiments/exp_NNN/design.md                       │
   ├──────────────────────────────────────────────────────────┤
   │  Phase 2  Implementation                                 │
   │   baseline → exp_NNN/run.py with design diff             │
   │   unit test → 3 retries max → postmortem.md if fail     │
   ├──────────────────────────────────────────────────────────┤
   │  Phase 3  Evaluation (protocol-fixed)                    │
   │   ≥ 3 seeds | mean ± std | bootstrap 1000 or std-based  │
   │   p < 0.05 vs current best → improvement candidate       │
   ├──────────────────────────────────────────────────────────┤
   │  Phase 4  Logging                                        │
   │   leaderboard.jsonl ← one-row append                     │
   │   paper_tried.jsonl ← method name + status               │
   │   methodology/directions.jsonl  ← global RLHF append     │
   │   methodology/logic_chains.jsonl← reasoning trace        │
   ├──────────────────────────────────────────────────────────┤
   │  Phase 5  Insight (every 5 iter)                         │
   │   state/insights.md  ← "works / doesn't work" patterns   │
   │   methodology/insights_global.md ← cross-project         │
   └──────────────────────────────────────────────────────────┘
```

Detailed protocol: `.global/protocol/engine_loop.md`. Report format: `.global/protocol/report_format.md` (8-section template).

### 2.2 RLHF data pipeline

The methodology layer is append-only across projects. Every new project inherits failure patterns from previous ones.

```
Per-iteration                         Cross-project (append-only)
─────────────                         ────────────────────────────
projects/NN/research_log/             methodology/
  claude_evaluation/                    directions.jsonl       (21 entries)
    directions.jsonl   ────────────►    logic_chains.jsonl     (7 entries)
    logic_chains.jsonl ────────────►    insights_global.md
                                      
                                        +1 week later
                                        ────────────
                                        JY scores hindsight (1-10)
                                        on each direction →
                                        DPO pair extraction
                                        (chosen vs rejected)
```

**Why retrospective scoring**: an immediate "good/bad" judgment biases toward agreeable proposals. A 1-week delay anchors the score to actual experimental outcome (did the suggested method actually help?), not to in-context plausibility. Direction `outcome` field is updated when the experiment lands.

### 2.3 Tool layer (2026 Claude features)

Operating rules for Claude itself (Plan mode, subagent delegation, adaptive thinking budget, prompt caching) are documented in `.global/protocol/claude_leverage_2026.md`. This sits one layer below `METHODOLOGY.md`: METHODOLOGY decides *what* to do per phase; `claude_leverage_2026.md` decides *which Claude tool* to use for it.

Mapping (excerpt):

| Phase | Tool | Effort budget |
|-------|------|---------------|
| Phase 1 (method search) | Plan mode + subagent for paper survey | high |
| Phase 2 (implementation) | Direct (no plan) | low → high if blocked |
| Phase 3 (evaluation) | Direct, tight loop | low |
| Phase 5 (insight) | Subagent for leaderboard stats | high |
| Debate framework (3 personas) | Adaptive thinking | max |

---

## 3. Data schemas

All state is JSONL. Each line is one independent record. Append-only.

### 3.1 `methodology/directions.jsonl`

One row per Claude proposal:

```json
{
  "id": "AUR-D001",
  "date": "2026-04-21",
  "session": "2026-04-21_gnn_roadmap_and_q1q2_strategy",
  "context": "ChatGPT가 제안한 FER×GNN 6단계 로드맵 중 2단계에서 8개 AU region을 노드로 하는 graph 사용 제안",
  "direction": "AU 8-node graph 폐기, 237K 이미지를 노드로 하는 sample-level kNN graph로 전환",
  "rationale": "8 node는 Set Transformer와 구분 안 됨. ME-GraphAU/ANFL 등 선행연구는 수십~수백 node.",
  "risk": "237K full-batch 불가 → GraphSAINT/SAGE 샘플링 필요. A6000 48GB 한계 주의.",
  "accepted": true,
  "hindsight_score": null,
  "outcome": null,
  "project": "01_au_regionformer_q1"
}
```

ID prefix scheme: `{PROJECT_PREFIX}-D{NNN}` where prefix is per-project (e.g. `AUR`, `EMA`). Prevents collision across projects sharing the global file.

`hindsight_score`: integer 1-10, set by JY ≥ 7 days after the direction. **Never set by Claude** (bias).
`outcome`: free text describing what actually happened when the direction was acted on.

### 3.2 `methodology/logic_chains.jsonl`

One row per multi-step reasoning sequence (typically one Claude response):

```json
{
  "chain_id": "AUR-LC001",
  "trigger_type": "proposal_review",
  "trigger": "ChatGPT의 FER×GNN 로드맵을 비판하고 전체 실험 로드맵 설계",
  "steps": [
    {"n": 1, "reasoning": "...", "action": "...", "outcome": "...", "sign": "+"},
    {"n": 2, "reasoning": "...", "action": "...", "outcome": "...", "sign": "+"},
    {"n": 5, "reasoning": "...", "action": "...", "outcome": "...", "sign": "-"}
  ],
  "final": "partial",
  "jy_feedback": "step 5 용어 오용은 후속 세션에서 교정",
  "good_pattern": "3인 페르소나 + 구체 수치 + 폐기/수정/수용 구분",
  "bad_pattern": "선행연구 용어의 세밀한 구분 생략",
  "dpo_extractable": true
}
```

`sign` per step: `+` (improved outcome) / `-` (wasted time / error) / `~` (neutral).

DPO pair extraction: when two chains for the same `trigger_type` differ in `final` outcome but share `good_pattern` / `bad_pattern` axes, they form a (chosen, rejected) pair.

### 3.3 `projects/NN/state/leaderboard.jsonl`

One row per evaluated experiment:

```json
{
  "exp_id": "exp_033_iemocap_hidden_cluster",
  "iteration": 38,
  "method": "Hidden-state cluster on IEMOCAP T1 vs T2 — cross-domain mechanism test",
  "paper_ref": "Paper B §5.2 mechanism cross-domain confirmation",
  "metric_name": "within_class_distance",
  "constraints_passed": true,
  "timestamp": "2026-04-27T18:30:00",
  "baseline_candidate": true,
  "notes": "...",
  "results": {"T1_zero_shot": {"within": 24.991}, "T2_icl_k4": {"within": 28.157}}
}
```

`baseline_candidate: true` flags an improvement awaiting JY approval before it replaces the current baseline (Phase 4 rule).

### 3.4 `projects/NN/state/paper_tried.jsonl`

One row per attempted method, regardless of success:

```json
{"method": "AU 8-node graph", "status": "rejected", "reason": "trivially equivalent to Set Transformer", "iter": 1}
```

Phase 1 strategy `a` (recent paper) and `c` (failure analysis) read this file to avoid retry. Phase 4 writes to it.

---

## 4. Repository layout

```
claude-research-engine/
├── README.md                          this file
├── CLAUDE.md                          global Claude config (auto-loaded each session)
├── METHODOLOGY.md                     10-section operating manual
├── HANDOFF.md                         current state for new sessions
│
├── .global/
│   ├── protocol/
│   │   ├── engine_loop.md             5-phase loop spec
│   │   ├── report_format.md           8-section experiment doc template
│   │   ├── gpu_sharing.md             A6000 48GB occupancy/sharing protocol
│   │   └── claude_leverage_2026.md    2026 Claude features → engine mapping
│   ├── rules_template/                target_metrics_template.md, constraints_template.md
│   └── templates/                     experiment.md, session.md, logic_chain.md, direction_eval.md
│
├── methodology/                       cross-project, append-only
│   ├── directions.jsonl               21 entries (all projects)
│   ├── logic_chains.jsonl             7 entries
│   └── insights_global.md             cross-project failure patterns
│
├── projects/
│   ├── 01_au_regionformer_q2/
│   │   ├── README.md, PROJECT_STATUS.md
│   │   ├── rules/                     target_metrics.md, constraints.md
│   │   ├── state/                     leaderboard.jsonl, paper_tried.jsonl, insights.md
│   │   ├── experiments/exp_NNN/       design.md, run.py, results, postmortem.md
│   │   ├── research_log/
│   │   │   ├── sessions/              dated session notes
│   │   │   └── claude_evaluation/     local directions/chains
│   │   ├── src/                       analysis code
│   │   └── results/                   summaries / figures (large artifacts external)
│   │
│   └── 02_emotion_agent/
│       ├── README.md, ROADMAP.md, Q1_WORKING.md, SNAPSHOT.md
│       ├── rules/, state/, experiments/, research_log/, src/, libs/
│       ├── baseline/                  reproducibility anchors
│       ├── references.bib, references/  literature
│       ├── figures/                   make_paper_figures.py + outputs
│       └── setup/                     env / data path verification
│
└── scripts/
    ├── init_project.sh                new-project scaffold
    └── kill_switch_info.md            emergency-stop semantics
```

---

## 5. Active projects

| # | Project | Stage | Iter count | Target venue | Notes |
|---|---------|-------|-----------|--------------|-------|
| 01 | au_regionformer_q2 | draft-ready, transferred to 후배 1저자 | 15 | ESWA (IF 7.5) | Counterintuitive Mouth > Eyes finding under Yonsei 298-person consensus |
| 02 | emotion_agent | active, Phase 6 (Stage 16 ablation + AffectNet finetune pending) | 36+ | IEEE TAFFC / ICMI / ACII | Mechanism finding: ICL benefit ↔ hidden-rep compression. Confirmed cross-domain (Korean FER + IEMOCAP) at exp_033. |

Project-level details: `projects/NN/README.md`. Current best per project: tail of `state/leaderboard.jsonl`. Per-project venue strategy: `projects/NN/rules/target_metrics.md`.

---

## 6. Operating invariants

These are hard rules. Violating them invalidates the iteration.

| Invariant | Rationale |
|-----------|-----------|
| Evaluation protocol (data split, seed, metric) is fixed at project start | Comparability across iterations. Drift = silent bias. |
| `p < 0.05` (or equivalent) required for "improvement" classification | Random fluctuation filtering. ≥ 3 seeds enforced. |
| No method retry — `paper_tried.jsonl` checked at Phase 1 | Prevents loop on already-failed approaches. |
| At most 1× same strategy (a/b/c/d) per 4 iter | Forced search diversity. |
| `scripts/kill_switch` file present → loop halts before next iter | Manual override. |
| Claude does **not** set `hindsight_score` on its own directions | Self-evaluation bias. JY-only field. |
| Failures (`paper_tried.jsonl: status=failed`) are written, not curated away | Negative results are signal. |
| Pre-experiment plan + JY approval for any destructive / shared-state action | Established convention. |
| Prior-research search before proposing new method/metric | Prevents reinventing or attacking obvious baselines. |
| Leaderboard rows immutable; corrections via new row + `notes` | Append-only audit trail. |

Full list: `METHODOLOGY.md §0, §10`.

---

## 7. New session bootstrap

For a Claude session entering this repo cold:

```
1. README.md                                       (this file)
2. CLAUDE.md                                       global config (priorities, style)
3. METHODOLOGY.md                                  engine ops
4. .global/protocol/claude_leverage_2026.md        tool-layer rules
5. projects/NN/README.md                           target project status
6. projects/NN/state/leaderboard.jsonl | tail      current best
7. projects/NN/research_log/sessions/ (latest)     last decisions
```

Same order applies after `/clear` or context compaction.

---

## 8. Hardware / runtime

- **GPU**: RTX A6000 48GB × 1 (Dell Precision 7960). Single-host. No multi-GPU.
- **Sharing**: BrandSpace FastAPI may occupy ~28-29 GB. Protocol for arbitration: `.global/protocol/gpu_sharing.md`.
- **Free VRAM thresholds**:
  - BrandSpace running: ≤ 7.7 GB → small inference / preprocessing only
  - BrandSpace stopped: ~48 GB → 7B LoRA fine-tune, full reproductions

Project 01 (sklearn / linear probe) is GPU-light. Project 02 (HF Trainer + bnb 4-bit + LoRA) needs the full 48 GB for fine-tunes.

---

## 9. License / IP

Private repository. Includes JY personal IP (POPR engine) and Yonsei lab assets (298-person consensus data — external distribution prohibited). Model weights gitignored. Source repo for AU-RegionFormer model code: `github.com/RyanAhn533/AU-RegionFormer`.

---

## 10. Citation (placeholder, conditional on paper acceptance)

```bibtex
@misc{ahn2026claude_research_engine,
  author = {Ahn, JY},
  title  = {claude-research-engine: a semi-autonomous research pipeline
            producing Q1 papers and RLHF data simultaneously},
  year   = {2026},
  note   = {Private framework. Companion to Ahn et al. (TAFFC/NHB submission).}
}
```

---

## References (engine design grounding)

- arxiv 2601.03315 — *Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts* (3/4 fail at impl/eval; motivates human-in-the-loop)
- arxiv 2501.04227 — *Agent Laboratory: Using LLM Agents as Research Assistants* (3-stage pipeline; we replace synthetic feedback with retrospective hindsight)
- Anthropic, *Building Effective Agents* (2025) — orchestrator/worker pattern, simplicity-first
- Anthropic, *2026 Agentic Coding Trends Report* — Plan mode, subagents, skills, hooks
- Claude API docs, *Adaptive thinking* — effort budget assignment per task class
