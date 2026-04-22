# claude-research-engine

> **Semi-autonomous research engine.** Human directs (JY as PI), Claude executes (as a PhD student), and the research process itself becomes RLHF training data.

[![Status](https://img.shields.io/badge/status-active-brightgreen)]() [![Project](https://img.shields.io/badge/active%20projects-1-blue)]() [![Best%20metric](https://img.shields.io/badge/current%20best-87.56%25-blueviolet)]() [![License](https://img.shields.io/badge/visibility-private-red)]()

---

## 🎯 What this is

A structured workspace where a real Q1 paper gets written by a **human PI + Claude partnership**, while every step — strategies, iterations, logic chains, failure modes — is recorded as data that can later train better LLMs.

Unlike AutoGPT (full autonomy) or a chat interface (no loop), this sits in between:
- **Human directs** the thesis, venue strategy, and blind-spot pivots
- **Claude runs** the 5-phase iteration loop (design → implement → evaluate → record → reflect)
- **Both produce** a leaderboard, a paper draft, and a cumulative RLHF log — in parallel

The real-world ground truth is **paper acceptance at a Q1 venue**, not a synthetic benchmark.

---

## 🌟 Novelty (3 axes — why this matters beyond just "notes")

### 1. Research process ≡ RLHF data pipeline
Every Claude proposal gets a stable ID (`AUR-D001`, `AUR-D002`…) and lands in `methodology/directions.jsonl`.
Every reasoning chain gets step-by-step `+/−/~` signs in `methodology/logic_chains.jsonl`.
A week later, JY retrospectively scores each direction (1–10 hindsight), enabling **DPO pair extraction from real research conversations** — not synthetic preference data.

Public RLHF datasets of *actual researcher-LLM dialogue with retrospective scoring* are essentially non-existent. Here it's a byproduct of working normally.

### 2. Real-world ground truth: **paper outcome**
Most LLM-agent benchmarks use synthetic tasks (MATH, HumanEval, GSM8K). Those measure narrow capability.
This project's primary signal is: *does the paper get accepted at IEEE TAFFC / NHB?*
That takes months, but when it lands, every iteration's `hindsight_score` is anchored to a real outcome — a far stronger training signal than preference pairs on toy tasks.

### 3. Cross-project cumulative methodology
`methodology/` is append-only across projects. When project 02 starts, it inherits the failure patterns of project 01 (see `methodology/insights_global.md`). Every new project calibrates against previously-learned blind spots. The engine gets better over time by construction.

---

## ⚖️ How this differs from existing tools

| Tool / Paradigm | What it does | What it's missing |
|---|---|---|
| **AutoGPT / ChatDev** | Fully autonomous agents | No human directing → wanders off thesis |
| **Obsidian / Notion** | Passive research notes | No iteration loop, no evaluation signal |
| **Jupyter / Colab** | Live code execution | No methodology accumulation, no paper-as-goal |
| **Zotero / Mendeley** | Literature management | No experiment tracking, no method novelty test |
| **AutoML / NAS** | Closed-loop optimization | No paper narrative, no cross-project transfer |
| **MLflow / W&B** | Experiment tracking | No human-AI collab structure, no RLHF angle |
| **This** | **Semi-autonomous + paper-anchored + RLHF-producing** | (by design, not these) |

---

## 📊 Current impact sanity-check

Project 01 — AU-RegionFormer Q1 (active, 9 iterations, ~1 day):

| Axis | Value |
|---|---|
| Best metric | **87.56%** linear probe (clean subset) |
| Gain over prior v1 trained model | **+7.86%p** (no-training, just smarter feature fusion) |
| Paper §4 sections with data | **5 of 7** (§4.1, 4.2, 4.3, 4.4, 4.5) |
| Counterintuitive finding | Mouth > Eyes in AI view, robust under Yonsei 298-person consensus — contradicts Jack 2012 PNAS |
| Negative findings logged | 2 (AU-Region redundancy, binary consensus weight) — prevents reviewer attacks |
| Direction IDs accumulated | 13 (`AUR-D001` ~ `D013`) |
| Logic chains accumulated | 7 (5 `+` pattern, 2 mixed) |

Venue probability estimates (current, after 9 iter):
- IEEE TAFFC (IF 11): **65–75%** (realistic)
- PNAS Social Sci (IF 9.4): 30–35%
- Nature Human Behaviour (IF 21): 22–28% (dream, needs psych co-author)

---

## 👥 Who this is for

- **A single researcher** who treats Claude as a full PhD student and wants structure, not chat chaos.
- **Labs / supervisors** who want to reproduce a methodology across students and projects without re-inventing the scaffold.
- **AI / HCI researchers** who need real human-LLM collaboration traces (not synthetic).
- **LLM research organizations** who want retrospectively-scored research-grade RLHF data.

---

## 🔧 Core workflow

```
┌────────────────────────── Every iteration ──────────────────────────┐
│                                                                     │
│  Phase 1 Design     →  experiments/exp_NNN/design.md                │
│  Phase 2 Implement  →  experiments/exp_NNN/run.py                   │
│  Phase 3 Evaluate   →  3-fold CV, random baseline, p-value          │
│  Phase 4 Record     →  state/leaderboard.jsonl append (one JSON)    │
│                    +→  methodology/directions.jsonl append          │
│                    +→  methodology/logic_chains.jsonl append        │
│  Phase 5 (every 5)  →  state/insights.md + insights_global.md       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each phase has rules enforced by `METHODOLOGY.md` and `.global/protocol/engine_loop.md`:
category diversity across iterations, strategy variety (a/b/c/d), no re-trying a failed method, forbidden protocol drift, kill switch honored.

---

## 📂 Structure

```
claude-research-engine/
├── README.md                  이 파일
├── CLAUDE.md                  전역 Claude 설정 (모든 프로젝트 상속)
├── METHODOLOGY.md             "Claude를 어떻게 쓸 것인가" 방법론 (10 섹션)
│
├── .global/
│   ├── rules_template/        target_metrics/constraints 템플릿
│   ├── templates/             session/experiment/logic_chain/direction_eval
│   └── protocol/
│       ├── engine_loop.md     iteration 5-phase 프로토콜 상세
│       └── report_format.md   실험 문서 8-section 표준
│
├── methodology/               ⭐ 프로젝트 횡단 RLHF 데이터 (누적)
│   ├── directions.jsonl       모든 direction (prefix ID로 프로젝트 구분)
│   ├── logic_chains.jsonl     모든 논리 체인
│   └── insights_global.md     cross-project 패턴
│
├── projects/
│   └── 01_au_regionformer_q1/
│       ├── README.md
│       ├── Q1_WORKING.md      논문 drafting + progress tracker
│       ├── rules/             target_metrics.md, constraints.md
│       ├── state/             leaderboard.jsonl, paper_tried.jsonl, insights.md
│       ├── experiments/       exp_NNN/ per iteration
│       ├── research_log/      sessions/, references/, claude_evaluation/
│       ├── src/analysis/      실험 코드
│       └── results/           작은 요약/figure만 (대용량은 외부)
│
└── scripts/
    ├── init_project.sh        새 프로젝트 scaffold
    └── kill_switch            자동화 중단 시그널 (touch으로 생성)
```

---

## 🚦 활성 프로젝트

| # | Name | Status | Target venue | Current best |
|---|------|--------|--------------|-------------|
| 01 | **au_regionformer_q2** | **Draft-ready** (후배 1저자 인계) | ESWA IF 7.5 | 87.56% (11 iter) |
| 02 | **emotion_agent** (Q1) | **Pre-setup** (ROADMAP 확정, Phase 0 대기) | IEEE TAFFC IF 11 / ICMI / ACII | — |
| 03 | *(reserved)* | — | — | — |

---

## 🛡️ Operating rules (excerpt)

- 평가 프로토콜 고정 (data split, seed, metric — 변경 즉시 실격)
- p < 0.05 미달은 개선으로 간주하지 않음
- 같은 방법 재시도 금지 (`paper_tried.jsonl` 체크)
- `scripts/kill_switch` 존재 시 즉시 중단
- 최근 10 iter 내 동일 category 연속 3회 초과 금지
- Claude의 **hindsight_score 자가 평가 금지** (bias)
- 실패 직렬 기록 필수 (success만 큐레이션 금지)

전체: `METHODOLOGY.md`, `.global/protocol/engine_loop.md`

---

## 🧭 새 세션 시작 순서 (Claude용)

1. 이 README
2. `CLAUDE.md` — 전역 원칙
3. `METHODOLOGY.md` — 엔진 운영
4. `projects/NN/README.md` — 해당 프로젝트 상태
5. `projects/NN/state/leaderboard.jsonl` — 현재 best
6. `projects/NN/research_log/sessions/` 최신 세션

---

## 🔒 License / IP

Private repo. JY 개인/랩 자산 포함. 연세대 298명 raw data 외부 공개 금지. 모델 가중치는 `.gitignore`로 제외 (원본 repo: `github.com/RyanAhn533/AU-RegionFormer`).

---

## ✍️ Citation (future, once paper is out)

```
@misc{ahn2026claude_research_engine,
  author = {Ahn, JY},
  title  = {claude-research-engine: a semi-autonomous research pipeline
           producing Q1 papers and RLHF data simultaneously},
  year   = {2026},
  note   = {Private framework. Companion to Ahn et al. (TAFFC/NHB submission).}
}
```
