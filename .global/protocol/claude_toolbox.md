# Protocol: Claude Toolbox Layer

> How Claude is invoked at every phase. This is the operational surface.
> Authority: `ARCHITECTURE.md` §13. Heritage (background reading): `.global/protocol/claude_leverage_2026.md`.
> Companion phase spec: `.global/protocol/state_machine.md`.

---

## 0. The six operational surfaces

| # | Surface | One-line |
|---|---------|----------|
| 1 | **Subagent** | Spawn isolated Claude → returns summary only. Context-pollution firewall + parallelism |
| 2 | **Plan mode** | Read-only Claude. Cannot mutate files. Used for design previews and risk-prone phases |
| 3 | **Adaptive thinking** | Budget {low, medium, high, max}. Claude self-titrates depth |
| 4 | **Prompt caching** | Server-side cache for stable prefix (CLAUDE.md, METHODOLOGY.md, ARCHITECTURE.md, leaderboard tail) |
| 5 | **Skills + Hooks** | Skills = codified workflows (`/distill`, `/debate`). Hooks = lifecycle scripts (pre-commit, post-tool) |
| 6 | **Memory bridge** | `~/.claude/projects/.../memory/` ↔ engine. Cross-session facts. Read-mostly across the line |

---

## 1. Adaptive thinking — when to spend each budget

```yaml
thinking_budget_rules:
  low:
    use_for:
      - syntactic repair (ImportError, ShapeMismatch)
      - mechanical file appends (Direction row, paper_tried row)
      - reading a single short log line
    why: thinking_overhead > task_complexity. wasting tokens.

  medium:
    use_for:
      - semantic repair (re-read design.md, write patch)
      - hypothesis registration (fill pre-registered fields)
      - reading one experiment's results.json
    why: needs comprehension, not deliberation.

  high:
    use_for:
      - METHOD_SEARCH (any of 5 strategies)
      - failure analysis (dynamics category)
      - insight summarizer (5-iter pattern extraction)
      - Direction rationale + risk drafting
    why: comparison across alternatives.

  max:
    use_for:
      - SELF_ATTACK (each of 3 adversarial personas)
      - 3-persona debate framework
      - first-principles redesign (Strategy e)
      - JY explicit "토론해줘 / 깊이 봐줘"
    why: adversarial depth. one-shot decisions. high blast radius.
```

**Anti-pattern**: setting `max` for an entire session. Budget is per-call. A `max`-budget Method Planner can still hand off to a `low`-budget syntactic repair on the next call.

---

## 2. Subagent routing — when to fork context

### 2.1 Always subagent

- **Strategy a (recent paper survey)** — fetching/reading N papers pollutes main with 30K+ tokens. Main only needs the comparison summary.
- **Insight Summarizer** — reads last 5 iterations of `leaderboard.jsonl`. Subagent returns the patterns, not the rows.
- **SELF_ATTACK personas (×3)** — adversarial agents must not see each other's reasoning. Parallel isolated subagents.
- **Leaderboard stats / leaderboard validation** — heavy read.

### 2.2 Never subagent

- **Live code editing** — Edit / Write tools are stateful; main owns them.
- **Hypothesis row fill** — too short to be worth a spawn.
- **Direction row append** — same.

### 2.3 Subagent contract

A subagent's response **MUST**:
- Be < 2K tokens
- Have a clear `verdict` / `summary` field
- Not return raw file contents (paths or excerpts only)
- Not start side-effects that outlive the spawn

If a subagent returns >2K tokens it has failed. Re-prompt or split.

---

## 3. Plan mode — when to lock to read-only

Plan mode is the formal mechanism for **"JY approval before destructive action."** Claude in plan mode cannot Edit / Write / Bash with side effects.

Use plan mode for:
- **METHOD_SEARCH** (Strategy a paper survey, Strategy c failure analysis) — Claude inspects code/state, proposes 3 method candidates, **exits plan mode only after JY approval**.
- **Large refactor previews** — Claude lays out the diff plan; JY accepts; then exits to apply.
- **Anything touching shared state** outside the project (git push, branch deletion, external API mutations).

Do **NOT** use plan mode for:
- IMPLEMENTATION (you need to write `run.py`)
- LOGGING (you need to append to JSONL)
- repair_router patches (mutation by definition)

`ExitPlanMode` is the explicit handoff. It is the engine's `HUMAN_APPROVAL` gate in tool form.

---

## 4. Prompt caching — what to cache

Stable prefix (changes < 1× per week):
- `CLAUDE.md` (project)
- `~/CLAUDE.md` (user global)
- `METHODOLOGY.md`, `ARCHITECTURE.md`
- `state/insights.md`

Cache the tail less aggressively:
- `state/leaderboard.jsonl` (grows each iteration)
- `state/paper_tried.jsonl` (grows each iteration)

In Claude Code: caching is automatic for system prompt and CLAUDE.md. For programmatic calls (`claude -p ...` or SDK), wrap stable context in `cache_control` markers.

**Anti-pattern**: re-reading the full leaderboard.jsonl into the prompt every call. Use a subagent that owns the file and returns summaries.

---

## 5. Skills + Hooks

### 5.1 Skills to keep using

| Skill | Where it helps the engine |
|-------|---------------------------|
| `/loop` | Overnight INSIGHT_UPDATE on cron |
| `/schedule` | Recurring `jy status` / `jy validate` |
| `/review` | Reviewer Simulator alternative for ad-hoc design.md review |
| `/security-review` | Pre-push audit when engine touches secrets |
| `/init` | New-project CLAUDE.md scaffold |

### 5.2 Hooks to add (settings.json)

```yaml
hooks_to_implement:
  pre_commit:
    - validate engine_state.json against schema
    - validate appended JSONL rows against their schemas
    - reject if any *.jsonl decreased in line count (=in-place edit)
  post_experiment:
    - append paper_tried.jsonl row if missing
    - append Direction row if Claude proposed any
    - write postmortem stub if Gate C verdict == fail
  pre_phase_transition:
    - read scripts/kill_switch → HALT if exists
    - read compute_budget.json → HALT if ≥ 95%
    - cmp current git_sha vs engine_state.git_sha → warn on drift
```

Status: not implemented yet (Tier 2 in `ARCHITECTURE.md` §11). Codify here so the spec is frozen before the code.

---

## 6. Memory bridge

### 6.1 What flows where

```
~/.claude/projects/-home-ajy/memory/
  ├── user_*.md       — JY profile, preferences           [engine reads]
  ├── feedback_*.md   — corrections / confirmations       [engine reads]
  ├── project_*.md    — cross-project context             [engine reads]
  └── reference_*.md  — external pointers                 [engine reads]

engine/projects/NN/state/
  ├── leaderboard.jsonl                                   [engine writes only]
  ├── paper_tried.jsonl                                   [engine writes only]
  ├── hypothesis_registry.jsonl                           [engine writes only]
  └── insights.md                                         [engine writes only]
```

**Rule**: engine never writes to `~/.claude/memory/`. User memory never appears as engine state. The two layers cross-reference by slug (`[[name]]`) but do not mutate each other.

### 6.2 Why this matters

User memory is **about JY** (career stage, scope preferences, working style). Engine state is **about experiments**. Mixing them = unreliable retrieval and accidental privacy bleed.

---

## 7. Quick reference card

When opening a new Claude session against this repo:

1. **Read order** (engine bootstrap):
   `README.md` → `CLAUDE.md` → `METHODOLOGY.md` → `ARCHITECTURE.md §0,§7,§13` → project `README.md` → `leaderboard.jsonl | tail`.
2. **First call** typically: `medium` thinking, no plan mode, no subagent. Just orient.
3. **Phase 1 (METHOD_SEARCH)**: plan mode + subagent per strategy + `high` thinking.
4. **SELF_ATTACK**: 3 subagents parallel, `max` thinking each.
5. **LOGGING**: hook-driven appends + main writes one Direction row at `low`.
