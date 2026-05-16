# Protocol: State Machine (v2)

> Phase-level FSM. The State Machine is **the** orchestrator. LLM agents never decide phase transitions.
> Companion: `engine/schemas/state.schema.json`. Authority: `ARCHITECTURE.md` §2.

## Phases

```
BOOTSTRAP
  → METHOD_SEARCH
  → HYPOTHESIS_REGISTRATION
  → HUMAN_APPROVAL
  → IMPLEMENTATION
  → EVALUATION
  → LOGGING
  → INSIGHT_UPDATE (every 5 iterations)
  → METHOD_SEARCH (loop)
```

## Failure transitions

| From | Trigger | To |
|------|---------|----|
| `IMPLEMENTATION` | Gate B fail (syntactic/semantic/dynamics) | `REPAIR` |
| `REPAIR` | repair success | `IMPLEMENTATION` |
| `REPAIR` | retry budget exhausted | `POSTMORTEM` |
| `POSTMORTEM` | postmortem.md written + paper_tried logged | `METHOD_SEARCH` |
| any | Gate A `protocol_violation` | `HALT` |
| any | Compute Budget Tracker `halt_at_95_percent` | `HALT` |
| any | `scripts/kill_switch` exists | `HALT` |

`HALT` requires human intervention. Restart: `jy bootstrap` after clearing the cause.

## Pre-transition checks (every phase boundary)

The State Machine, before entering each phase, runs:

1. **Budget check** — Compute Budget Tracker reports usage. If ≥ 95%, transition to `HALT`.
2. **Kill switch check** — if `scripts/kill_switch` exists, `HALT`.
3. **Schema validation** — any persisted state matches its `engine/schemas/*.json`.
4. **Append-only audit** — verify no in-place edits since last checkpoint (file hash).

If any check fails, transition is blocked. Failures are logged to `state/gate_log.jsonl` with `gate: "A"` and `violation_type`.

## Phase entry/exit contract

Every phase has:

- **Preconditions** — what must be true before entry (e.g., METHOD_SEARCH requires `paper_tried.jsonl` readable).
- **Task DAG** — fixed intra-phase tasks (see `ARCHITECTURE.md` §2.2).
- **Postconditions** — what must be written before exit (e.g., LOGGING requires an append to `leaderboard.jsonl`).
- **Phase artifact** — a JSON record appended to `phase_history` inside `engine_state.json`.

| Phase | Entry precondition | Exit postcondition |
|-------|--------------------|--------------------|
| `BOOTSTRAP` | none | engine_state.json valid |
| `METHOD_SEARCH` | `paper_tried.jsonl` exists, `insights.md` exists | 3 method candidates produced |
| `HYPOTHESIS_REGISTRATION` | candidates present | each candidate has hypothesis row in `hypothesis_registry.jsonl` with `falsifiability_check: PASS` |
| `HUMAN_APPROVAL` | hypotheses registered | one candidate selected (or all rejected → re-enter METHOD_SEARCH) |
| `IMPLEMENTATION` | approved candidate | Gate B `pass` |
| `EVALUATION` | run artifacts present | Gate C verdict written |
| `LOGGING` | Gate C verdict | leaderboard.jsonl + paper_tried.jsonl appended; hypothesis outcome updated |
| `INSIGHT_UPDATE` | iteration % 5 == 0 | insights.md updated |

## Append-only enforcement

- `leaderboard.jsonl`, `paper_tried.jsonl`, `hypothesis_registry.jsonl`, `gate_log.jsonl`, `directions.jsonl`, `logic_chains.jsonl` are **append-only**.
- Corrections are new rows referencing the prior row's key (`supersedes` / `supersedes_exp_id`).
- The State Machine refuses to enter `LOGGING` if it detects an in-place edit (line count decreased OR earlier line hash changed).

## Engine version

This document describes engine **v2.0**. v1 (5-phase loop in `engine_loop.md`) is retained for trace. Mixed-version runs are not supported — projects choose one in their `rules/`.
