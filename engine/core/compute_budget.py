"""
compute_budget.py — budget accounting for the engine.

Lives on the engine_state row's `budget` field (state.schema.json).
State machine calls `check_before_phase()` before any compute-spending transition.
Experiments call `record_usage()` at the end of their run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class BudgetStatus:
    ok: bool
    reason: str
    used: float
    total: float
    fraction: float  # 0..1+


class BudgetExceeded(RuntimeError):
    pass


def check(used: float, total: float, *, halt_at: float = 1.0, warn_at: float = 0.8) -> BudgetStatus:
    if total is None or total <= 0:
        return BudgetStatus(True, "no budget cap", used, 0, 0)
    frac = used / total
    if frac >= halt_at:
        return BudgetStatus(False, f"exhausted: {used}/{total} ({frac:.0%})", used, total, frac)
    if frac >= warn_at:
        return BudgetStatus(True, f"warn: {used}/{total} ({frac:.0%})", used, total, frac)
    return BudgetStatus(True, "ok", used, total, frac)


def record_usage(
    current_budget: dict,
    *,
    gpu_hours: float = 0,
    cost_usd: float = 0,
    wall_clock_hours: float = 0,
) -> dict:
    """Pure: returns a new budget dict with usage added. Caller appends a new state row with it."""
    b = dict(current_budget or {})
    b["gpu_hours_used"] = b.get("gpu_hours_used", 0) + gpu_hours
    b["cost_usd_used"] = b.get("cost_usd_used", 0) + cost_usd
    b["wall_clock_hours_used"] = b.get("wall_clock_hours_used", 0) + wall_clock_hours
    return b


def per_experiment_check(current_budget: dict, gpu_hours_this_exp: float) -> BudgetStatus:
    cap = (current_budget or {}).get("per_experiment_cap_gpu_hours")
    if cap is None or cap <= 0:
        return BudgetStatus(True, "no per-exp cap", gpu_hours_this_exp, 0, 0)
    return check(gpu_hours_this_exp, cap, halt_at=1.0, warn_at=0.9)
