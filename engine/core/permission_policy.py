"""
permission_policy.py — resolves auto_mode <-> HUMAN_APPROVAL conflict.

External review §3 problem 2:
- HANDOFF says "auto mode, don't ask permission"
- METHODOLOGY says "destructive actions need JY consent"
Both are right but they need a *machine-readable* policy table, not prose.

Design:
- Policy is data, not code. Stored as append-only jsonl matching permission.schema.json.
- Each action_class maps to one of:
    auto_safe              → engine proceeds without prompting
    human_gate_required    → engine halts, surfaces a gate ticket
    hard_block             → engine refuses, never overridable
- Hooks in .claude/settings.json enforce this at the tool-call level (defense-in-depth).

Default policies are seeded by `seed_default_policies()` and can be amended by
appending new rows with `supersedes`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .append_only_logger import AppendOnlyLog

Policy = Literal["auto_safe", "human_gate_required", "hard_block"]


DEFAULT_POLICY_TABLE: list[dict] = [
    # ---- auto_safe: engine proceeds ----
    {"action_class": "read", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Read is always safe."},
    {"action_class": "analysis", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Pure compute."},
    {"action_class": "design_draft", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Draft files don't affect state."},
    {"action_class": "local_experiment_run", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Local to engine, budget-capped separately."},
    {"action_class": "local_result_parse", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Read-after-write of local artifacts."},
    {"action_class": "local_report_generation", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Reports never replace authoritative state."},
    {"action_class": "jsonl_append", "default_policy": "auto_safe", "override_allowed": False,
     "hook_enforcer": None, "rationale": "Append is the only legal write to jsonl."},
    {"action_class": "schema_validate", "default_policy": "auto_safe", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Pure check."},

    # ---- human_gate_required: destructive / claim-affecting ----
    {"action_class": "git_push", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/require_human_gate.sh", "rationale": "Push exposes work externally."},
    {"action_class": "branch_delete", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/require_human_gate.sh", "rationale": "Loses history."},
    {"action_class": "rm", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/require_human_gate.sh", "rationale": "Loses files."},
    {"action_class": "overwrite", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/require_human_gate.sh", "rationale": "Loses prior content."},
    {"action_class": "baseline_replacement", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": None, "rationale": "Baseline change affects all future comparisons."},
    {"action_class": "paper_claim_update", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": None, "rationale": "Narrative drift risk. JY decides."},
    {"action_class": "claim_status_change", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": None, "rationale": "Retiring or contradicting a claim is editorial."},
    {"action_class": "public_release", "default_policy": "human_gate_required", "override_allowed": False,
     "hook_enforcer": None, "rationale": "IRB / IP considerations."},
    {"action_class": "expensive_full_run", "default_policy": "human_gate_required", "override_allowed": True,
     "hook_enforcer": None, "rationale": "Budget impact. Override allowed in batch contexts."},

    # ---- hard_block: never, not even with JY override ----
    {"action_class": "git_force_push", "default_policy": "hard_block", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/require_human_gate.sh", "rationale": "Force push destroys history. Use revert."},
    {"action_class": "global_method_block", "default_policy": "hard_block", "override_allowed": False,
     "hook_enforcer": None, "rationale": "method_globally_blocked=true is human-only field. Append by JY directly."},
    {"action_class": "in_place_jsonl_edit", "default_policy": "hard_block", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/block_jsonl_in_place.sh", "rationale": "Append-only invariant."},
    {"action_class": "schema_modification", "default_policy": "hard_block", "override_allowed": False,
     "hook_enforcer": ".claude/hooks/block_jsonl_in_place.sh", "rationale": "Schemas are frozen post-Day-2."},
    {"action_class": "kill_switch_remove", "default_policy": "hard_block", "override_allowed": False,
     "hook_enforcer": None, "rationale": "JY removes kill switch by hand. Not the engine."},
]


def seed_default_policies(policy_log_path: str | Path, schema_dir: str | Path) -> int:
    """Write all DEFAULT_POLICY_TABLE rows if the log is empty."""
    log = AppendOnlyLog(
        path=policy_log_path,
        schema=Path(schema_dir) / "permission.schema.json",
    )
    existing = list(log.iter_rows())
    if existing:
        return 0
    n = 0
    for row in DEFAULT_POLICY_TABLE:
        log.append(row)
        n += 1
    return n


def resolve(policy_log_path: str | Path, action_class: str) -> tuple[Policy, dict]:
    """
    Get the current effective policy for an action_class, honoring `supersedes`.
    Returns (policy, full_row).
    """
    log_path = Path(policy_log_path)
    if not log_path.exists():
        raise FileNotFoundError(
            f"permission policy log not found at {log_path}. Run seed_default_policies first."
        )
    # walk all rows, build latest-per-action_class respecting supersedes
    latest: dict[str, dict] = {}
    superseded: set[str] = set()
    for row in _iter_jsonl(log_path):
        sup = row.get("supersedes")
        if sup:
            superseded.add(sup)
        latest[row["action_class"]] = row
    # remove explicitly superseded
    for row_id in superseded:
        for ac, row in list(latest.items()):
            if row.get("row_id") == row_id:
                del latest[ac]
    if action_class not in latest:
        raise KeyError(f"unknown action_class: {action_class}")
    row = latest[action_class]
    return row["default_policy"], row


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_action_allowed(
    policy_log_path: str | Path,
    action_class: str,
    *,
    auto_mode: bool,
    human_consent_given: bool = False,
) -> tuple[bool, str]:
    """
    Decide if an action is allowed right now.
    Returns (allowed, reason).
    """
    policy, row = resolve(policy_log_path, action_class)
    if policy == "hard_block":
        return False, f"hard_block: {row['rationale']}"
    if policy == "auto_safe":
        return True, "auto_safe"
    if policy == "human_gate_required":
        if human_consent_given:
            return True, "human_gate_required: consent given"
        if auto_mode and row.get("override_allowed"):
            return True, "human_gate_required: override_allowed under auto_mode"
        return False, f"human_gate_required: {row['rationale']}"
    return False, f"unknown policy: {policy}"
