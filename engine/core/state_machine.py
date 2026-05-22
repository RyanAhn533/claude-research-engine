"""
state_machine.py — phase FSM.

Design invariant: phase transitions are decided HERE, not by the LLM.
The LLM proposes a transition; the FSM validates pre/post-conditions before allowing it.

Usage:
    from engine.core.state_machine import StateMachine

    sm = StateMachine(project="02_emotion_agent")
    sm.current_phase()             # → "METHOD_SEARCH"
    sm.can_transition("HUMAN_APPROVAL")     # → (False, "missing hypothesis row for current exp")
    sm.transition("HYPOTHESIS_REGISTRATION", actor="claude", reason="strategy a complete")
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .append_only_logger import AppendOnlyLog, AppendOnlyViolation


# ── FSM definition. Data, not code. ────────────────────────────────────────
# Maps current_phase → set of legal next_phases.
LEGAL_TRANSITIONS: dict[str, set[str]] = {
    "BOOTSTRAP":                  {"STATUS_SYNC", "HALT"},
    "STATUS_SYNC":                {"METHOD_SEARCH", "HALT"},
    "METHOD_SEARCH":              {"HYPOTHESIS_REGISTRATION", "HALT"},
    "HYPOTHESIS_REGISTRATION":    {"HUMAN_APPROVAL", "AUTO_SAFE_APPROVAL", "METHOD_SEARCH", "HALT"},
    "HUMAN_APPROVAL":             {"IMPLEMENTATION", "METHOD_SEARCH", "HALT"},
    "AUTO_SAFE_APPROVAL":         {"IMPLEMENTATION", "HALT"},
    "IMPLEMENTATION":             {"EVALUATION", "REPAIR", "HALT"},
    "REPAIR":                     {"IMPLEMENTATION", "POSTMORTEM", "HALT"},
    "EVALUATION":                 {"CLAIM_LINKING", "HALT"},
    "CLAIM_LINKING":              {"CONDITIONAL_SELF_ATTACK", "LOGGING", "HALT"},
    "CONDITIONAL_SELF_ATTACK":    {"LOGGING", "HALT"},
    "LOGGING":                    {"PAPER_QUEUE_UPDATE", "HALT"},
    "PAPER_QUEUE_UPDATE":         {"INSIGHT_UPDATE", "METHOD_SEARCH", "HALT"},
    "INSIGHT_UPDATE":             {"METHOD_SEARCH", "HALT"},
    "POSTMORTEM":                 {"METHOD_SEARCH", "HALT"},
    "HALT":                       set(),  # terminal until human intervention
}


# Pre-conditions per phase. Each returns (ok, reason).
# These are *necessary* conditions to ENTER the phase.
def _check_method_search(sm: "StateMachine") -> tuple[bool, str]:
    return True, "always allowed (entry from BOOTSTRAP, INSIGHT, POSTMORTEM, etc.)"


def _check_hypothesis_registration(sm: "StateMachine") -> tuple[bool, str]:
    exp = sm.last_state().get("current_exp")
    if not exp:
        return False, "no current_exp set; METHOD_SEARCH must select one first"
    design_md = sm.repo_root / "projects" / sm.project / "experiments" / exp / "design.md"
    if not design_md.exists():
        return False, f"design.md not found at {design_md.relative_to(sm.repo_root)}"
    return True, "design.md present"


def _check_human_approval(sm: "StateMachine") -> tuple[bool, str]:
    # Must have a hypothesis row for current_exp with falsifiability_check=PASS
    exp = sm.last_state().get("current_exp")
    if not exp:
        return False, "no current_exp"
    hp_path = sm.state_dir / "hypothesis_registry.jsonl"
    if not hp_path.exists():
        return False, "no hypothesis_registry.jsonl"
    import json
    found = False
    for line in hp_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("exp_id") == exp and row.get("falsifiability_check") == "PASS":
            found = True
            break
    if not found:
        return False, f"no PASS hypothesis row for {exp}"
    return True, "hypothesis registered with falsifiability_check=PASS"


def _check_implementation(sm: "StateMachine") -> tuple[bool, str]:
    # Approval gate must have been crossed (we got here from HUMAN/AUTO_SAFE_APPROVAL).
    # plus reproducibility_manifest must be planned (created during IMPLEMENTATION itself, so just
    # confirm experiments/exp_NNN dir exists).
    exp = sm.last_state().get("current_exp")
    exp_dir = sm.repo_root / "projects" / sm.project / "experiments" / exp
    if not exp_dir.exists():
        return False, f"experiment dir missing: {exp_dir.relative_to(sm.repo_root)}"
    return True, "experiment dir present"


def _check_evaluation(sm: "StateMachine") -> tuple[bool, str]:
    exp = sm.last_state().get("current_exp")
    results = sm.repo_root / "projects" / sm.project / "experiments" / exp / "results.json"
    if not results.exists():
        return False, f"results.json missing for {exp}"
    return True, "results.json present"


def _check_logging(sm: "StateMachine") -> tuple[bool, str]:
    # Manifest must exist (this is the v2 hard rule from external review).
    exp = sm.last_state().get("current_exp")
    manifest_dir = sm.repo_root / "projects" / sm.project / "reproducibility_manifests"
    candidates = list(manifest_dir.glob(f"{exp}*"))
    if not candidates:
        return False, f"no reproducibility_manifest for {exp} (blocks LOGGING per METHODOLOGY §0.4)"
    return True, f"manifest present: {candidates[0].name}"


PRECONDITIONS = {
    "METHOD_SEARCH":              _check_method_search,
    "HYPOTHESIS_REGISTRATION":    _check_hypothesis_registration,
    "HUMAN_APPROVAL":             _check_human_approval,
    "AUTO_SAFE_APPROVAL":         _check_human_approval,  # same: needs registered hypothesis
    "IMPLEMENTATION":             _check_implementation,
    "EVALUATION":                 _check_evaluation,
    "LOGGING":                    _check_logging,
}


class TransitionRefused(RuntimeError):
    """Raised when a transition is illegal or pre-conditions fail."""


class StateMachine:
    def __init__(self, project: str, repo_root: Path | None = None):
        self.project = project
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.state_dir = self.repo_root / "projects" / project / "state"
        if not self.state_dir.exists():
            raise FileNotFoundError(
                f"project state dir missing: {self.state_dir}. Run `jy bootstrap` first."
            )
        self._state_log = AppendOnlyLog(
            path=self.state_dir / "engine_state.jsonl",
            schema=self.repo_root / "engine" / "schemas" / "state.schema.json",
        )

    # ── Reads ──────────────────────────────────────────────────────────────

    def last_state(self) -> dict[str, Any]:
        rows = list(self._state_log.iter_rows())
        if not rows:
            raise RuntimeError(f"no engine_state row for {self.project}. Run bootstrap.")
        return rows[-1]

    def current_phase(self) -> str:
        return self.last_state()["current_phase"]

    # ── Transition guard ───────────────────────────────────────────────────

    def can_transition(self, target_phase: str) -> tuple[bool, str]:
        # 1. kill switch (always blocks except HALT)
        if self._kill_switch_present() and target_phase != "HALT":
            return False, "kill switch present → only HALT allowed"

        # 2. legal next-phase per FSM
        cur = self.current_phase()
        if target_phase not in LEGAL_TRANSITIONS.get(cur, set()):
            return False, f"illegal transition: {cur} → {target_phase}. Legal: {sorted(LEGAL_TRANSITIONS.get(cur, set()))}"

        # 3. budget check (only for compute-spending phases)
        if target_phase in ("IMPLEMENTATION", "EVALUATION", "METHOD_SEARCH"):
            ok, reason = self._budget_ok()
            if not ok:
                return False, f"budget: {reason}"

        # 4. phase-specific pre-conditions
        pre = PRECONDITIONS.get(target_phase)
        if pre is not None:
            ok, reason = pre(self)
            if not ok:
                return False, f"precondition: {reason}"

        return True, "ok"

    # ── Transition execution ───────────────────────────────────────────────

    def transition(
        self,
        target_phase: str,
        *,
        actor: str = "claude",
        reason: str = "",
        new_exp: str | None = None,
        pending_human_gate: str | None = None,
    ) -> dict[str, Any]:
        ok, why = self.can_transition(target_phase)
        if not ok:
            raise TransitionRefused(f"{self.current_phase()} → {target_phase}: {why}")

        prev = self.last_state()
        new_row = {
            "project": self.project,
            "current_phase": target_phase,
            "iter": prev["iter"] + (1 if target_phase == "METHOD_SEARCH" else 0),
            "current_exp": new_exp if new_exp is not None else prev.get("current_exp"),
            "direction_prefix": prev.get("direction_prefix", "XXX"),
            "kill_switch_present": self._kill_switch_present(),
            "auto_mode": prev.get("auto_mode", False),
            "budget": prev.get("budget", {}),
            "pending_human_gate": pending_human_gate,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Carry over supersedes nullable — this is a new row, not a correction.
        try:
            written = self._state_log.append(new_row)
        except AppendOnlyViolation:
            raise
        return written

    def halt(self, reason: str) -> dict[str, Any]:
        """Force-halt. Always permitted from any phase."""
        return self.transition("HALT", actor="system", reason=reason)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _kill_switch_present(self) -> bool:
        return (self.repo_root / "scripts" / "kill_switch").exists()

    def _budget_ok(self) -> tuple[bool, str]:
        b = self.last_state().get("budget") or {}
        used = b.get("gpu_hours_used", 0)
        total = b.get("gpu_hours_total")
        if total is None or total == 0:
            return True, "no budget cap set"
        if used >= total:
            return False, f"gpu_hours_used ({used}) >= gpu_hours_total ({total})"
        if total and used / total > 0.95:
            # warn but proceed; the engine halts at exhaustion not at warn
            return True, f"warn: {used / total:.0%} used"
        return True, "ok"
