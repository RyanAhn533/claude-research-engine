#!/usr/bin/env python3
"""
Regression test for the 6 bugs flagged in the external review of v2 engine.

Each test demonstrates the OLD behavior would have failed AND that the new
behavior succeeds. Run alongside test_tier2.py.

Usage:
    python -m engine.tests.test_bug_fixes
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ── #1: AppendOnlyLog race condition ──────────────────────────────────────

def _race_worker(args):
    """Run in subprocess. Append N rows to the shared log."""
    log_path, schema_path, worker_id, n_rows = args
    sys.path.insert(0, str(REPO_ROOT))
    from engine.core.append_only_logger import AppendOnlyLog

    log = AppendOnlyLog(path=log_path, schema=schema_path)
    for i in range(n_rows):
        log.append({
            "project": "99_race_test",
            "current_phase": "BOOTSTRAP",
            "iter": worker_id * n_rows + i,
            "current_exp": None,
            "direction_prefix": "RAC",
            "kill_switch_present": False,
            "auto_mode": False,
            "budget": {"gpu_hours_used": 0.0, "gpu_hours_total": 10.0},
            "pending_human_gate": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "supersedes": None,
        })
    return worker_id


def test_appendonly_race():
    section("#1: AppendOnlyLog concurrent append → no hash chain corruption")
    from engine.core.append_only_logger import AppendOnlyLog
    schema = REPO_ROOT / "engine" / "schemas" / "state.schema.json"

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "race.jsonl"
        n_workers = 4
        n_per = 25
        with mp.Pool(n_workers) as pool:
            results = pool.map(_race_worker,
                               [(str(log_path), str(schema), w, n_per) for w in range(n_workers)])

        assert len(results) == n_workers
        log = AppendOnlyLog(path=log_path, schema=schema)
        rows = list(log.iter_rows())
        expected = n_workers * n_per
        assert len(rows) == expected, f"expected {expected} rows, got {len(rows)}"
        n = log.verify_chain()
        assert n == expected
        print(f"  ✓ {n_workers} workers × {n_per} appends = {expected} rows, chain verified")


# ── #2: EVALUATION precondition ───────────────────────────────────────────

def _bootstrap_throwaway(name: str) -> Path:
    pdir = REPO_ROOT / "projects" / name
    if pdir.exists():
        shutil.rmtree(pdir)
    res = subprocess.run(
        [sys.executable, "-m", "engine.cli.jy", "bootstrap",
         "--project", name, "--prefix", "BUG", "--gpu-hours", "10"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        raise RuntimeError("bootstrap failed")
    return pdir


def _write_hypothesis(pdir: Path, exp_id: str) -> None:
    """Write a minimal PASS-falsifiability hypothesis row to satisfy HUMAN_APPROVAL gate."""
    from engine.core.append_only_logger import AppendOnlyLog
    schema = REPO_ROOT / "engine" / "schemas" / "hypothesis.schema.json"
    log = AppendOnlyLog(path=pdir / "state" / "hypothesis_registry.jsonl", schema=schema)
    log.append({
        "hypothesis_id": f"H_{exp_id}",
        "exp_id": exp_id,
        "primary": "fix verification claim — falsifiable",
        "null_hypothesis": "fix does not work as advertised",
        "success_criterion": {"metric": "dummy", "direction": "higher_is_better", "delta_threshold": 1.0},
        "failure_implication": "regression test fails",
        "falsifiability_check": "PASS",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "outcome": None,
        "linked_claim": None,
    })


def test_evaluation_precondition():
    section("#2: EVALUATION entry no longer requires results.json")
    from engine.core.state_machine import StateMachine, TransitionRefused

    pdir = _bootstrap_throwaway("99_bugfix_eval")
    exp = "exp_001_eval"
    exp_dir = pdir / "experiments" / exp
    exp_dir.mkdir(parents=True)
    (exp_dir / "design.md").write_text("# design")
    _write_hypothesis(pdir, exp)

    sm = StateMachine(project="99_bugfix_eval", repo_root=REPO_ROOT)
    sm.transition("STATUS_SYNC", reason="t"); sm.transition("METHOD_SEARCH", reason="t", new_exp=exp)
    sm.transition("HYPOTHESIS_REGISTRATION", reason="t"); sm.transition("HUMAN_APPROVAL", reason="t")
    sm.transition("IMPLEMENTATION", reason="t")

    # Try EVALUATION with NO run.py and NO manifest — should fail with precondition error
    ok, why = sm.can_transition("EVALUATION")
    assert not ok and "run.py" in why, f"expected run.py error, got: {why}"
    print(f"  ✓ EVALUATION blocked w/o run.py: {why}")

    # Add run.py — still needs manifest
    (exp_dir / "run.py").write_text("print('ok')")
    ok, why = sm.can_transition("EVALUATION")
    assert not ok and "manifest" in why, f"expected manifest error, got: {why}"
    print(f"  ✓ EVALUATION blocked w/o manifest: {why}")

    # Add manifest — should now pass (NO results.json yet, which is the whole point)
    (pdir / "reproducibility_manifests" / f"{exp}.yaml").parent.mkdir(parents=True, exist_ok=True)
    (pdir / "reproducibility_manifests" / f"{exp}.yaml").write_text("exp_id: " + exp + "\n")
    assert not (exp_dir / "results.json").exists(), "test setup invariant"
    sm.transition("EVALUATION", reason="t")
    print(f"  ✓ EVALUATION entered WITHOUT results.json (the whole bug)")

    # Now LOGGING should be the one that needs results.json
    sm.transition("CLAIM_LINKING", reason="t")
    ok, why = sm.can_transition("LOGGING")
    assert not ok and "results.json" in why, f"expected LOGGING to need results.json, got: {why}"
    print(f"  ✓ LOGGING correctly demands results.json: {why}")

    (exp_dir / "results.json").write_text('{"ok": true}')
    sm.transition("LOGGING", reason="t")
    print(f"  ✓ LOGGING entered once results.json exists")

    shutil.rmtree(pdir)


# ── #3: AUTO_SAFE_APPROVAL requires auto_mode ─────────────────────────────

def test_auto_safe_approval_gate():
    section("#3: AUTO_SAFE_APPROVAL refuses manual mode")
    from engine.core.state_machine import StateMachine
    from engine.core.append_only_logger import AppendOnlyLog

    pdir = _bootstrap_throwaway("99_bugfix_auto")
    exp = "exp_001_auto"
    (pdir / "experiments" / exp).mkdir(parents=True)
    (pdir / "experiments" / exp / "design.md").write_text("# design")
    _write_hypothesis(pdir, exp)

    sm = StateMachine(project="99_bugfix_auto", repo_root=REPO_ROOT)
    sm.transition("STATUS_SYNC", reason="t"); sm.transition("METHOD_SEARCH", reason="t", new_exp=exp)
    sm.transition("HYPOTHESIS_REGISTRATION", reason="t")

    # Default bootstrap sets auto_mode=False → AUTO_SAFE_APPROVAL must refuse
    ok, why = sm.can_transition("AUTO_SAFE_APPROVAL")
    assert not ok and "auto_mode" in why, f"expected auto_mode error, got: {why}"
    print(f"  ✓ manual mode → AUTO_SAFE_APPROVAL blocked: {why}")

    # HUMAN_APPROVAL still works (unchanged)
    ok, why = sm.can_transition("HUMAN_APPROVAL")
    assert ok, why
    print(f"  ✓ HUMAN_APPROVAL still passes (unchanged behavior)")

    # Flip auto_mode=True by appending a new state row
    state_log = AppendOnlyLog(
        path=pdir / "state" / "engine_state.jsonl",
        schema=REPO_ROOT / "engine" / "schemas" / "state.schema.json",
    )
    cur = list(state_log.iter_rows())[-1]
    new = dict(cur); new["auto_mode"] = True; new["updated_at"] = datetime.now(timezone.utc).isoformat()
    new.pop("row_id", None); new.pop("prev_hash", None); new["supersedes"] = None
    state_log.append(new)

    sm2 = StateMachine(project="99_bugfix_auto", repo_root=REPO_ROOT)
    ok, why = sm2.can_transition("AUTO_SAFE_APPROVAL")
    assert ok, f"expected auto_mode=True to allow AUTO_SAFE_APPROVAL, got: {why}"
    print(f"  ✓ auto_mode=True → AUTO_SAFE_APPROVAL allowed: {why}")

    shutil.rmtree(pdir)


# ── #4: TOST renamed → practical_effect_passed ────────────────────────────

def test_tost_renamed():
    section("#4: gate_c uses practical_effect_passed (not tost_passed)")
    from engine.gates.gate_c import evaluate, GateCResult
    import dataclasses

    fields = {f.name for f in dataclasses.fields(GateCResult)}
    assert "practical_effect_passed" in fields, fields
    assert "tost_passed" not in fields, fields
    print(f"  ✓ GateCResult field renamed in dataclass")

    # Schema check
    schema = json.loads((REPO_ROOT / "engine/schemas/experiment.schema.json").read_text())
    props = schema["properties"]
    assert "practical_effect_passed" in props
    assert "tost_passed" not in props
    print(f"  ✓ experiment.schema.json updated")

    schema = json.loads((REPO_ROOT / "engine/schemas/hypothesis.schema.json").read_text())
    props = schema["properties"]["outcome"]["properties"]
    assert "practical_effect_passed" in props
    assert "tost_passed" not in props
    print(f"  ✓ hypothesis.schema.json updated")

    # Fixtures
    for f in ["experiment.valid.json", "hypothesis.valid.json"]:
        text = (REPO_ROOT / "engine/tests/fixtures" / f).read_text()
        assert "tost_passed" not in text, f"{f} still has tost_passed"
        assert "practical_effect_passed" in text, f"{f} missing practical_effect_passed"
    print(f"  ✓ fixtures updated")

    # Functional smoke (paper_ready needs scipy + n=7)
    try:
        import scipy  # noqa
        r = evaluate(
            new_results=[70.5, 66.2, 73.3, 69.8, 70.2, 71.5, 68.9],
            baseline_results=[48.0, 47.5, 48.5, 47.0, 48.2, 47.8, 48.3],
            stage="paper_ready", delta_threshold=10.0,
        )
        assert r.practical_effect_passed is True, r
        print(f"  ✓ practical_effect_passed populated correctly: {r.practical_effect_passed}")
    except ImportError:
        print(f"  ℹ scipy not installed — functional check skipped")


# ── #5: Bash hook blocks jsonl mutation ───────────────────────────────────

def _hook_decision(cmd: str) -> tuple[int, str]:
    """Run require_human_gate.sh with cmd. Return (exit_code, stderr)."""
    hook = REPO_ROOT / ".claude/hooks/require_human_gate.sh"
    payload = json.dumps({"tool_input": {"command": cmd}})
    res = subprocess.run(
        ["bash", str(hook)],
        input=payload, capture_output=True, text=True,
    )
    return res.returncode, res.stderr


def test_hook_blocks_jsonl_mutation():
    section("#5: Bash hook blocks jsonl mutation routes")
    cases_blocked = [
        "echo '{}' >> projects/foo/state/leaderboard.jsonl",
        "echo '{}' > projects/foo/state/leaderboard.jsonl",
        "cat row.json | tee -a state/leaderboard.jsonl",
        "sed -i 's/old/new/' state/leaderboard.jsonl",
        "perl -pi -e 's/x/y/' state/foo.jsonl",
        "python -c \"open('foo.jsonl','a').write('x')\"",
        "python -c \"open('foo.jsonl','w').write('x')\"",
    ]
    for cmd in cases_blocked:
        rc, err = _hook_decision(cmd)
        assert rc == 2, f"expected BLOCK for: {cmd!r}, got rc={rc}, stderr={err[:200]}"
        print(f"  ✓ blocked: {cmd}")

    cases_allowed = [
        "cat state/leaderboard.jsonl",
        "head -5 state/leaderboard.jsonl",
        "wc -l state/leaderboard.jsonl",
        "ls *.jsonl",
        "python -c \"open('foo.jsonl','r').read()\"",
        "cp results.json /tmp/",
        "echo hello",
    ]
    for cmd in cases_allowed:
        rc, err = _hook_decision(cmd)
        assert rc == 0, f"expected ALLOW for: {cmd!r}, got rc={rc}, stderr={err[:200]}"
        print(f"  ✓ allowed: {cmd}")


# ── #6: git_sha hard-fail in non-git context ──────────────────────────────

def test_git_sha_hard_fail():
    section("#6: reproducibility_manifest hard-fails outside git repo")
    from engine.core.reproducibility_manifest import generate

    with tempfile.TemporaryDirectory() as td:
        env_lock = Path(td) / "env.lock"
        env_lock.write_text("x==1.0\n")
        try:
            generate(
                exp_id="exp_001_smoke",
                random_seeds={"python": 1, "numpy": 1, "torch": 1},
                env_lock_path=env_lock,
                data_hash="sha256:" + "f" * 64,
                repo_root=Path(td),  # NOT a git repo
            )
            raise AssertionError("generate() must raise in non-git context")
        except ValueError as e:
            assert "git_sha" in str(e), e
            print(f"  ✓ non-git context refused: {e}")

    # Sanity: real repo still works
    mf = generate(
        exp_id="exp_001_smoke",
        random_seeds={"python": 1, "numpy": 1, "torch": 1},
        env_lock_path="env.lock",  # doesn't have to exist for this check
        data_hash="sha256:" + "f" * 64,
        repo_root=REPO_ROOT,
    )
    assert mf.git_sha and mf.git_sha != "unknown"
    print(f"  ✓ git context still works: git_sha={mf.git_sha[:7]}")


def main():
    test_appendonly_race()
    test_evaluation_precondition()
    test_auto_safe_approval_gate()
    test_tost_renamed()
    test_hook_blocks_jsonl_mutation()
    test_git_sha_hard_fail()
    print("\n✅ All 6 bug-fix regression tests passed")


if __name__ == "__main__":
    main()
