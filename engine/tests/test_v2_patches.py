"""
test_v2_patches.py — verify all 6 external-review patches actually work.

Each test is a positive AND negative case where applicable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _section(name: str) -> None:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────
# PATCH #1: AppendOnlyLog lock scope (race)
# ──────────────────────────────────────────────────────────────────
def _spawn_appender(args):
    """Run in subprocess — append N rows to same log."""
    log_path, schema, n, label = args
    sys.path.insert(0, str(REPO))
    from engine.core.append_only_logger import AppendOnlyLog
    log = AppendOnlyLog(log_path, schema)
    for i in range(n):
        log.append({
            "row_id": f"{label}_{i:03d}",
            "project": "99_race_test",
            "current_phase": "METHOD_SEARCH",
            "iter": i,
            "auto_mode": False,
            "updated_at": "2026-05-22T00:00:00+00:00",
        })
    return label, n


def test_patch_1_append_only_race():
    _section("PATCH #1: AppendOnlyLog concurrent append (race)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log_path = tmp / "engine_state.jsonl"
        schema = REPO / "engine/schemas/state.schema.json"

        # spawn 4 processes, each appends 10 rows concurrently
        N_PROC, N_ROWS = 4, 10
        from engine.core.append_only_logger import AppendOnlyLog
        with ProcessPoolExecutor(max_workers=N_PROC) as ex:
            futures = [
                ex.submit(_spawn_appender, (str(log_path), str(schema), N_ROWS, f"p{i}"))
                for i in range(N_PROC)
            ]
            for f in as_completed(futures):
                f.result()

        # verify total rows
        log = AppendOnlyLog(log_path, schema)
        n = log.verify_chain()
        if n != N_PROC * N_ROWS:
            _fail(f"row count mismatch: got {n}, expected {N_PROC*N_ROWS}")
        _ok(f"{N_PROC} procs × {N_ROWS} rows = {n} total, chain verified (no race)")


# ──────────────────────────────────────────────────────────────────
# PATCH #2: StateMachine EVALUATION precondition
# ──────────────────────────────────────────────────────────────────
def test_patch_2_evaluation_precondition():
    _section("PATCH #2: EVALUATION entry only needs run.py + manifest (not results.json)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # mini repo layout
        proj = tmp / "projects" / "test_proj"
        (proj / "state").mkdir(parents=True)
        (proj / "experiments" / "exp_001").mkdir(parents=True)
        (proj / "reproducibility_manifests").mkdir(parents=True)
        # symlink schemas/gates so state_machine finds them
        (tmp / "engine").symlink_to(REPO / "engine")

        from engine.core.state_machine import StateMachine, _check_evaluation, _check_claim_linking
        sm = StateMachine("test_proj", repo_root=tmp)

        # Hack: give it a current_exp
        class FakeState(dict): pass
        sm.last_state = lambda: {"current_exp": "exp_001"}

        # Case A: nothing exists → reject
        ok, reason = _check_evaluation(sm)
        if ok: _fail(f"should have rejected (no run.py): {reason}")
        _ok(f"reject without run.py: {reason}")

        # Case B: run.py exists but no manifest → reject
        (proj / "experiments/exp_001/run.py").write_text("# stub")
        ok, reason = _check_evaluation(sm)
        if ok: _fail(f"should have rejected (no manifest): {reason}")
        _ok(f"reject without manifest: {reason}")

        # Case C: both exist, no results.json → SHOULD PASS (this is the fix)
        (proj / "reproducibility_manifests/exp_001_manifest.json").write_text("{}")
        ok, reason = _check_evaluation(sm)
        if not ok: _fail(f"should have ALLOWED (run.py+manifest enough): {reason}")
        _ok(f"allow without results.json: {reason}")

        # Case D: CLAIM_LINKING requires results.json
        ok, reason = _check_claim_linking(sm)
        if ok: _fail(f"CLAIM_LINKING should reject without results.json: {reason}")
        _ok(f"CLAIM_LINKING rejects without results.json: {reason}")

        (proj / "experiments/exp_001/results.json").write_text("{}")
        ok, reason = _check_claim_linking(sm)
        if not ok: _fail(f"CLAIM_LINKING should pass with results.json: {reason}")
        _ok(f"CLAIM_LINKING passes with results.json: {reason}")


# ──────────────────────────────────────────────────────────────────
# PATCH #3: AUTO_SAFE_APPROVAL strictness
# ──────────────────────────────────────────────────────────────────
def test_patch_3_auto_safe_approval():
    _section("PATCH #3: AUTO_SAFE_APPROVAL checks auto_mode + permission policy")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        proj = tmp / "projects" / "test_proj"
        (proj / "state").mkdir(parents=True)
        (proj / "experiments" / "exp_001").mkdir(parents=True)
        (tmp / "engine").symlink_to(REPO / "engine")

        # hypothesis PASS row required
        hp = proj / "state" / "hypothesis_registry.jsonl"
        hp.write_text(json.dumps({"exp_id": "exp_001", "falsifiability_check": "PASS"}) + "\n")

        # permission policy: local_experiment_run = auto_safe
        pp = proj / "state" / "permission_policy.jsonl"
        pp.write_text("\n".join([
            json.dumps({"action_class": "local_experiment_run", "decision": "auto_safe"}),
            json.dumps({"action_class": "expensive_full_run", "decision": "human_gate_required"}),
        ]) + "\n")

        from engine.core.state_machine import StateMachine, _check_auto_safe_approval
        sm = StateMachine("test_proj", repo_root=tmp)

        # Case A: auto=False → reject (this is the bug being fixed)
        sm.last_state = lambda: {"current_exp": "exp_001", "auto": False}
        ok, reason = _check_auto_safe_approval(sm)
        if ok: _fail(f"auto=False should reject: {reason}")
        _ok(f"auto=False rejected: {reason}")

        # Case B: auto=True, default required = local_experiment_run (auto_safe) → pass
        sm.last_state = lambda: {"current_exp": "exp_001", "auto": True, "gate": None}
        ok, reason = _check_auto_safe_approval(sm)
        if not ok: _fail(f"auto=True + auto_safe action should pass: {reason}")
        _ok(f"auto=True + auto_safe action passed: {reason}")

        # Case C: paper_ready gate → triggers expensive_full_run requirement → reject
        sm.last_state = lambda: {"current_exp": "exp_001", "auto": True, "gate": "C_paper_ready"}
        ok, reason = _check_auto_safe_approval(sm)
        if ok: _fail(f"expensive_full_run is human_gate, should reject: {reason}")
        _ok(f"expensive_full_run rejected: {reason}")


# ──────────────────────────────────────────────────────────────────
# PATCH #4: Gate C TOST naming
# ──────────────────────────────────────────────────────────────────
def test_patch_4_gate_c_naming():
    _section("PATCH #4: Gate C exposes practical_effect_passed + true tost_equivalent")
    from engine.gates.gate_c import evaluate

    # baseline ~ 70, new ~ 73 → 3.0 improvement
    baseline = [70.1, 69.8, 70.5, 70.2, 69.9, 70.0, 70.3]
    new      = [73.2, 72.9, 73.5, 73.0, 73.1, 72.8, 73.4]

    r = evaluate(
        new_results=new, baseline_results=baseline,
        stage="paper_ready", delta_threshold=1.0,
    )
    if r.practical_effect_passed is None:
        _fail("practical_effect_passed should be set")
    _ok(f"practical_effect_passed = {r.practical_effect_passed}")
    if r.tost_equivalent is None:
        _fail("tost_equivalent should be set when delta_threshold given")
    _ok(f"tost_equivalent = {r.tost_equivalent} (large delta → not equivalent)")
    if r.tost_passed != r.practical_effect_passed:
        _fail(f"tost_passed alias mismatch: {r.tost_passed} vs {r.practical_effect_passed}")
    _ok(f"tost_passed alias = practical_effect_passed = {r.tost_passed}")

    # Now test TRUE equivalence: tiny delta within threshold
    new_eq = [70.15, 69.85, 70.55, 70.25, 69.95, 70.05, 70.35]  # near-identical
    r2 = evaluate(
        new_results=new_eq, baseline_results=baseline,
        stage="paper_ready", delta_threshold=1.0,
    )
    _ok(f"tiny delta: tost_equivalent={r2.tost_equivalent}, "
        f"p_lo={r2.tost_equivalent_p_lo:.3f}, p_hi={r2.tost_equivalent_p_hi:.3f}")
    if not r2.tost_equivalent:
        _fail("tiny delta should be equivalent")
    _ok("TOST correctly identifies equivalence")


# ──────────────────────────────────────────────────────────────────
# PATCH #5: Hook blocks bash bypass
# ──────────────────────────────────────────────────────────────────
def test_patch_5_hook_bash_bypass():
    _section("PATCH #5: hook blocks Bash bypass (>>, tee -a, sed -i, python open(...,a))")
    hook = REPO / ".claude/hooks/block_jsonl_in_place.sh"
    if not hook.exists() or not os.access(hook, os.X_OK):
        _fail("hook not executable")

    def run_hook(input_dict: dict) -> tuple[int, str]:
        proc = subprocess.run(
            [str(hook)],
            input=json.dumps(input_dict),
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stderr.strip()

    # Each case: input + expected exit code (2 = blocked)
    cases = [
        ("Bash >> jsonl", {"tool_name": "Bash", "tool_input": {"command": "echo '{}' >> projects/foo/state/leaderboard.jsonl"}}, 2),
        ("Bash tee -a", {"tool_name": "Bash", "tool_input": {"command": "echo x | tee -a state/log.jsonl"}}, 2),
        ("Bash sed -i jsonl", {"tool_name": "Bash", "tool_input": {"command": "sed -i 's/x/y/' state/log.jsonl"}}, 2),
        ("Bash python open(a) jsonl", {"tool_name": "Bash", "tool_input": {"command": "python3 -c 'open(\"x.jsonl\",\"a\").write(\"y\")'"}}, 2),
        ("Bash truncate > jsonl", {"tool_name": "Bash", "tool_input": {"command": "echo > state/log.jsonl"}}, 2),
        ("Edit jsonl", {"tool_name": "Edit", "tool_input": {"file_path": "state/log.jsonl"}}, 2),
        ("Bash safe ls (allowed)", {"tool_name": "Bash", "tool_input": {"command": "ls state/log.jsonl"}}, 0),
        ("Bash cat jsonl (allowed)", {"tool_name": "Bash", "tool_input": {"command": "cat state/log.jsonl | head"}}, 0),
        ("Edit non-jsonl (allowed)", {"tool_name": "Edit", "tool_input": {"file_path": "README.md"}}, 0),
    ]
    for name, inp, expected_rc in cases:
        rc, err = run_hook(inp)
        if rc != expected_rc:
            _fail(f"{name}: expected exit {expected_rc}, got {rc}  stderr={err[:120]}")
        _ok(f"{name} → exit {rc}{'  (blocked)' if rc == 2 else ''}")


# ──────────────────────────────────────────────────────────────────
# PATCH #6: git_sha unknown → NotAGitRepo
# ──────────────────────────────────────────────────────────────────
def test_patch_6_git_sha_unknown():
    _section("PATCH #6: manifest refuses outside git repo (was: silent 'unknown')")
    from engine.core.reproducibility_manifest import generate, NotAGitRepo

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env_lock = tmp / "env.lock"
        env_lock.write_text("dummy")
        try:
            generate(
                exp_id="exp_test",
                repo_root=tmp,
                data_hash="a" * 64,
                env_lock_path=env_lock,
                random_seeds={"python": 0, "numpy": 0, "torch": 0},
            )
            _fail("should have raised NotAGitRepo")
        except NotAGitRepo as e:
            _ok(f"NotAGitRepo raised: {str(e)[:80]}...")

    # Inside an actual git repo (this REPO) → should succeed
    env_lock = REPO / "tests_env.lock"
    env_lock.write_text("dummy")
    try:
        mf = generate(
            exp_id="exp_test_2",
            repo_root=REPO,
            data_hash="b" * 64,
            env_lock_path=env_lock,
            random_seeds={"python": 0, "numpy": 0, "torch": 0},
        )
        if not mf.git_sha or len(mf.git_sha) < 7:
            _fail(f"git_sha unexpected: {mf.git_sha}")
        _ok(f"manifest inside git repo: git_sha={mf.git_sha[:12]}")
    finally:
        env_lock.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_patch_1_append_only_race()
    test_patch_2_evaluation_precondition()
    test_patch_3_auto_safe_approval()
    test_patch_4_gate_c_naming()
    test_patch_5_hook_bash_bypass()
    test_patch_6_git_sha_unknown()

    print(f"\n{'='*60}\n  ✅ ALL 6 PATCHES VERIFIED\n{'='*60}")
