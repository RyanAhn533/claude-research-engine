#!/usr/bin/env python3
"""
Tier 2 smoke test. Runs after install + bootstrap. Exits 0 only if every
Tier 2 module behaves correctly on its happy path AND blocks its key failure path.
"""
import json
import sys
import tempfile
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def section(title):
    print(f"\n== {title} ==")


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    project = "99_tier2_smoke"

    # Pre-cleanup if prior failed run left state
    import shutil
    pdir = repo_root / "projects" / project
    if pdir.exists():
        shutil.rmtree(pdir)

    # ── Step 1: bootstrap a throwaway project ──
    section("step 1: bootstrap project")
    import subprocess
    res = subprocess.run(
        [sys.executable, "-m", "engine.cli.jy", "bootstrap",
         "--project", project, "--prefix", "TSMK", "--gpu-hours", "10"],
        cwd=repo_root, capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        sys.exit("bootstrap failed")
    print("  ✓ bootstrap ok")

    # ── Step 2: state_machine — illegal transition must refuse ──
    section("step 2: state_machine FSM enforcement")
    from engine.core.state_machine import StateMachine, TransitionRefused

    sm = StateMachine(project=project, repo_root=repo_root)
    assert sm.current_phase() == "BOOTSTRAP", sm.current_phase()
    print(f"  ✓ initial phase: {sm.current_phase()}")

    # legal: BOOTSTRAP → STATUS_SYNC
    sm.transition("STATUS_SYNC", reason="smoke")
    assert sm.current_phase() == "STATUS_SYNC"
    print(f"  ✓ legal transition BOOTSTRAP → STATUS_SYNC")

    # illegal: STATUS_SYNC → LOGGING (skipping everything)
    try:
        sm.transition("LOGGING", reason="should fail")
        sys.exit("FAIL: illegal transition was allowed")
    except TransitionRefused as e:
        print(f"  ✓ illegal transition blocked: {e}")

    # legal: STATUS_SYNC → METHOD_SEARCH
    sm.transition("METHOD_SEARCH", reason="smoke")
    print(f"  ✓ legal: STATUS_SYNC → METHOD_SEARCH (iter→{sm.last_state()['iter']})")

    # ── Step 3: compute_budget ──
    section("step 3: compute_budget")
    from engine.core.compute_budget import check, record_usage, per_experiment_check

    bs = check(used=5, total=10)
    assert bs.ok and "ok" in bs.reason
    print(f"  ✓ budget under cap: {bs.reason}")

    bs = check(used=11, total=10)
    assert not bs.ok and "exhausted" in bs.reason
    print(f"  ✓ budget exhausted detected: {bs.reason}")

    new_b = record_usage({"gpu_hours_used": 5}, gpu_hours=2.5)
    assert new_b["gpu_hours_used"] == 7.5
    print(f"  ✓ record_usage immutable: 5 → {new_b['gpu_hours_used']}")

    # ── Step 4: reproducibility_manifest ──
    section("step 4: reproducibility_manifest")
    from engine.core.reproducibility_manifest import generate

    with tempfile.TemporaryDirectory() as td:
        envlock = Path(td) / "env.lock"
        envlock.write_text("numpy==1.26.0\n")
        mf = generate(
            exp_id="exp_001_smoke",
            random_seeds={"python": 42, "numpy": 42, "torch": 42},
            env_lock_path=envlock,
            data_hash="sha256:" + "f" * 64,
            repo_root=repo_root,
        )
        d = mf.to_dict()
        assert d["exp_id"] == "exp_001_smoke"
        assert d["random_state"]["python"] == 42
        assert d["data_hash"].startswith("sha256:")
        # validate against schema
        import jsonschema
        schema = json.loads((repo_root / "engine/schemas/reproducibility_manifest.schema.json").read_text())
        jsonschema.Draft7Validator(schema).validate(d)
        print(f"  ✓ manifest generated + schema-valid (git_sha={d['git_sha'][:7]}, dirty={d['git_dirty']})")

    # missing data_hash must raise
    try:
        generate(exp_id="x", random_seeds={"python": 1, "numpy": 1, "torch": 1},
                 env_lock_path="nonexistent", data_hash="")
        sys.exit("FAIL: empty data_hash was accepted")
    except ValueError as e:
        print(f"  ✓ empty data_hash refused: {e}")

    # ── Step 5: leakage_auditor ──
    section("step 5: leakage_auditor")
    from engine.core.leakage_auditor import audit

    # clean run
    r = audit(
        exp_id="exp_001_smoke",
        train_indices={1, 2, 3, 4},
        test_indices={5, 6, 7},
        val_indices={8, 9},
        declarations={
            "preprocessing_stats_fit_on_train_only": True,
            "no_future_information_in_features": True,
            "augmentation_only_on_train_split": True,
        },
        label_predictability_upper_bound=0.02,
    )
    assert r.passed, r.fail_summary()
    print(f"  ✓ clean audit passes ({len(r.checks)} checks)")

    # dirty run — overlap + missing declaration
    r = audit(
        exp_id="exp_002_smoke",
        train_indices={1, 2, 3, 4},
        test_indices={3, 4, 5},  # overlap!
        declarations={"preprocessing_stats_fit_on_train_only": True},
        label_predictability_upper_bound=0.30,  # leaky
    )
    assert not r.passed
    fails = [c.name for c in r.checks if not c.passed]
    print(f"  ✓ dirty audit detects: {fails}")

    # ── Step 6: gate_c paired-delta ──
    section("step 6: gate_c paired-delta")
    from engine.gates.gate_c import evaluate

    # prototype: 3 seeds, directional only
    r = evaluate(
        new_results=[70.5, 66.2, 73.3],
        baseline_results=[48.0, 47.5, 48.5],
        stage="prototype",
    )
    assert r.verdict == "improvement_candidate", r
    assert r.cohen_d_paired is None  # not computed at prototype
    print(f"  ✓ prototype: {r.verdict}, mean_delta={r.paired_delta_mean:.2f}")

    # paper_ready needs n>=7
    r = evaluate(
        new_results=[70.5, 66.2, 73.3],
        baseline_results=[48.0, 47.5, 48.5],
        stage="paper_ready",
    )
    assert r.verdict == "failed" and "n_seeds=3" in r.reason
    print(f"  ✓ paper_ready with n=3 refused: {r.reason}")

    # paper_ready with 7 seeds + clear win — requires scipy
    try:
        import scipy  # noqa
        new = [70.5, 66.2, 73.3, 69.8, 70.2, 71.5, 68.9]
        base = [48.0, 47.5, 48.5, 47.0, 48.2, 47.8, 48.3]
        r = evaluate(
            new_results=new, baseline_results=base,
            stage="paper_ready",
            delta_threshold=10.0,
        )
        assert r.verdict == "improvement_candidate", r
        print(f"  ✓ paper_ready n=7: {r.verdict}")
        print(f"    paired_delta={r.paired_delta_mean:.2f}, CI=[{r.paired_delta_ci_95[0]:.2f},{r.paired_delta_ci_95[1]:.2f}], "
              f"d={r.cohen_d_paired:.2f}, p={r.wilcoxon_p:.4f}")
    except ImportError:
        print(f"  ℹ scipy not installed — paper_ready Gate C path skipped")

    # ── Step 7: cleanup ──
    section("step 7: cleanup")
    import shutil
    shutil.rmtree(repo_root / "projects" / project)
    print(f"  ✓ removed {project}")

    print("\n✅ Tier 2 smoke test — ALL PASS")


if __name__ == "__main__":
    main()
