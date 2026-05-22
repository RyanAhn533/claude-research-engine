"""
jy — CLI entry point for the engine.

Usage:
    jy bootstrap --project 02_emotion_agent --prefix EMA
    jy status    --project 02_emotion_agent
    jy validate  --project 02_emotion_agent
    jy verify-chain --project 02_emotion_agent --log leaderboard
    jy seed-policy   # one-shot, idempotent

Installed by install_v2.sh as `jy` on PATH (via a thin wrapper).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = REPO_ROOT / "engine" / "schemas"


def _project_dir(project: str) -> Path:
    p = REPO_ROOT / "projects" / project
    if not p.exists():
        sys.exit(f"!! project not found: {p}")
    return p


def _state_dir(project: str) -> Path:
    d = _project_dir(project) / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------ bootstrap ------

LOG_TO_SCHEMA = {
    "engine_state": "state.schema.json",
    "leaderboard": "experiment.schema.json",
    "paper_tried": "paper_tried_entry.schema.json",
    "hypothesis_registry": "hypothesis.schema.json",
    "claim_registry": "claim.schema.json",
    "gate_log": "task.schema.json",  # gate output rides task schema for now
    "task_log": "task.schema.json",
    "permission_policy": "permission.schema.json",
}


def cmd_bootstrap(args):
    from engine.core.permission_policy import seed_default_policies

    project = args.project
    pdir = REPO_ROOT / "projects" / project
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "state").mkdir(exist_ok=True)
    (pdir / "experiments").mkdir(exist_ok=True)
    (pdir / "negative_results").mkdir(exist_ok=True)
    (pdir / "reproducibility_manifests").mkdir(exist_ok=True)
    (pdir / "rules").mkdir(exist_ok=True)

    # direction prefix
    prefix_file = pdir / "rules" / "direction_prefix.txt"
    if not prefix_file.exists():
        prefix_file.write_text(args.prefix + "\n")

    # seed permission policy (project-local copy for portability)
    pol_path = pdir / "state" / "permission_policy.jsonl"
    n = seed_default_policies(pol_path, SCHEMA_DIR)
    print(f"  seeded {n} permission policies → {pol_path.relative_to(REPO_ROOT)}")

    # write initial engine_state row
    from engine.core.append_only_logger import AppendOnlyLog
    state_log = AppendOnlyLog(
        path=pdir / "state" / "engine_state.jsonl",
        schema=SCHEMA_DIR / "state.schema.json",
    )
    if not list(state_log.iter_rows()):
        state_log.append({
            "project": project,
            "current_phase": "BOOTSTRAP",
            "iter": 0,
            "current_exp": None,
            "direction_prefix": args.prefix,
            "kill_switch_present": False,
            "auto_mode": False,
            "budget": {"gpu_hours_used": 0.0, "gpu_hours_total": float(args.gpu_hours or 100.0)},
            "pending_human_gate": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "supersedes": None,
        })
        print(f"  wrote initial engine_state row")

    print(f"  ✅ bootstrap complete for {project}")


# ------ status ------

def cmd_status(args):
    from engine.core.append_only_logger import AppendOnlyLog

    project = args.project
    sd = _state_dir(project)
    state_log = AppendOnlyLog(
        path=sd / "engine_state.jsonl",
        schema=SCHEMA_DIR / "state.schema.json",
    )
    rows = list(state_log.iter_rows())
    if not rows:
        sys.exit(f"!! no engine_state row for {project}. Run `jy bootstrap` first.")
    s = rows[-1]
    print(f"== {project} ==")
    print(f"  phase    : {s['current_phase']}")
    print(f"  iter     : {s['iter']}")
    print(f"  exp      : {s.get('current_exp') or '-'}")
    print(f"  auto     : {s['auto_mode']}")
    print(f"  kill_sw  : {s['kill_switch_present']}")
    print(f"  gate     : {s.get('pending_human_gate') or '-'}")
    print(f"  budget   : {s.get('budget', {}).get('gpu_hours_used', 0)}/{s.get('budget', {}).get('gpu_hours_total', '?')} gpu_h")
    print(f"  updated  : {s['updated_at']}")
    # counts
    for log_name, schema_name in LOG_TO_SCHEMA.items():
        p = sd / f"{log_name}.jsonl"
        if p.exists():
            n = sum(1 for _ in p.open() if _.strip())
            print(f"  {log_name:22s} : {n} rows")


# ------ validate ------

def cmd_validate(args):
    """Re-run schema self-validation + project log chain verification."""
    from engine.core.append_only_logger import AppendOnlyLog

    # 1) schema self-validation
    import subprocess
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "engine" / "tests" / "validate_schemas.py")],
        cwd=REPO_ROOT,
    )
    if res.returncode != 0:
        sys.exit("!! schema self-validation failed")

    # 2) project log chain verification
    if args.project:
        sd = _state_dir(args.project)
        for log_name, schema_name in LOG_TO_SCHEMA.items():
            p = sd / f"{log_name}.jsonl"
            if not p.exists():
                continue
            log = AppendOnlyLog(path=p, schema=SCHEMA_DIR / schema_name)
            n = log.verify_chain()
            print(f"  [ok ] {log_name:22s} chain verified ({n} rows)")

    print("✅ validate passed")


# ------ verify-chain ------

def cmd_verify_chain(args):
    from engine.core.append_only_logger import AppendOnlyLog

    sd = _state_dir(args.project)
    schema_name = LOG_TO_SCHEMA[args.log]
    log = AppendOnlyLog(path=sd / f"{args.log}.jsonl", schema=SCHEMA_DIR / schema_name)
    n = log.verify_chain()
    print(f"✅ {args.log} chain ok ({n} rows)")


# ------ seed-policy ------

def cmd_seed_policy(args):
    from engine.core.permission_policy import seed_default_policies

    sd = _state_dir(args.project)
    n = seed_default_policies(sd / "permission_policy.jsonl", SCHEMA_DIR)
    print(f"seeded {n} policies (0 = already present)")


# ------ main ------

def main():
    p = argparse.ArgumentParser(prog="jy", description="Claude Research Engine v2 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="initialize a project")
    b.add_argument("--project", required=True)
    b.add_argument("--prefix", required=True, help="Direction ID prefix, e.g. EMA")
    b.add_argument("--gpu-hours", type=float, default=100.0)
    b.set_defaults(func=cmd_bootstrap)

    s = sub.add_parser("status", help="show project status")
    s.add_argument("--project", required=True)
    s.set_defaults(func=cmd_status)

    v = sub.add_parser("validate", help="schema + chain validation")
    v.add_argument("--project", default=None)
    v.set_defaults(func=cmd_validate)

    vc = sub.add_parser("verify-chain", help="verify one jsonl's hash chain")
    vc.add_argument("--project", required=True)
    vc.add_argument("--log", required=True, choices=list(LOG_TO_SCHEMA.keys()))
    vc.set_defaults(func=cmd_verify_chain)

    sp = sub.add_parser("seed-policy", help="seed default permission policies")
    sp.add_argument("--project", required=True)
    sp.set_defaults(func=cmd_seed_policy)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
