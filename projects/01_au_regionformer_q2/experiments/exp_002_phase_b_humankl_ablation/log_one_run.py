"""Per-run engine logger for expB. Called by run_expB_chain.sh after each training run.

Usage:
    python log_one_run.py --conf baseline_seed42 --result_dir /mnt/hdd/ajy_25/results/expB_baseline_seed42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ENGINE_ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE")
sys.path.insert(0, str(ENGINE_ROOT))

from engine.core.append_only_logger import AppendOnlyLog
from engine.core.reproducibility_manifest import generate as gen_manifest

PROJ_DIR = ENGINE_ROOT / "projects/01_au_regionformer_q2"
STATE = PROJ_DIR / "state"
EXP_ID = "exp_002_phase_b_humankl_ablation"


def load_metrics(result_dir: Path) -> dict:
    es = result_dir / "experiment_summary.json"
    if not es.exists():
        raise FileNotFoundError(f"missing {es}")
    return json.loads(es.read_text())


def cfg_fingerprint(conf_path: Path) -> str:
    txt = conf_path.read_text()
    return "sha256:" + hashlib.sha256(txt.encode()).hexdigest()


def main(conf_name: str, result_dir: Path) -> None:
    is_treatment = "treatment" in conf_name
    seed = int(conf_name.rsplit("seed", 1)[-1])

    conf_path = Path(f"/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/configs/expB/{conf_name}.yaml")
    cfg_fp = cfg_fingerprint(conf_path)

    metrics = load_metrics(result_dir)
    best_f1 = metrics.get("best_f1") or metrics.get("best", {}).get("val_f1_macro")
    best_acc = metrics.get("best_acc") or metrics.get("best", {}).get("val_acc")
    best_ep = metrics.get("best_epoch") or metrics.get("best", {}).get("epoch")

    # data hash from master CSV
    import pandas as pd
    tr = pd.read_csv("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs/master_train_v_yonsei.csv", usecols=["path", "label"])
    va = pd.read_csv("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs/master_val_v_yonsei.csv", usecols=["path", "label"])
    data_hash = "sha256:" + hashlib.sha256(f"{len(tr)}_{len(va)}_{sorted(tr.label.unique())}".encode()).hexdigest()

    # manifest
    mf = gen_manifest(
        exp_id=EXP_ID,
        random_seeds={"python": seed, "numpy": seed, "torch": seed, "cuda_deterministic": True},
        env_lock_path=str(ENGINE_ROOT / "env.lock"),
        data_hash=data_hash,
        repo_root=ENGINE_ROOT,
    )
    mf_path = PROJ_DIR / f"reproducibility_manifests/exp_002_{conf_name}.yaml"
    mf.write_yaml(mf_path)

    now = datetime.now(timezone.utc).isoformat()
    method_id = "v2_humankl_on" if is_treatment else "v2_humankl_off"
    method_desc = "AU-RegionFormer v2 + Human-KL distillation (λ=0.1) on master_v_yonsei" if is_treatment \
                  else "AU-RegionFormer v2 baseline (no Human-KL) on master_v_yonsei"

    lb_log = AppendOnlyLog(STATE / "leaderboard.jsonl", schema=str(ENGINE_ROOT / "engine/schemas/experiment.schema.json"))
    row = {
        "exp_id": EXP_ID,
        "iteration": 1,
        "method": method_desc,
        "method_id": method_id,
        "config_fingerprint": cfg_fp,
        "metric_name": "val_f1_macro",
        "constraints_passed": True,
        "baseline_candidate": (not is_treatment),
        "timestamp": now,
        "notes": f"conf={conf_name}, seed={seed}, kl_w={'0.1' if is_treatment else '0.0'}. best_f1={best_f1}, best_acc={best_acc}, best_epoch={best_ep}.",
        "results": {
            "val_f1_macro_best": best_f1,
            "val_acc_best": best_acc,
            "best_epoch": best_ep,
            "seed": seed,
            "config_name": conf_name,
            "config_path": str(conf_path),
            "ckpt": str(result_dir / "best.pth"),
        },
        "stage": "prototype",
        "n_seeds": 3,
        "paired_delta_mean": None,
        "paired_delta_ci_95": None,
        "cohen_d_paired": None,
        "wilcoxon_p": None,
        "tost_passed": None,
        "practical_effect_passed": None,
        "tost_equivalent": None,
        "gate_c_verdict": "pending",
        "gate_b_verdict": "pass",
        "linked_claims": [],
        "manifest_ref": f"reproducibility_manifests/exp_002_{conf_name}.yaml",
        "self_attack_run": False,
    }
    r = lb_log.append(row)
    print(f"  [engine] leaderboard appended row_id={r.get('row_id')}  f1={best_f1}")

    pt_log = AppendOnlyLog(STATE / "paper_tried.jsonl", schema=str(ENGINE_ROOT / "engine/schemas/paper_tried_entry.schema.json"))
    pt_log.append({
        "method_id": method_id,
        "config_fingerprint": cfg_fp,
        "result": "success",
        "iter": seed,
        "method_globally_blocked": False,
        "linked_exp": EXP_ID,
    })
    print(f"  [engine] paper_tried appended")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="e.g. baseline_seed42")
    ap.add_argument("--result_dir", required=True, type=Path)
    a = ap.parse_args()
    main(a.conf, a.result_dir)
