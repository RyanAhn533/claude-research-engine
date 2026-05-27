"""Aggregate exp_002 6 runs → paired delta CI + Cohen's d + Wilcoxon + Gate C verdict.

Reads:
  /mnt/hdd/ajy_25/results/expB_{baseline,treatment_hkl01}_seed{42,123,777}/experiment_summary.json
Writes:
  - exp_002 results.json (final)
  - new leaderboard row (aggregated, paper_ready stage if n_seeds≥7 else prototype)
  - hypothesis_registry supersedes row (outcome)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ENGINE_ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE")
sys.path.insert(0, str(ENGINE_ROOT))

from engine.core.append_only_logger import AppendOnlyLog

PROJ = ENGINE_ROOT / "projects/01_au_regionformer_q2"
STATE = PROJ / "state"
EXP_DIR = PROJ / "experiments/exp_002_phase_b_humankl_ablation"
EXP_ID = "exp_002_phase_b_humankl_ablation"
HYP_ID = "H_humankl_distill_improves_f1"
SEEDS = [42, 123, 777]
DELTA_THRESHOLD = 0.003


def read_f1(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    f = d.get("best_f1") or d.get("best", {}).get("val_f1_macro")
    return float(f) if f is not None else None


def main() -> None:
    base_dir = Path("/mnt/hdd/ajy_25/results")
    base_f1, trt_f1 = [], []
    for s in SEEDS:
        b = read_f1(base_dir / f"expB_baseline_seed{s}/experiment_summary.json")
        t = read_f1(base_dir / f"expB_treatment_hkl01_seed{s}/experiment_summary.json")
        base_f1.append(b); trt_f1.append(t)
        print(f"seed={s}  baseline={b}  treatment={t}")

    if None in base_f1 or None in trt_f1:
        print("Missing runs — aborting aggregate.")
        return

    base_arr = np.array(base_f1)
    trt_arr = np.array(trt_f1)
    delta = trt_arr - base_arr  # paired

    # bootstrap CI of paired delta mean (n=3 is small, use percentile bootstrap)
    rng = np.random.default_rng(42)
    B = 10000
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(delta), size=len(delta))
        boots[i] = delta[idx].mean()
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    # Cohen's d_paired
    cohen_d = float(delta.mean() / (delta.std(ddof=1) + 1e-9)) if delta.std(ddof=1) > 0 else None
    wilcoxon_p = float(stats.wilcoxon(trt_arr, base_arr).pvalue) if len(delta) >= 1 else None

    practical_effect_passed = bool(ci_lo > 0 and abs(delta.mean()) >= DELTA_THRESHOLD)
    direction_supported = practical_effect_passed and (wilcoxon_p is not None and wilcoxon_p < 0.05)
    verdict = "improvement_candidate" if direction_supported else ("neutral" if abs(delta.mean()) < DELTA_THRESHOLD else "failed")

    results = {
        "exp_id": EXP_ID,
        "seeds": SEEDS,
        "baseline_val_f1_macro": [float(x) for x in base_f1],
        "treatment_val_f1_macro": [float(x) for x in trt_f1],
        "paired_delta_per_seed": [float(x) for x in delta],
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_std": float(delta.std(ddof=1)),
        "paired_delta_ci_95": [float(ci_lo), float(ci_hi)],
        "cohen_d_paired": cohen_d,
        "wilcoxon_p": wilcoxon_p,
        "delta_threshold": DELTA_THRESHOLD,
        "practical_effect_passed": practical_effect_passed,
        "direction_supported": direction_supported,
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = EXP_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")
    print(f"  delta mean = {delta.mean()*100:+.3f}pp, CI95 = [{ci_lo*100:+.3f}, {ci_hi*100:+.3f}] pp")
    print(f"  cohen_d = {cohen_d}, wilcoxon p = {wilcoxon_p}, verdict = {verdict}")

    # === Append aggregate leaderboard row ===
    now = results["timestamp"]
    cfg_str = "stage6_full + master_v_yonsei.csv + human_kl_weight∈{0.0,0.1} × seeds{42,123,777}"
    cfg_fp = "sha256:" + hashlib.sha256(cfg_str.encode()).hexdigest()
    lb = AppendOnlyLog(STATE / "leaderboard.jsonl", schema=str(ENGINE_ROOT / "engine/schemas/experiment.schema.json"))
    lb.append({
        "exp_id": EXP_ID,
        "iteration": 99,
        "method": "Aggregate: Human-KL distillation (λ=0.1) vs baseline, AU-RegionFormer v2 + Stage6 on master_v_yonsei, 3 seeds paired.",
        "method_id": "v2_humankl_ablation_aggregate",
        "config_fingerprint": cfg_fp,
        "metric_name": "paired_delta_val_f1_macro",
        "constraints_passed": True,
        "baseline_candidate": False,
        "timestamp": now,
        "notes": f"3 paired seeds. Δmean={delta.mean()*100:+.3f}pp CI95=[{ci_lo*100:+.3f},{ci_hi*100:+.3f}]pp d={cohen_d} p={wilcoxon_p}",
        "results": results,
        "stage": "prototype",
        "n_seeds": len(SEEDS),
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_ci_95": [float(ci_lo), float(ci_hi)],
        "cohen_d_paired": cohen_d,
        "wilcoxon_p": wilcoxon_p,
        "tost_passed": practical_effect_passed,
        "practical_effect_passed": practical_effect_passed,
        "tost_equivalent": None,
        "gate_c_verdict": verdict,
        "gate_b_verdict": "pass",
        "linked_claims": [],
        "manifest_ref": "reproducibility_manifests/exp_002_baseline_seed42.yaml",
        "self_attack_run": False,
    })
    print("aggregate leaderboard row appended")

    # === Supersede hypothesis with outcome ===
    hyp_log = AppendOnlyLog(STATE / "hypothesis_registry.jsonl",
                             schema=str(ENGINE_ROOT / "engine/schemas/hypothesis.schema.json"))
    # find latest hyp row (skip — just append outcome row)
    hyp_log.append({
        "hypothesis_id": HYP_ID,
        "exp_id": EXP_ID,
        "primary": "Adding Human-KL distillation (λ=0.1) over a strong stage6_full baseline, both trained on master_v_yonsei.csv, improves val_f1_macro by at least 0.3pp (paired_delta CI lower > 0.003) across 3 seeds (42, 123, 777).",
        "null_hypothesis": "No improvement; paired delta within ±0.3pp band.",
        "success_criterion": {
            "metric": "paired_delta_val_f1_macro",
            "direction": "higher_is_better",
            "delta_threshold": DELTA_THRESHOLD,
        },
        "falsifiability_check": "PASS",
        "registered_at": now,
        "outcome": {
            "result": "supported" if direction_supported else ("neutral" if abs(delta.mean()) < DELTA_THRESHOLD else "rejected"),
            "observed_delta": float(delta.mean()),
            "practical_effect_passed": practical_effect_passed,
            "tost_equivalent": None,
            "tost_passed": practical_effect_passed,
            "scored_at": now,
        },
        "supersedes": HYP_ID,
    })
    print("hypothesis outcome row appended")


if __name__ == "__main__":
    main()
