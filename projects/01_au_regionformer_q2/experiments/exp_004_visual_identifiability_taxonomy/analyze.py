"""exp_004 visual identifiability taxonomy (V1-V4). No GPU.
Reuses exp_003's predictions ⟕ yon_reject_rate (basename join), 3 baseline seeds.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("/mnt/hdd/ajy_25/results")
VAL_CSV = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs/master_val_v_yonsei.csv")
CLASSES = ["angry", "happy", "neutral", "sad"]
BASE = {42: "expB_baseline_seed42", 123: "expB_baseline_seed123", 777: "expB_baseline_seed777"}

va = pd.read_csv(VAL_CSV)[["path", "yon_reject_rate", "yon_n_evals"]]
va["bn"] = va["path"].map(os.path.basename)


def taxo_one(result_dir: Path) -> dict | None:
    pp = result_dir / "paper_artifacts/predictions.csv"
    if not pp.exists():
        return None
    pr = pd.read_csv(pp)[["path", "true_label", "pred_label", "correct"]]
    pr["bn"] = pr["path"].map(os.path.basename)
    m = pr.merge(va.drop(columns=["path"]), on="bn", how="left")
    m = m[m.yon_n_evals > 0].copy()                      # evaluated only
    m["observer_agree"] = m.yon_reject_rate <= 1e-9       # agree=0 reject_rate
    m["observer_reject"] = m.yon_reject_rate >= 0.5
    def grp(r):
        if r.observer_agree:  return "V1" if r.correct else "V2"
        if r.observer_reject: return "V3" if r.correct else "V4"
        return "mid"
    m["V"] = m.apply(grp, axis=1)
    out = {}
    for scope, sub in [("all", m)] + [(c, m[m.true_label == c]) for c in CLASSES]:
        n = len(sub)
        cnt = {k: int((sub.V == k).sum()) for k in ["V1", "V2", "V3", "V4", "mid"]}
        v2, v4, v1, v3 = cnt["V2"], cnt["V4"], cnt["V1"], cnt["V3"]
        out[scope] = {
            "N": n, **cnt,
            "P_Vk": {k: (cnt[k] / n if n else None) for k in ["V1", "V2", "V3", "V4"]},
            "error_share_reject": (v4 / (v2 + v4) if (v2 + v4) else None),
            "model_correct_given_reject": (v3 / (v3 + v4) if (v3 + v4) else None),
            "model_wrong_given_agree": (v2 / (v1 + v2) if (v1 + v2) else None),
        }
    return out


per = {s: taxo_one(RESULTS / d) for s, d in BASE.items()}
per = {s: v for s, v in per.items() if v}

def mean_over_seeds(scope, key):
    vals = [per[s][scope][key] for s in per if per[s][scope][key] is not None]
    return float(np.mean(vals)) if vals else None

# aggregate (mean of per-seed metrics)
agg = {}
for scope in ["all"] + CLASSES:
    agg[scope] = {
        "mean_counts": {k: float(np.mean([per[s][scope][k] for s in per])) for k in ["V1","V2","V3","V4"]},
        "model_correct_given_reject": mean_over_seeds(scope, "model_correct_given_reject"),
        "error_share_reject": mean_over_seeds(scope, "error_share_reject"),
        "model_wrong_given_agree": mean_over_seeds(scope, "model_wrong_given_agree"),
    }

mcgr_all = agg["all"]["model_correct_given_reject"]
THRESH = 0.50
supported = bool(mcgr_all is not None and mcgr_all >= THRESH)

results = {
    "exp_id": "exp_004_visual_identifiability_taxonomy",
    "hypothesis_id": "H_taxonomy_v3_substantial",
    "seeds": list(per.keys()),
    "per_seed": per,
    "aggregate": agg,
    "model_correct_given_reject_all": mcgr_all,
    "threshold": THRESH,
    "supported": supported,
    "verdict": "supported" if supported else "rejected",
}
out = Path(__file__).parent / "results.json"
out.write_text(json.dumps(results, indent=2))

print(f"seeds used: {list(per.keys())}")
print("\n=== overall taxonomy (mean counts over seeds) ===")
c = agg["all"]["mean_counts"]
print(f"  V1(agree,correct)={c['V1']:.0f}  V2(agree,wrong)={c['V2']:.0f}  "
      f"V3(reject,correct)={c['V3']:.0f}  V4(reject,wrong)={c['V4']:.0f}")
print(f"  model_correct_given_reject = V3/(V3+V4) = {mcgr_all:.4f}  (threshold {THRESH})")
print(f"  error_share_reject = V4/(V2+V4) = {agg['all']['error_share_reject']:.4f}")
print(f"  model_wrong_given_agree = V2/(V1+V2) = {agg['all']['model_wrong_given_agree']:.4f}")
print("\n=== per class: model_correct_given_reject (V3/(V3+V4)) ===")
for cl in CLASSES:
    v = agg[cl]["model_correct_given_reject"]
    print(f"  {cl:8} {v:.4f}" if v is not None else f"  {cl:8} NA")
print(f"\nVERDICT = {results['verdict']}  (H: V3/(V3+V4) >= 0.50)")
print(f"results -> {out}")
