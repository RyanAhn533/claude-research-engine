"""exp_003 ceiling analysis: join saved predictions with yon_reject_rate, compute
model accuracy in A(agreed)/B(rejected)/C(unevaluated), overall + per-class, across seeds.
No GPU. Reads paper_artifacts/predictions.csv + master_val_v_yonsei.csv.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("/mnt/hdd/ajy_25/results")
VAL_CSV = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs/master_val_v_yonsei.csv")
CLASSES = ["angry", "happy", "neutral", "sad"]
BASE = {42: "expB_baseline_seed42", 123: "expB_baseline_seed123", 777: "expB_baseline_seed777"}
TRT = {42: "expB_treatment_hkl01_seed42", 123: "expB_treatment_hkl01_seed123", 777: "expB_treatment_hkl01_seed777"}

import os
va = pd.read_csv(VAL_CSV)[["path", "yon_reject_rate", "yon_n_evals"]]
# Training used a path-fixed CSV (different prefix, identical basenames). Join on basename.
va["bn"] = va["path"].map(os.path.basename)
assert va["bn"].is_unique, "val basenames not unique — join key unsafe"


def group_of(row):
    if row.yon_n_evals == 0:
        return "C_uneval"
    if row.yon_reject_rate <= 1e-9:
        return "A_agree"
    if row.yon_reject_rate >= 0.5:
        return "B_reject"
    return "mid"


va["grp"] = va.apply(group_of, axis=1)


def analyze_run(result_dir: Path) -> dict | None:
    pred_path = result_dir / "paper_artifacts/predictions.csv"
    if not pred_path.exists():
        return None
    pr = pd.read_csv(pred_path)[["path", "true_label", "pred_label", "correct"]]
    pr["bn"] = pr["path"].map(os.path.basename)
    m = pr.merge(va.drop(columns=["path"]), on="bn", how="left", indicator=True)
    cov = (m["_merge"] == "both").mean()
    out = {"join_coverage": float(cov), "n": int(len(m))}
    for g in ["A_agree", "B_reject", "C_uneval", "mid"]:
        sub = m[m.grp == g]
        if len(sub) == 0:
            out[g] = {"n": 0, "acc": None}
            continue
        d = {"n": int(len(sub)), "acc": float(sub.correct.mean())}
        for c in CLASSES:
            cc = sub[sub.true_label == c]
            d[f"acc_{c}"] = float(cc.correct.mean()) if len(cc) else None
            d[f"n_{c}"] = int(len(cc))
        out[g] = d
    return out


def agg(runs: dict) -> dict:
    per = {s: analyze_run(RESULTS / d) for s, d in runs.items()}
    accs = {g: [per[s][g]["acc"] for s in runs if per[s] and per[s][g]["acc"] is not None]
            for g in ["A_agree", "B_reject", "C_uneval", "mid"]}
    mean = {g: (float(np.mean(v)) if v else None) for g, v in accs.items()}
    rng = {g: ([float(min(v)), float(max(v))] if v else None) for g, v in accs.items()}
    return {"per_seed": per, "mean_acc": mean, "acc_range": rng}


base = agg(BASE)
trt = agg(TRT)

mA, mB, mC = base["mean_acc"]["A_agree"], base["mean_acc"]["B_reject"], base["mean_acc"]["C_uneval"]
gap = (mA - mB) if (mA is not None and mB is not None) else None
DELTA = 0.30
supported = bool(gap is not None and gap >= DELTA and mA >= 0.95)
verdict = "ceiling_confirmed" if supported else ("headroom_exists" if (gap is not None and gap < 0.15) else "ambiguous")

# per-class acc in B for baseline (where the asymmetry signal should live)
b_perclass = {}
for c in CLASSES:
    vals = [base["per_seed"][s]["B_reject"].get(f"acc_{c}") for s in BASE
            if base["per_seed"][s] and base["per_seed"][s]["B_reject"].get(f"acc_{c}") is not None]
    b_perclass[c] = float(np.mean(vals)) if vals else None

results = {
    "exp_id": "exp_003_ceiling_analysis",
    "hypothesis_id": "H_ceiling_annotation_agreement",
    "baseline": base, "treatment": trt,
    "mean_acc_A_agree": mA, "mean_acc_B_reject": mB, "mean_acc_C_uneval": mC,
    "acc_gap_A_minus_B": gap, "delta_threshold": DELTA,
    "baseline_B_reject_per_class_acc": b_perclass,
    "supported": supported, "verdict": verdict,
}
out = Path(__file__).parent / "results.json"
out.write_text(json.dumps(results, indent=2))

print(f"join coverage (base s42): {base['per_seed'][42]['join_coverage']:.4f}")
print(f"\ngroup sizes (base s42): " +
      ", ".join(f"{g}={base['per_seed'][42][g]['n']}" for g in ['A_agree','B_reject','C_uneval','mid']))
print(f"\n=== BASELINE mean accuracy by group (3 seeds) ===")
for g, lab in [("A_agree","A 외부인 동의"),("B_reject","B 외부인 거부"),("C_uneval","C 평가없음")]:
    print(f"  {lab:16} acc={base['mean_acc'][g]}  range={base['acc_range'][g]}")
print(f"\n  acc(A) - acc(B) = {gap*100 if gap else None:.2f} pp   (threshold {DELTA*100:.0f}pp)")
print(f"  B(거부)에서 클래스별 정확도: " + ", ".join(f"{c}={v:.3f}" if v else f"{c}=NA" for c,v in b_perclass.items()))
print(f"\n  VERDICT = {verdict}  (hypothesis supported={supported})")
print(f"\nresults -> {out}")
