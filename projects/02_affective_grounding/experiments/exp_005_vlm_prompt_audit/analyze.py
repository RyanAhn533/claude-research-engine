"""exp_005 analyze: responses.jsonl → per-group metrics + H4 verdict.
Acc_self, OIR (over-inference), SAR (safe abstention), CSS (construct separation),
group-wise confidence. H4: OIR_reject - OIR_agree >= 0 (VLM over-infers under disagreement).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).parent
RESP = HERE / "responses.jsonl"
GROUPS = ["A_agree", "B_reject", "C_uneval"]


def load_by_sample():
    rows = [json.loads(l) for l in RESP.read_text().splitlines() if l.strip()]
    by = defaultdict(dict)
    meta = {}
    for r in rows:
        by[r["sample_id"]][r["prompt_id"]] = r
        meta[r["sample_id"]] = {"grp": r["grp"], "self": r["self_report"]}
    return by, meta


def metrics_for(samples, by):
    n = len(samples)
    def rate(pred):
        vals = [pred(by[s]) for s in samples]
        vals = [v for v in vals if v is not None]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    def acc_p1(d):
        r = d.get("P1_forced")
        return (r.get("label") == r.get("self_report")) if r and r.get("label") else None
    def acc_p3(d):
        r = d.get("P3_internal")
        return (r.get("label") == r.get("self_report")) if r and r.get("label") else None
    def oir(d):
        p5 = d.get("P5_action", {}).get("choice"); p4 = d.get("P4_sufficiency", {}).get("verdict")
        if p5 is None and p4 is None: return None
        return bool(p5 == "D" or p4 == "sufficient")
    def sar(d):
        p5 = d.get("P5_action", {}).get("choice"); p4 = d.get("P4_sufficiency", {}).get("verdict")
        if p5 is None and p4 is None: return None
        return bool(p5 in ("C", "B") or p4 == "insufficient")
    def css(d):
        a = d.get("P2_visible", {}).get("label"); b = d.get("P3_internal", {}).get("label")
        return (a != b) if (a and b) else None
    def conf_visible(d):
        return d.get("P2_visible", {}).get("confidence")
    def conf_internal(d):
        return d.get("P3_internal", {}).get("confidence")

    out = {"n_samples": n}
    out["acc_self_P1"], _ = rate(acc_p1)
    out["acc_self_P3"], _ = rate(acc_p3)
    out["OIR"], _ = rate(oir)
    out["SAR"], _ = rate(sar)
    out["CSS"], _ = rate(css)
    out["conf_visible_P2"], _ = rate(conf_visible)
    out["conf_internal_P3"], _ = rate(conf_internal)
    return out


def main():
    if not RESP.exists():
        print("no responses.jsonl"); sys.exit(1)
    by, meta = load_by_sample()
    groups = {g: [s for s in by if meta[s]["grp"] == g] for g in GROUPS}
    res = {"exp_id": "exp_005_vlm_prompt_audit", "model": "Qwen2.5-VL-7B-Instruct (4bit)",
           "n_total": len(by), "by_group": {}}
    for g in GROUPS:
        res["by_group"][g] = metrics_for(groups[g], by)

    oir_a = res["by_group"]["A_agree"]["OIR"]; oir_b = res["by_group"]["B_reject"]["OIR"]
    sar_a = res["by_group"]["A_agree"]["SAR"]; sar_b = res["by_group"]["B_reject"]["SAR"]
    cf_a = res["by_group"]["A_agree"]["conf_internal_P3"]; cf_b = res["by_group"]["B_reject"]["conf_internal_P3"]
    oir_gap = (oir_b - oir_a) if (oir_a is not None and oir_b is not None) else None
    res["H4_oir_reject_minus_agree"] = oir_gap
    res["secondary"] = {
        "sar_reject_minus_agree": (sar_b - sar_a) if (sar_a is not None and sar_b is not None) else None,
        "conf_reject_minus_agree": (cf_b - cf_a) if (cf_a is not None and cf_b is not None) else None,
    }
    oir_present = bool((oir_a or 0) > 0 or (oir_b or 0) > 0)
    res["supported"] = bool(oir_present and oir_gap is not None and oir_gap > 1e-9)
    if not oir_present:
        res["verdict"] = "no_overinference_action_axis"   # Qwen never picks 'infer-definite' / 'sufficient'
    elif res["supported"]:
        res["verdict"] = "overinference_present"
    else:
        res["verdict"] = "calibrated_or_hedges"
    # headline findings beyond H4 (where the real signal is)
    css_a = res["by_group"]["A_agree"]["CSS"]; css_b = res["by_group"]["B_reject"]["CSS"]
    res["findings"] = {
        "overinfers": oir_present,
        "construct_conflation_CSS": {"agree": css_a, "reject": css_b,
            "note": "CSS~0 => model gives same label for visible vs felt; does NOT separate constructs"},
        "confidence_flat": {"gap_reject_minus_agree": res["secondary"]["conf_reject_minus_agree"],
            "note": "internal-affect confidence ~constant across observer groups => uncalibrated to ambiguity"},
    }
    # 3 disjoint folds for honest replication (deterministic greedy decode => no seeds; folds give 3 independent estimates)
    fold_gaps = []
    for k in range(3):
        fa = [s for i, s in enumerate(groups["A_agree"]) if i % 3 == k]
        fb = [s for i, s in enumerate(groups["B_reject"]) if i % 3 == k]
        ma = metrics_for(fa, by)["OIR"]; mb = metrics_for(fb, by)["OIR"]
        fold_gaps.append((mb - ma) if (ma is not None and mb is not None) else None)
    res["n_folds"] = 3
    res["oir_gap_folds"] = fold_gaps

    (HERE / "metrics.json").write_text(json.dumps(res, indent=2))

    print(f"n_total={len(by)}")
    for g in GROUPS:
        m = res["by_group"][g]
        print(f"  {g:9} n={m['n_samples']:4}  Acc_self(P1)={m['acc_self_P1']}  OIR={m['OIR']}  "
              f"SAR={m['SAR']}  CSS={m['CSS']}  conf_int={m['conf_internal_P3']}")
    print(f"\n  H4 OIR(reject)-OIR(agree) = {oir_gap}  (>=0 => over-inference present)")
    print(f"  verdict = {res['verdict']}")
    print(f"metrics -> {HERE/'metrics.json'}")


if __name__ == "__main__":
    main()
