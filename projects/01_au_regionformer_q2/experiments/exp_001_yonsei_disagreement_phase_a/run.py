"""exp_001 — Yonsei Disagreement Phase A.

Statistical analysis of self-observer affective congruence on AI Hub Korean FER
with the Yonsei 298-person consensus annotation.

Output: results.json with all numbers needed for the leaderboard row + paper RQ1.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
CSV_DIR = Path("/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs")
SEED = 42
BOOTSTRAP_B = 1000


def load() -> pd.DataFrame:
    tr = pd.read_csv(CSV_DIR / "master_train_v_yonsei.csv")
    va = pd.read_csv(CSV_DIR / "master_val_v_yonsei.csv")
    df = pd.concat([tr.assign(split="train"), va.assign(split="val")], ignore_index=True)
    return df[df.yon_evaluated == 1].copy()


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(str(df.yon_majority_reject.sum()).encode())
    h.update(str(sorted(df.label.unique())).encode())
    h.update(str(df.subject_hash.nunique()).encode())
    return f"sha256:{h.hexdigest()}"


def marginal(ev: pd.DataFrame) -> dict:
    g = ev.groupby("label").yon_majority_reject.agg(["count", "mean"])
    g.columns = ["n", "reject_rate"]
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(ev.label, ev.yon_majority_reject))
    return {
        "per_emotion": {k: {"n": int(v["n"]), "reject_rate": float(v["reject_rate"])}
                        for k, v in g.iterrows()},
        "chi2": {"stat": float(chi2), "dof": int(dof), "p": float(p)},
    }


def reliability(ev: pd.DataFrame) -> dict:
    multi = ev[ev.yon_n_evals >= 2]
    unanim = ((multi.yon_reject_rate == 0) | (multi.yon_reject_rate == 1)).mean()
    g = multi.groupby("label").apply(
        lambda x: pd.Series({"n": len(x),
                             "unanimous_pct": float(((x.yon_reject_rate == 0) | (x.yon_reject_rate == 1)).mean() * 100),
                             "mean_rate": float(x.yon_reject_rate.mean())})
    )
    return {
        "n_multi_rater": int(len(multi)),
        "unanimous_overall_pct": float(unanim * 100),
        "per_emotion": {k: {"n": int(v["n"]),
                             "unanimous_pct": float(v["unanimous_pct"]),
                             "mean_rate": float(v["mean_rate"])}
                        for k, v in g.iterrows()},
    }


def glm_cluster(ev: pd.DataFrame, n_per: int = 12_000) -> dict:
    samp = ev.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(n_per, len(x)), random_state=SEED)
    )
    samp["emo_cat"] = pd.Categorical(samp.label, categories=["neutral", "happy", "angry", "sad"])
    model = smf.glm(
        "yon_majority_reject ~ C(emo_cat, Treatment(reference='neutral'))",
        data=samp, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": samp["subject_hash"]})
    out = {"n_sample": int(len(samp)), "terms": {}}
    conf = model.conf_int()
    for term, beta in model.params.items():
        if term == "Intercept":
            out["intercept_logodds"] = float(beta)
            continue
        emo = term.split("[T.")[-1].rstrip("]")
        lo, hi = conf.loc[term]
        out["terms"][emo] = {
            "beta": float(beta), "se": float(model.bse[term]),
            "z": float(model.tvalues[term]), "p": float(model.pvalues[term]),
            "OR": float(np.exp(beta)), "OR_ci_lo": float(np.exp(lo)), "OR_ci_hi": float(np.exp(hi)),
        }
    return out


def within_subject(ev: pd.DataFrame, rng: np.random.Generator) -> dict:
    piv = ev.groupby(["subject_hash", "label"]).yon_majority_reject.mean().unstack().dropna()
    piv["neg_mean"] = (piv["angry"] + piv["sad"]) / 2
    piv["pos_mean"] = (piv["happy"] + piv["neutral"]) / 2
    piv["delta"] = piv["neg_mean"] - piv["pos_mean"]
    w = stats.wilcoxon(piv["neg_mean"], piv["pos_mean"])

    delta = piv["delta"].values
    boots = np.empty(BOOTSTRAP_B)
    n = len(delta)
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        boots[b] = delta[idx].mean()
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    return {
        "n_subjects_with_all_4_emotions": int(len(piv)),
        "mean_delta_pp": float(delta.mean() * 100),
        "median_delta_pp": float(np.median(delta) * 100),
        "bootstrap_ci_95_pp": [float(ci_lo * 100), float(ci_hi * 100)],
        "pct_subjects_delta_positive": float((delta > 0).mean() * 100),
        "wilcoxon": {"stat": float(w.statistic), "p": float(w.pvalue)},
    }


def gate_c_verdict(glm_out: dict, ws: dict) -> dict:
    or_angry = glm_out["terms"]["angry"]["OR"]
    or_sad = glm_out["terms"]["sad"]["OR"]
    or_angry_lo = glm_out["terms"]["angry"]["OR_ci_lo"]
    delta_ci_lo = ws["bootstrap_ci_95_pp"][0]
    wilcoxon_p = ws["wilcoxon"]["p"]

    practical_effect_passed = bool(or_angry > 1.5 and or_angry_lo > 1.0 and delta_ci_lo > 0)
    direction_supported = practical_effect_passed and wilcoxon_p < 0.05

    return {
        "practical_effect_passed": practical_effect_passed,
        "tost_equivalent": None,
        "direction_supported": direction_supported,
        "verdict": "improvement_candidate" if direction_supported else "failed",
        "primary_metric": "or_angry_vs_neutral",
        "primary_value": float(or_angry),
        "delta_threshold": 0.5,
    }


def main() -> Path:
    rng = np.random.default_rng(SEED)
    ev = load()
    results = {
        "exp_id": "exp_001_yonsei_disagreement_phase_a",
        "n_total_evaluated": int(len(ev)),
        "data_hash": data_hash(ev),
        "seed": SEED,
        "bootstrap_B": BOOTSTRAP_B,
    }
    results["marginal"] = marginal(ev)
    results["reliability_bound"] = reliability(ev)
    results["glm_cluster_robust"] = glm_cluster(ev)
    results["within_subject_paired"] = within_subject(ev, rng)
    results["gate_c"] = gate_c_verdict(results["glm_cluster_robust"], results["within_subject_paired"])

    out = ROOT / "results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=False))
    print(f"results → {out}")
    print(f"verdict: {results['gate_c']['verdict']}  "
          f"OR_angry={results['gate_c']['primary_value']:.2f}  "
          f"Wilcoxon p={results['within_subject_paired']['wilcoxon']['p']:.2e}")
    return out


if __name__ == "__main__":
    main()
