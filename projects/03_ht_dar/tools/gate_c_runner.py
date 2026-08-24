"""Gate C: M1 (Hierarchical) vs M0 (baseline) — 5-seed paired, MOSEI acc_2.

엔진 prototype Gate C 판정 + 리치 통계(bootstrap CI / Cohen's d / Wilcoxon / TOST).
판정을 leaderboard M1 row 갱신(supersedes) + hypothesis outcome(supersedes)로 기록.
"""
import json, re, sys, math, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE"); sys.path.insert(0, str(ROOT))
SD = ROOT / "projects/03_ht_dar/state"; SCH = ROOT / "engine/schemas"
LOGS = Path("/home/ajy/HT-DAR/code/logs_run")
EXP = "exp_001_mosei_stripped_ladder"
FP = json.loads((ROOT / "projects/03_ht_dar/artifacts/CONFIG_FINGERPRINTS.json").read_text())
from engine.core.append_only_logger import AppendOnlyLog
from engine.gates import gate_c

VAL_RE = re.compile(r"VAL-\(.*?\).*?acc_2:\s*([\d.]+).*?Loss:\s*([\d.]+)")
TEST_RE = re.compile(r"TEST-\(.*?\).*?acc_2:\s*([\d.]+)")
SEEDS = [1111, 1112, 1113, 1114, 1115]
DELTA = 0.005  # pre-registered threshold (0.5pp)


def best_val_test_acc2(variant, seed):
    val, test = [], []
    for line in (LOGS / f"{variant}_s{seed}.log").read_text(errors="ignore").splitlines():
        m = VAL_RE.search(line)
        if m: val.append(float(m.group(2)))
        m = TEST_RE.search(line)
        if m: test.append(float(m.group(1)))
    n = min(len(val), len(test))
    bi = min(range(n), key=lambda i: val[i])
    return test[bi]


def main():
    m0 = [best_val_test_acc2("M0", s) for s in SEEDS]
    m1 = [best_val_test_acc2("M1", s) for s in SEEDS]
    deltas = np.array(m1) - np.array(m0)
    print("seeds:", SEEDS)
    print("M0:", [round(x, 4) for x in m0], f"mean={statistics.mean(m0):.4f}")
    print("M1:", [round(x, 4) for x in m1], f"mean={statistics.mean(m1):.4f}")
    print("Δ (M1-M0):", [round(x, 4) for x in deltas], f"mean={deltas.mean():+.4f}")

    # ── 엔진 Gate C (prototype, n=5) ──
    res = gate_c.evaluate(new_results=m1, baseline_results=m0, stage="prototype",
                          delta_threshold=DELTA, higher_is_better=True)
    print(f"\n[engine Gate C / prototype] verdict={res.verdict} :: {res.reason}")

    # ── 리치 통계 (수동, n<7이라 paper_ready 미달이지만 기록용) ──
    rng = np.random.default_rng(0)
    boot = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    sd = float(deltas.std(ddof=1)); d = float(deltas.mean() / sd) if sd > 0 else 0.0
    try:
        w_p = float(sps.wilcoxon(m1, m0).pvalue)
    except Exception:
        w_p = float("nan")
    t_p = float(sps.ttest_rel(m1, m0).pvalue)
    # TOST equivalence (±DELTA): 두 one-sided t-test
    n = len(deltas); se = sd / math.sqrt(n); mean = float(deltas.mean())
    t_lo = (mean - (-DELTA)) / se; t_hi = (mean - DELTA) / se
    p_lo = 1 - sps.t.cdf(t_lo, n - 1); p_hi = sps.t.cdf(t_hi, n - 1)
    tost_p = max(p_lo, p_hi); tost_equiv = tost_p < 0.05
    print(f"  bootstrap95 CI={ci}, Cohen_d={d:.3f}, Wilcoxon_p={w_p:.4f}, paired_t_p={t_p:.4f}")
    print(f"  TOST(±{DELTA}) p={tost_p:.4f} → equivalent={tost_equiv}")

    # ── outcome 결정 ──
    if ci[1] < 0:
        outcome = "rejected"   # M1 유의하게 나쁨
    elif res.verdict == "improvement_candidate" and ci[0] > 0:
        outcome = "supported"
    else:
        outcome = "neutral"    # noise 범위 (CI가 0 포함)
    print(f"\n[OUTCOME] H_htdar_m1 → {outcome}")

    now = datetime.now(timezone.utc).isoformat()
    # 1) leaderboard M1 row supersede (gate_c 통계 부착)
    prior_m1 = "leaderboard_20260603T001426Z_501049"
    lb = AppendOnlyLog(SD / "leaderboard.jsonl", SCH / "experiment.schema.json")
    lb.append(dict(
        exp_id=EXP, iteration=5, method="DLF_HTCMA_Stripped + Hierarchical pyramid L=3 (Gate C vs M0)",
        method_id="strip_m1", config_fingerprint=FP["M1"], metric_name="mosei_acc2",
        constraints_passed=True, baseline_candidate=False, timestamp=now,
        notes=(f"Gate C M1 vs M0 5-seed: Δ={mean:+.4f} CI95=[{ci[0]:.4f},{ci[1]:.4f}] "
               f"d={d:.2f} wilcoxon_p={w_p:.3f} t_p={t_p:.3f} TOST±0.5pp p={tost_p:.3f} "
               f"equiv={tost_equiv} → {outcome}. M1 단일seed 0.8608은 운; 5seed 평균 M0와 동급(약간 낮음)."),
        results=dict(mosei_acc2=float(statistics.mean(m1)),
                     per_seed_m1=[float(x) for x in m1], per_seed_m0=[float(x) for x in m0],
                     deltas=[float(x) for x in deltas], tost_p=float(tost_p),
                     tost_equivalent=bool(tost_equiv), paired_t_p=float(t_p)),
        stage="prototype", n_seeds=5,
        paired_delta_mean=float(mean), paired_delta_ci_95=[float(ci[0]), float(ci[1])],
        cohen_d_paired=float(d), wilcoxon_p=float(w_p) if not math.isnan(w_p) else None,
        practical_effect_passed=bool(ci[0] > 0 and abs(mean) >= DELTA),
        gate_c_verdict=res.verdict, gate_b_verdict="pass",
        linked_claims=[], manifest_ref=f"reproducibility_manifests/{EXP}.pip_freeze.txt",
        self_attack_run=False, supersedes=prior_m1))
    print("  -> leaderboard M1 row superseded with Gate C stats")

    # 2) hypothesis outcome supersede
    prior_h = "hypothesis_registry_20260602T080304Z_76952a"
    h = AppendOnlyLog(SD / "hypothesis_registry.jsonl", SCH / "hypothesis.schema.json")
    h.append(dict(
        hypothesis_id="H_htdar_m1_hier_beats_m0", exp_id=EXP,
        primary="M1 (Hierarchical pyramid L=3) > M0 (meanpool) on MOSEI Acc2 by >=0.5pp, reproducing HT-DAR HCB-H013.",
        null_hypothesis="M1 - M0 <= 0 (hierarchical adds nothing over meanpool)",
        success_criterion=dict(metric="mosei_acc2", direction="higher_is_better", delta_threshold=DELTA),
        failure_implication="Hierarchical pyramid is not the gain source; revert to noise framing.",
        falsifiability_check="PASS", registered_at=now,
        outcome=dict(result=outcome, observed_delta=float(mean),
                     practical_effect_passed=bool(ci[0] > 0 and abs(mean) >= DELTA),
                     scored_at=now),
        supersedes=prior_h))
    print("  -> hypothesis H_htdar_m1 outcome recorded")


if __name__ == "__main__":
    main()
