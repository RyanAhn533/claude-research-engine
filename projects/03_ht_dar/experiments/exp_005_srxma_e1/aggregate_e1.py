"""exp_005 SR-XMA E1 집계 + 판정 (학습 후 실행, GPU 불필요).

12런(4모드×3seed) 로그 → best-val TEST acc_2 → 모드 평균 → gate_c paired:
  ★ learned vs random  (E1 생사)
    learned vs full    (정확도 유지)
    learned vs fixed   (참고)

부분 로그도 처리(미완 런은 'pending' 표시). 판정만 출력, leaderboard 로깅은 별도.
실행: python projects/03_ht_dar/experiments/exp_005_srxma_e1/aggregate_e1.py
"""
import re, sys, statistics, math
from pathlib import Path
import numpy as np
from scipy import stats as sps

ROOT = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE"); sys.path.insert(0, str(ROOT))
from engine.gates import gate_c

CODE = Path("/home/ajy/HT-DAR/code")
E1DIR = CODE / "logs_srxma_e1"
SMOKE = CODE / "smoke_SR_LEARN_s1111.log"
SEEDS = [1111, 1112, 1113]
MODES = ["SR_FULL", "SR_FIXED", "SR_RAND", "SR_LEARN"]
DELTA = 0.005

VAL_RE = re.compile(r"VAL-\(.*?\).*?acc_2:\s*([\d.]+).*?Loss:\s*([\d.]+)")
TEST_RE = re.compile(r"TEST-\(.*?\).*?acc_2:\s*([\d.]+)")


def logpath(mode, seed):
    if mode == "SR_LEARN" and seed == 1111:
        return SMOKE                      # smoke 런이 곧 learned/s1111
    return E1DIR / f"{mode}_s{seed}.log"


def best_val_test_acc2(mode, seed):
    p = logpath(mode, seed)
    if not p.exists():
        return None
    val, test = [], []
    for line in p.read_text(errors="ignore").splitlines():
        m = VAL_RE.search(line)
        if m: val.append(float(m.group(2)))
        m = TEST_RE.search(line)
        if m: test.append(float(m.group(1)))
    n = min(len(val), len(test))
    if n == 0:
        return None
    bi = min(range(n), key=lambda i: val[i])
    return test[bi] / 100.0 if test[bi] > 1.5 else test[bi]   # log은 % 단위(78.0) → 0.78


def collect(mode):
    return [best_val_test_acc2(mode, s) for s in SEEDS]


def paired(name, new, base):
    pairs = [(a, b) for a, b in zip(new, base) if a is not None and b is not None]
    if len(pairs) < 2:
        print(f"  [{name}] 미완 (완료 seed {len(pairs)}/3) — 대기"); return None
    nw = np.array([a for a, b in pairs]); bs = np.array([b for a, b in pairs])
    d = nw - bs
    res = gate_c.evaluate(new_results=list(nw), baseline_results=list(bs),
                          stage="prototype", delta_threshold=DELTA, higher_is_better=True)
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    coh = float(d.mean() / sd) if sd > 0 else 0.0
    try: t_p = float(sps.ttest_rel(nw, bs).pvalue)
    except Exception: t_p = float("nan")
    print(f"  [{name}] n={len(d)} Δ={d.mean():+.4f} CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"d={coh:+.2f} t_p={t_p:.3f} → gate_c={res.verdict}")
    return dict(delta=float(d.mean()), ci=ci, cohen_d=coh, t_p=t_p, verdict=res.verdict, n=len(d))


def main():
    print("="*64); print("exp_005 SR-XMA E1 — best-val TEST acc_2 (MOSEI)"); print("="*64)
    data = {}
    for mode in MODES:
        vals = collect(mode)
        data[mode] = vals
        shown = [f"{v:.4f}" if v is not None else "·····" for v in vals]
        done = [v for v in vals if v is not None]
        mean = f"mean={statistics.mean(done):.4f}" if done else "(pending)"
        print(f"{mode:9s} {shown}  {mean}")
    print("\n--- paired (gate_c prototype, 0.5pp threshold) ---")
    e1 = paired("LEARNED vs RANDOM ★E1", data["SR_LEARN"], data["SR_RAND"])
    paired("LEARNED vs FULL  (유지)", data["SR_LEARN"], data["SR_FULL"])
    paired("LEARNED vs FIXED (참고)", data["SR_LEARN"], data["SR_FIXED"])

    print("\n" + "="*64)
    if e1 is None:
        print("E1 판정: 미완 — learned/random 3seed 끝나면 재실행"); return
    if e1["ci"][0] > 0 and e1["delta"] >= DELTA:
        print(f"★ E1 PASS: learned > random (Δ={e1['delta']:+.4f}, CI 하한 {e1['ci'][0]:+.4f}>0)")
        print("  → SR-XMA 계속. 다음 E2 Pareto.")
    elif e1["ci"][1] < 0:
        print(f"★ E1 FAIL(역전): random >= learned (CI 상한 {e1['ci'][1]:+.4f}<0). STOP.")
    else:
        print(f"★ E1 NEUTRAL: learned ≈ random (CI=[{e1['ci'][0]:+.4f},{e1['ci'][1]:+.4f}] 0 포함).")
        print("  → 라우팅 무의미. SR-XMA STOP, negative 기록 권장.")


if __name__ == "__main__":
    main()
