"""exp_057 analysis — PAIRED. 어떤 AU 사전이 이기는가, 그리고 감정별로 갈리는가?

★설계상 paired인 이유:
   load(seed)가 결정적이라 같은 seed 안에서 5개 사전 조건이 **동일한 exemplar와
   동일한 test set**을 본다. 따라서 조건 간 비교는 seed별 쌍대차(paired delta)로 재야 한다.
   mean(B) - mean(A)로 재면 baseline의 seed 변동이 CI에 섞여 실제보다 넓어진다.

   실측 근거(qwen7b, 3seed): 절대 per-emotion은 seed에 크게 흔들리는데
   (angry: seed42 단독 1 → 3seed 평균 17) 조건 간 delta는 안정적이다
   (angry +11 → +13). 쌍대로 재야 하는 이유가 이것.

질문:
  Q1. 사전이 전체 정확도를 바꾸는가            -> paired delta ± CI
  Q2. ★감정별로 최적 사전이 갈리는가            -> 감정별 paired delta + argmax
  Q3. 교과서(B) vs 데이터유래(C)                -> 직접 대조
  Q4. 입 vs 눈 (Jack 2012 검증)                 -> D_mouth vs D_eyes
"""
import json, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = ["angry", "happy", "neutral", "sad"]
PRIOR_ORDER = ["A_none", "B_textbook", "C_derived", "D_mouth", "D_eyes"]
BASE = "A_none"


def load_all(shots=4):
    out = {}
    for f in sorted(glob.glob(str(HERE / "cache" / "au_prior_*.json"))):
        d = json.load(open(f))
        out[d["model"]] = {k.rsplit("_k", 1)[0]: v for k, v in d["results"].items()
                           if k.endswith(f"_k{shots}")}
    return out


def paired(a, b):
    """b - a, seed별 쌍대. 반환 (mean, std, n, 95%CI 반폭, 전 seed 동일부호 여부)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    if n == 0:
        return None
    d = b[:n] - a[:n]
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / np.sqrt(n) if n > 1 else float("nan")
    consistent = bool(np.all(d > 0) or np.all(d < 0))
    return m, sd, n, ci, consistent, d.tolist()


def main(shots=4):
    data = load_all(shots)
    if not data:
        print("no cache yet"); return
    models = list(data.keys())
    priors = [p for p in PRIOR_ORDER if any(p in data[m] for m in models)]
    others = [p for p in priors if p != BASE]
    print(f"\nmodels={models}  priors={priors}  k={shots}")

    # ── Q1 overall
    print(f"\n=== Q1. overall acc (%) ===")
    hdr = f"{'model':<10}" + "".join(f"{p:>15}" for p in priors)
    print(hdr); print("-" * len(hdr))
    for m in models:
        row = f"{m:<10}"
        for p in priors:
            r = data[m].get(p)
            row += f"{r['acc_mean']:>10.2f}±{r['acc_std']:<4.1f}" if r else f"{'-':>15}"
        print(row)

    print(f"\n=== Q1. PAIRED delta vs {BASE} (pp, seed별 쌍대) ===")
    print(f"{'model':<10}" + "".join(f"{p:>22}" for p in others))
    print("-" * (10 + 22 * len(others)))
    agg = {p: [] for p in others}
    for m in models:
        base = data[m].get(BASE)
        if not base or "acc_seeds" not in base: continue
        row = f"{m:<10}"
        for p in others:
            r = data[m].get(p)
            res = paired(base["acc_seeds"], r["acc_seeds"]) if r else None
            if res:
                mn, sd, n, ci, cons, ds = res
                agg[p].extend(ds)
                mark = "*" if cons else " "
                row += f"{mn:>+11.2f}±{ci:<5.2f}{mark}    "
            else:
                row += f"{'-':>22}"
        print(row)
    print("-" * (10 + 22 * len(others)))
    row = f"{'ALL':<10}"
    for p in others:
        if agg[p]:
            d = np.array(agg[p]); ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            row += f"{d.mean():>+11.2f}±{ci:<5.2f}     "
        else:
            row += f"{'-':>22}"
    print(row)
    print("  * = 전 seed에서 부호 일치")

    # ── Q2 ★ per-emotion paired
    print(f"\n=== Q2. ★감정별 PAIRED delta vs {BASE} (pp) ===")
    have_seeds = all("per_emotion_seeds" in data[m].get(BASE, {}) for m in models if BASE in data[m])
    if not have_seeds:
        print("  (per_emotion_seeds 없는 모델 존재 → 해당 모델은 seed평균 차이로 대체 표기)")
    best_by_emo = {}
    for l in LABELS:
        print(f"\n  [{l}]")
        vals = {}
        for p in others:
            ds = []
            for m in models:
                b, r = data[m].get(BASE), data[m].get(p)
                if not (b and r): continue
                if "per_emotion_seeds" in b and "per_emotion_seeds" in r:
                    res = paired(b["per_emotion_seeds"].get(l, []), r["per_emotion_seeds"].get(l, []))
                    if res: ds.extend(res[5])
                else:  # fallback: seed평균 차이 1개
                    bv = b.get("per_emotion_acc", {}).get(l)
                    rv = r.get("per_emotion_acc", {}).get(l)
                    if bv is not None and rv is not None: ds.append(rv - bv)
            if ds:
                d = np.array(ds)
                ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                vals[p] = d.mean()
                print(f"    {p:<12} {d.mean():>+7.2f} ± {ci:<5.2f}  (n={len(d)})")
        # baseline 절대값도 같이
        bl = [data[m][BASE]["per_emotion_acc"].get(l) for m in models
              if BASE in data[m] and data[m][BASE].get("per_emotion_acc")]
        bl = [x for x in bl if x is not None]
        if bl: print(f"    {'(A_none 절대)':<12} {np.mean(bl):>7.2f}")
        if vals:
            best_by_emo[l] = max(vals, key=vals.get)

    print(f"\n  감정별 최적 사전: {best_by_emo}")
    n_distinct = len(set(best_by_emo.values()))
    verdict = ("★감정별로 갈림 — 사전 효과는 감정 의존적 (Step 1 신호 있음)"
               if n_distinct > 1 else "갈리지 않음 — 단일 사전이 전 감정 우세")
    print(f"  서로 다른 사전이 이긴 감정 수: {n_distinct}/{len(best_by_emo)}  → {verdict}")

    # ── Q3 B vs C
    print("\n=== Q3. 교과서(B) vs 데이터유래(C) — paired ===")
    alld = []
    for m in models:
        b, c = data[m].get("B_textbook"), data[m].get("C_derived")
        if not (b and c and "acc_seeds" in b): continue
        res = paired(b["acc_seeds"], c["acc_seeds"])
        if not res: continue
        mn, sd, n, ci, cons, ds = res
        alld.extend(ds)
        tag = "C" if mn > 0 else "B"
        print(f"  {m:<10} B={b['acc_mean']:6.2f}  C={c['acc_mean']:6.2f}  C−B={mn:+6.2f}±{ci:<5.2f} -> {tag}")
    if alld:
        d = np.array(alld); ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  => 전체 C−B = {d.mean():+.2f} ± {ci:.2f} (n={len(d)})  "
              f"{'C(데이터유래) 우세' if d.mean() > 0 else 'B(교과서) 우세'}")

    # ── Q4 mouth vs eyes
    print("\n=== Q4. 입(D_mouth) vs 눈(D_eyes) — Jack 2012는 동아시아=눈 예측 ===")
    alld = []
    for m in models:
        mo, ey = data[m].get("D_mouth"), data[m].get("D_eyes")
        if not (mo and ey and "acc_seeds" in mo): continue
        res = paired(ey["acc_seeds"], mo["acc_seeds"])   # mouth - eyes
        if not res: continue
        mn, sd, n, ci, cons, ds = res
        alld.extend(ds)
        print(f"  {m:<10} mouth={mo['acc_mean']:6.2f}  eyes={ey['acc_mean']:6.2f}  "
              f"m−e={mn:+6.2f}±{ci:<5.2f}")
    if alld:
        d = np.array(alld); ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  => 전체 mouth−eyes = {d.mean():+.2f} ± {ci:.2f} (n={len(d)})")
        print(f"     우리 linear probe: mouth 75.2 vs eyes 61.5 → mouth 우세 예측")
        print(f"     Jack 2012(동아시아=눈) 예측과는 {'반대' if d.mean() > 0 else '일치'}")

    # ── parse fail
    fails = {m: {p: data[m][p].get("parse_fail_total", 0) for p in priors if p in data[m]}
             for m in models}
    if any(v for f in fails.values() for v in f.values()):
        print("\n=== parse fail (라벨 문자열 없는 응답) ===")
        for m, f in fails.items():
            if any(f.values()): print(f"  {m:<10} {f}")
    else:
        print("\n(parse fail 전 모델 0건)")
    print()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
