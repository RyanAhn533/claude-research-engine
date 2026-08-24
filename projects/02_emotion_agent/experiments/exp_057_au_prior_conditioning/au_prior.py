"""
exp_057: AU PRIOR CONDITIONING -> does the *choice of AU prior* change VLM/LLM emotion judgment?

TWO things are being fixed/tested here.

(1) ***BUG FIX (blocking)***  exp_038's numeric_fmt assumes AU intensities are 0-100 and
    filters `row[au] >= thr(50)`. The parquet actually stores 0-1 floats
    (opengraphau_41au_237k_v2.parquet: min 0.0 / max 0.9956 / mean 0.134).
    Measured: 0 of 205,336 rows have ANY key-AU >= 50, so the threshold branch NEVER fires
    and every prompt falls through to the top-3 fallback, where f"{v:.0f}" rounds 0-1 floats
    to 0 or 1. Empirically 92% of rendered values are "1", 8% are "0".
    => the model receives "which 3 AUs are largest" and NO intensity, while the system prompt
       claims "AU intensities (0-100)".
    Fix here: scale to 0-100 (AU_SCALE) and threshold in the native unit. Old experiments are
    NOT touched (exp_038/039/040/056 keep their renderer) so published numbers stay reproducible.

(2) ***PRIOR CONDITIONING (the actual question)***  Four prompt conditions differing ONLY in
    what AU->emotion prior is stated:
      A none      : AU values only (current no-FACS baseline)
      B textbook  : Ekman/EMFACS prototypes (Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23)
      C derived   : ranking learned from OUR data (per-AU linear probe, phase0/03)
      D region    : region-level instruction, mouth-first vs eyes-first (Jack 2012 PNAS
                    predicts eyes for East Asians; our linear probe says mouth 75.2 vs eyes 61.5)

Design note (why emotion-wise first, not country-wise):
    A cross-dataset ("per country") comparison confounds nation with capture protocol.
    Our own 2026-08-04 finding is that the Korean AI-Hub *anger* labels are contaminated
    (acted collection; EmotiEffLib pass-group Anger 0.06, POSTER++ pass-group Anger 0.20 <
    Fear 0.25, blind judges consensus-excluded 86% of rejected anger). So a Korea-vs-West
    anger difference would measure collection method, not culture. Step 1 therefore stays
    inside ONE dataset and asks whether the best prior differs BY EMOTION.

Reuses exp_035 loaders/parser and exp_038 sampling. Batched 4-bit, greedy, N=400, k=4.
"""
import os, sys, json, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "exp_035_multimodel_gating"))
sys.path.insert(0, str(HERE.parents[1] / "exp_038_format_intervention"))
import run_gating as G

OUT = HERE.parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)
SEEDS_FULL = [42, 123, 777, 2024, 31337, 1234, 9001]

AU_SCALE = 100.0          # parquet is 0-1; prompt claims 0-100
ACTIVE_THR = 20.0         # in 0-100 units. mean*100 = 13.4, so ~20 keeps the salient ones
MAX_AUS = 8               # cap so the prompt stays readable
MIN_AUS = 3               # if nothing passes the threshold, show top-3 (never emit empty)


# ───────────────────────────────── renderer (BUG-FIXED)

def numeric_fmt_fixed(row, thr=ACTIVE_THR, scale=AU_SCALE, max_aus=MAX_AUS, min_aus=MIN_AUS):
    """AU -> 'AU6 (cheek raiser)=87, AU12 (lip corner puller)=91'  (values now 0-100)."""
    vals = [(au, float(row[au]) * scale) for au in G.KEY_AUS
            if au in row and not pd.isna(row[au])]
    vals.sort(key=lambda x: -x[1])
    active = [(a, v) for a, v in vals if v >= thr][:max_aus]
    if len(active) < min_aus:
        active = vals[:min_aus]
    return ", ".join(f"{a} ({G.AU_DESC[a]})={v:.0f}" for a, v in active)


def load(seed):
    """exp_038.load with the fixed renderer (same sampling: k=4 exemplars + 100/class test)."""
    df = pd.read_parquet(G.AU_PARQUET)
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(G.KR_TO_EN)
    df = df[df["emotion_en"].notna()]
    rng = np.random.default_rng(seed)
    ex_idx = [df[df["emotion_en"] == l].iloc[rng.integers(0, (df["emotion_en"] == l).sum())].name
              for l in G.LABELS]
    ex_df = df.loc[ex_idx]
    exemplars = [(numeric_fmt_fixed(r), r["emotion_en"]) for _, r in ex_df.iterrows()]
    remaining = df.drop(index=ex_idx)
    rng2 = np.random.default_rng(seed)
    parts = []
    for l in G.LABELS:
        sub = remaining[remaining["emotion_en"] == l]
        if len(sub) > 100:
            sub = sub.iloc[rng2.choice(len(sub), size=100, replace=False)]
        parts.append(sub)
    test_df = pd.concat(parts)
    items = [(numeric_fmt_fixed(r), r["emotion_en"]) for _, r in test_df.iterrows()]
    return items, exemplars


# ───────────────────────────────── the four priors

PRIOR_NONE = ""

# B. Ekman / EMFACS prototypes (same string exp_035 used, + neutral spelled out)
PRIOR_TEXTBOOK = (
    "FACS prototypes (Ekman): Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23; "
    "Neutral=no strong AU activation.\n"
)

# C. derived from OUR data: outputs/phase0/03_per_au_linear_probe/per_au_ranking.csv
#    (single-AU linear probe accuracy on 4-class, chance 25%)
PRIOR_DERIVED = (
    "Empirically, these AUs carry the most emotion information in this dataset "
    "(single-AU discriminability, chance=25%): "
    "AU6 cheek raiser 43%, AU12 lip corner puller 43%, AU10 upper lip raiser 42%, "
    "AU7 lid tightener 41%, AU25 lips part 40%, AU15 lip corner depressor 37%, "
    "AU9 nose wrinkler 36%, AU4 brow lowerer 34%. Weight them accordingly.\n"
)

# D. region-level instruction, two directions
PRIOR_REGION_MOUTH = (
    "Focus on the mouth and cheek region (AU6, AU10, AU12, AU14, AU15, AU20, AU23, AU25, "
    "AU26, AU27); it is the most diagnostic region for this population.\n"
)
PRIOR_REGION_EYES = (
    "Focus on the eye and brow region (AU1, AU2, AU4, AU5, AU7); it is the most diagnostic "
    "region for this population.\n"
)

# ── 판별 조건 (2026-08-05 추가) ───────────────────────────────────────────
# 관찰: 3모델 20칸 부호검사에서 살아남은 4칸 중 가장 깨끗한 것이
#   B_textbook -> angry +5~+13 (3/3 일치).
# 반면 C_derived(AU 이름+중요도)와 D_eyes(AU 이름+영역지시)는 같은 AU4/5/7을
# 언급하는데도 분노 방향이 모델마다 갈렸다.
#   => 가설 H: 편향을 만드는 것은 "AU를 호명하는 것"이 아니라
#              "감정↔AU 매핑을 명시하는 것"이다.
# 판별: 분노 매핑만 남긴 E vs 같은 AU를 매핑 없이 이름만 준 F.
#   E에서 angry↑ AND F에서 angry~0  -> H 지지
#   둘 다 angry↑                    -> H 기각 (호명만으로 충분)
#   둘 다 angry~0                   -> B의 효과는 다른 감정 매핑과의 상호작용
E_MAP_ONLY = "FACS prototype (Ekman): Angry=AU4+5+7+23.\n"
F_NAMES_ONLY = ("The following AUs are important in this dataset: "
                "AU4 (brow lowerer), AU5 (upper lid raiser), AU7 (lid tightener), "
                "AU23 (lip tightener).\n")

PRIORS = {
    "A_none":        PRIOR_NONE,
    "B_textbook":    PRIOR_TEXTBOOK,
    "C_derived":     PRIOR_DERIVED,
    "D_mouth":       PRIOR_REGION_MOUTH,
    "D_eyes":        PRIOR_REGION_EYES,
    "E_map_only":    E_MAP_ONLY,      # 매핑만 (AU 동일)
    "F_names_only":  F_NAMES_ONLY,    # 이름만 (AU 동일, 매핑 없음)
}


def prompt_au(x, ex, prior=""):
    """Mirrors exp_038 prompt_numeric exactly; the ONLY delta is the inserted prior line."""
    sys_p = ("You are a facial emotion classifier using FACS action units (AUs). "
             "Given detected AU intensities (0-100) of a Korean subject, classify into: "
             "angry, happy, neutral, sad.\n")
    sys_p += prior
    if ex:
        sys_p += f"Here are {len(ex)} Korean examples with correct labels:\n"
        for i, (t, l) in enumerate(ex):
            sys_p += f"\nExample {i+1}:\n  AUs: {t}\n  LABEL: {l}\n"
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    u = f"Target AUs:\n{x}\n\nClassify emotion." if ex else f"AUs: {x}\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": u}]


# ───────────────────────────────── eval

@torch.inference_mode()
def run_cfg(items, ex, model, tok, prior, shots, bs=32):
    """shots in {0, 4}. Returns (acc, macro_f1, per_emotion_acc, n_parse_fail)."""
    use_ex = ex if shots else None
    y = [g for _, g in items]
    msgs = [prompt_au(x if x else "AU6 (cheek raiser)=0", use_ex, prior) for x, _ in items]
    preds, fails = [], 0
    for b in range(0, len(msgs), bs):
        ps = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
              for m in msgs[b:b + bs]]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for g in out[:, enc.input_ids.shape[1]:]:
            raw = tok.decode(g, skip_special_tokens=True)
            # explicit parse-failure accounting (exp_035 silently maps failures to "neutral")
            if not any(l in raw.lower() for l in G.LABELS):
                fails += 1
            preds.append(G.parse_output(raw))
    acc = float(accuracy_score(y, preds))
    f1 = float(f1_score(y, preds, labels=G.LABELS, average="macro", zero_division=0))
    per_emo = {}
    for l in G.LABELS:
        idx = [i for i, g in enumerate(y) if g == l]
        per_emo[l] = float(np.mean([preds[i] == l for i in idx])) if idx else None
    return acc, f1, per_emo, fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--priors", default="A_none,B_textbook,C_derived,D_mouth,D_eyes")
    ap.add_argument("--shots", default="4")          # "0,4" to also get zero-shot
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    SEEDS = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS_FULL
    priors = [p.strip() for p in args.priors.split(",")]
    shots_list = [int(s) for s in args.shots.split(",")]

    snap = glob.glob(f"{G.HUB}/models--{G.MODELS[args.model]}/snapshots/*")[0]
    print(f"[load] {args.model} seeds={SEEDS} priors={priors} shots={shots_list}", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16,
                                                 trust_remote_code=True).eval()

    results = {}
    for shots in shots_list:
        for pname in priors:
            accs, f1s, fails_tot = [], [], 0
            per_emo_acc = {l: [] for l in G.LABELS}
            for s in SEEDS:
                items, ex = load(s)
                a, f1, pe, nf = run_cfg(items, ex, model, tok, PRIORS[pname], shots, bs=args.bs)
                accs.append(a); f1s.append(f1); fails_tot += nf
                for l in G.LABELS:
                    if pe[l] is not None: per_emo_acc[l].append(pe[l])
            key = f"{pname}_k{shots}"
            results[key] = {
                "acc_mean": float(np.mean(accs) * 100), "acc_std": float(np.std(accs) * 100),
                "f1_mean": float(np.mean(f1s)), "acc_seeds": [float(a * 100) for a in accs],
                "per_emotion_acc": {l: float(np.mean(v) * 100) for l, v in per_emo_acc.items() if v},
                # per-SEED per-emotion — Q2(감정별 최적 사전)를 paired로 재려면 필수.
                # 조건 간 exemplar/test는 seed가 같으면 동일하므로(load(seed) 결정적) 쌍대 비교가 성립한다.
                "per_emotion_seeds": {l: [float(x * 100) for x in v] for l, v in per_emo_acc.items() if v},
                "parse_fail_total": int(fails_tot),
            }
            pe_str = " ".join(f"{l[:3]}={np.mean(v)*100:.0f}" for l, v in per_emo_acc.items() if v)
            print(f"  {args.model} {key}: acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f} "
                  f"f1={np.mean(f1s):.3f} [{pe_str}] fail={fails_tot}", flush=True)

    # deltas vs A_none at the same shot count
    for shots in shots_list:
        base = results.get(f"A_none_k{shots}")
        if not base: continue
        for pname in priors:
            k = f"{pname}_k{shots}"
            if k in results:
                results[k]["delta_vs_none_pp"] = results[k]["acc_mean"] - base["acc_mean"]

    json.dump({"model": args.model, "seeds": SEEDS,
               "renderer": {"scale": AU_SCALE, "active_thr": ACTIVE_THR,
                            "max_aus": MAX_AUS, "min_aus": MIN_AUS},
               "results": results},
              open(OUT / f"au_prior_{args.model}.json", "w"), indent=2)
    print(f"[SAVED] au_prior_{args.model}.json", flush=True)
