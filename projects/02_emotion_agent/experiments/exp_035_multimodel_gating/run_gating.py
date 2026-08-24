"""
exp_035: Multi-model gating replication.

Goal: kill the single-model desk-reject. Replicate the modality-gated ICL finding
(novel input AU-text → big ICL gain; familiar English dialog → flat) on >1 model family.

One model load → 3 datasets {au(novel), iemocap(familiar), meld(familiar)} × 3 seeds ×
2 configs {zero_shot, icl_k4}. Mirrors exp_012 (AU) and exp_021/022 (text) prompt logic
exactly so numbers are comparable to the existing Qwen2.5-7B results.

Usage:
  python run_gating.py --model mistral
  python run_gating.py --model qwen7b   # sanity re-check vs published numbers
"""
import os, sys, json, time, glob, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
HUB = f"{HF_CACHE}/hub"

P = Path("/home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent")
AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
IEMOCAP_PARQUET = P / "experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet"
MELD_PARQUET = P / "experiments/exp_002_meld_preproc/cache/meld_4class.parquet"
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry", "happy", "neutral", "sad"]
SEEDS = [42, 123, 777]

MODELS = {
    "qwen7b":   "Qwen--Qwen2.5-7B-Instruct",
    "mistral":  "mistralai--Mistral-7B-Instruct-v0.3",
    "llama8b":  "meta-llama--Llama-3.1-8B-Instruct",
    "qwen14b":  "Qwen--Qwen2.5-14B-Instruct",
    "phi35":    "microsoft--Phi-3.5-mini-instruct",
    "yi6b":     "01-ai--Yi-1.5-6B-Chat",
    "qwen3b":   "Qwen--Qwen2.5-3B-Instruct",
    "falcon7b": "tiiuae--Falcon3-7B-Instruct",
}

# ---------- parse (inlined from agent.emotion_agent) ----------
def parse_output(raw: str):
    label = None
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            lab = line.split(":", 1)[1].strip().lower()
            for l in LABELS:
                if l in lab:
                    label = l; break
    if label is None:
        low = raw.lower()
        for l in LABELS:
            if l in low:
                label = l; break
    return label or "neutral"

# ---------- AU (novel modality) — mirrors exp_012 ----------
KR_TO_EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}
AU_DESC = {
    "AU1": "inner brow raiser", "AU2": "outer brow raiser", "AU4": "brow lowerer",
    "AU5": "upper lid raiser", "AU6": "cheek raiser", "AU7": "lid tightener",
    "AU9": "nose wrinkler", "AU10": "upper lip raiser", "AU12": "lip corner puller",
    "AU14": "dimpler", "AU15": "lip corner depressor", "AU17": "chin raiser",
    "AU20": "lip stretcher", "AU23": "lip tightener", "AU25": "lips part",
    "AU26": "jaw drop", "AU27": "mouth stretch",
}
KEY_AUS = list(AU_DESC.keys())

def au_to_text(row, threshold=50):
    active = []
    for au in KEY_AUS:
        if au in row and not pd.isna(row[au]) and row[au] >= threshold:
            active.append(f"{au} ({AU_DESC[au]})={row[au]:.0f}")
    if not active:
        scored = [(au, float(row[au])) for au in KEY_AUS if au in row and not pd.isna(row[au])]
        scored.sort(key=lambda x: -x[1])
        active = [f"{a} ({AU_DESC[a]})={v:.0f}" for a, v in scored[:3]]
    return ", ".join(active)

def au_prompt(target, exemplars):
    sys_p = (
        "You are a facial emotion classifier using FACS action units (AUs). "
        "Given detected AU intensities (0-100) of a Korean subject, classify into: "
        "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23.\n"
    )
    if exemplars:
        sys_p += f"Here are {len(exemplars)} Korean examples with correct labels:\n"
        for i, (t, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  AUs: {t}\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target.\n"
    sys_p += "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    user_p = f"Target AUs:\n{target}\n\nClassify emotion." if exemplars else f"AUs: {target}\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]

def au_load(seed):
    df = pd.read_parquet(AU_PARQUET)
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    rng = np.random.default_rng(seed)
    ex_idx = [df[df["emotion_en"] == l].iloc[rng.integers(0, len(df[df["emotion_en"] == l]))].name for l in LABELS]
    ex_df = df.loc[ex_idx]
    remaining = df.drop(index=ex_idx)
    rng2 = np.random.default_rng(seed)
    pieces = []
    for l in LABELS:
        sub = remaining[remaining["emotion_en"] == l]
        if len(sub) > 100:
            sub = sub.iloc[rng2.choice(len(sub), size=100, replace=False)]
        pieces.append(sub)
    test = pd.concat(pieces).reset_index(drop=True)
    exemplars = [(au_to_text(r), r["emotion_en"]) for _, r in ex_df.iterrows()]
    items = [(au_to_text(r), r["emotion_en"]) for _, r in test.iterrows()]
    return items, exemplars

# ---------- text (familiar modality) — mirrors exp_021/022 ----------
def text_prompt(text, exemplars):
    sys_p = ("You are an emotion classifier for spoken English utterances. "
             "Given a transcription, classify the speaker's emotion into: angry, happy, neutral, sad.\n")
    if exemplars:
        sys_p += f"\nHere are {len(exemplars)} labeled examples:\n"
        for i, (t, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  Utterance: \"{t}\"\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
    user_p = f"Utterance: \"{text}\"\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]

def text_load(seed, parquet, text_col, label_col, label_map):
    df = pd.read_parquet(parquet)
    df["emotion_en"] = df[label_col].map(label_map)
    df = df[df["emotion_en"].isin(LABELS)].copy()
    rng = np.random.default_rng(seed)
    test_idx = []
    for l in LABELS:
        sub = df[df["emotion_en"] == l]
        n = min(100, len(sub))
        test_idx.extend(sub.iloc[rng.choice(len(sub), size=n, replace=False)].index.tolist())
    test_df = df.loc[test_idx]
    remaining = df.drop(index=test_idx)
    ex_idx = [remaining[remaining["emotion_en"] == l].iloc[rng.integers(0, len(remaining[remaining["emotion_en"] == l]))].name for l in LABELS]
    ex_df = remaining.loc[ex_idx]
    exemplars = [(r[text_col], r["emotion_en"]) for _, r in ex_df.iterrows()]
    items = [(r[text_col] if not pd.isna(r[text_col]) else "", r["emotion_en"]) for _, r in test_df.iterrows()]
    return items, exemplars

DATASETS = {
    "au":      ("novel",    lambda s: au_load(s), au_prompt),
    "iemocap": ("familiar", lambda s: text_load(s, IEMOCAP_PARQUET, "transcription", "label_name",
                                                 {"ang": "angry", "hap": "happy", "exc": "happy", "neu": "neutral", "sad": "sad"}), text_prompt),
    "meld":    ("familiar", lambda s: text_load(s, MELD_PARQUET, "text", "label_4_name",
                                                 {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}), text_prompt),
}

def run_cfg(items, exemplars, model, tok, use_icl):
    y_true, preds = [], []
    t0 = time.time()
    for i, (x, gold) in enumerate(items):
        y_true.append(gold)
        if not x:
            preds.append("neutral"); continue
        msgs = prompt_fn(x, exemplars if use_icl else None)
        ps = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        preds.append(parse_output(tok.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)))
        if (i + 1) % 200 == 0:
            print(f"      [{i+1}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
    return acc, f1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--datasets", default="au,iemocap,meld")
    args = ap.parse_args()

    snaps = glob.glob(f"{HUB}/models--{MODELS[args.model]}/snapshots/*")
    assert snaps, f"snapshot not cached for {args.model} ({MODELS[args.model]})"
    snap = snaps[0]
    print(f"[load] {args.model} <- {snap}", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16)
    model.eval()
    print(f"[load] VRAM {torch.cuda.memory_allocated()/1024**2:.0f} MB", flush=True)

    results = {}
    for ds in args.datasets.split(","):
        novelty, loader, prompt_fn = DATASETS[ds]
        per = {"zero_shot": {"acc": [], "f1": []}, "icl_k4": {"acc": [], "f1": []}}
        for seed in SEEDS:
            items, exemplars = loader(seed)
            dist = Counter(g for _, g in items)
            print(f"\n#### {args.model} / {ds} ({novelty}) / seed {seed}  N={len(items)} {dict(dist)}", flush=True)
            for cfg, use_icl in [("zero_shot", False), ("icl_k4", True)]:
                acc, f1 = run_cfg(items, exemplars, model, tok, use_icl)
                per[cfg]["acc"].append(acc); per[cfg]["f1"].append(f1)
                print(f"    {cfg:10s} acc={acc*100:.2f}% f1={f1:.3f}", flush=True)
        agg = {}
        for cfg, d in per.items():
            a = np.array(d["acc"])
            agg[cfg] = {"acc_mean": float(a.mean()), "acc_std": float(a.std(ddof=1)), "accs": d["acc"]}
        agg["icl_gain_pp"] = (agg["icl_k4"]["acc_mean"] - agg["zero_shot"]["acc_mean"]) * 100
        results[ds] = {"novelty": novelty, **agg}
        print(f"  >>> {args.model}/{ds}: zero={agg['zero_shot']['acc_mean']*100:.2f}% "
              f"icl={agg['icl_k4']['acc_mean']*100:.2f}%  GAIN={agg['icl_gain_pp']:+.2f}pp", flush=True)

    outpath = OUT / f"gating_{args.model}.json"
    with open(outpath, "w") as f:
        json.dump({"model": args.model, "model_id": MODELS[args.model], "seeds": SEEDS, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {outpath}", flush=True)
    print("[SUMMARY] " + " | ".join(f"{ds}:{r['icl_gain_pp']:+.1f}pp({r['novelty']})" for ds, r in results.items()), flush=True)
