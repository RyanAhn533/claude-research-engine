"""
K-EmoCon bio-proxy agent: aggregated arousal/valence (1-5) as bio signal proxy.
Agent 'reads' these as autonomic-state hints and predicts discrete emotion.

This is a simplified bio-grounded labeling experiment (text-free, Korean data).
Ground truth: label_discrete mapped to 4-class (ang/hap/neu/sad).
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE

sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import LABEL_4, CULTURAL_PRIOR_KR, parse_output

KEMO_PATH = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_003_kemocon_meta/cache/kemocon_aggregated.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABEL_ID_TO_NAME = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

MAP_4 = {
    "cheerful": "hap", "happy": "hap", "delight": "hap", "pride": "hap",
    "angry": "ang", "frustration": "ang", "contempt": "ang",
    "sad": "sad", "sorrow": "sad", "dejection": "sad",
    "concentration": "neu", "none_1": "neu", "none_2": "neu",
}

def build_prompt_bio(arousal, valence, use_cultural=False):
    sys_p = (
        "You are a bio-grounded emotion agent. "
        "Given arousal and valence ratings observed in a Korean subject during a debate, "
        "classify the person's emotional state into: angry, happy, neutral, sad. "
        "arousal (1=very calm, 5=very intense), valence (1=very negative, 5=very positive). "
        "Reason briefly.\n"
        "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    )
    if use_cultural:
        sys_p += "\n\n" + CULTURAL_PRIOR_KR

    user_p = (
        f"Observed bio-proxy:\n"
        f"  arousal = {arousal} / 5\n"
        f"  valence = {valence} / 5\n"
        f"Classify the subject's emotional state."
    )
    return [
        {"role": "system", "content": sys_p},
        {"role": "user",   "content": user_p},
    ]


def load_test_subset():
    df = pd.read_parquet(KEMO_PATH)
    df["label_4"] = df["label_discrete"].map(MAP_4)
    df = df[df["label_4"].notna()].copy()
    # 4-class stratified sample (K-EmoCon class imbalance 심함)
    pieces = []
    per = 100
    rng = np.random.default_rng(42)
    for lab in ["ang", "hap", "neu", "sad"]:
        sub = df[df["label_4"] == lab]
        if len(sub) == 0:
            print(f"  [warn] class {lab} has 0 samples in kemocon_aggregated")
            continue
        n = min(per, len(sub))
        if len(sub) > n:
            sub = sub.iloc[rng.choice(len(sub), size=n, replace=False)]
        pieces.append(sub)
    out = pd.concat(pieces).reset_index(drop=True)
    return out


def run_config(test_df, model, tokenizer, use_cultural):
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        messages = build_prompt_bio(int(row["arousal"]), int(row["valence"]), use_cultural)
        ps = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        label, _ = parse_output(raw)
        preds.append(label)
        if (i+1) % 50 == 0:
            el = time.time() - t0
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s {el/(i+1)*1000:.0f}ms/sample")
    return preds


# ===== Load =====
test_df = load_test_subset()
MAP_NAME_TO_FULL = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}
test_df["label_full"] = test_df["label_4"].map(MAP_NAME_TO_FULL)
print(f"[info] K-EmoCon bio-proxy subset: {len(test_df)}  dist: {Counter(test_df['label_full'].tolist())}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

y_true = test_df["label_full"].tolist()
results = {}
for name, uc in [("baseline", False), ("+cultural(Korean prior)", True)]:
    print(f"\n=== {name} ===")
    preds = run_config(test_df, model, tokenizer, uc)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", labels=LABEL_4, zero_division=0)
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    print(classification_report(y_true, preds, labels=LABEL_4, zero_division=0, digits=3))
    results[name] = {"acc": float(acc), "f1": float(f1), "use_cultural": uc, "preds": preds}

with open(OUT / "kemocon_bio_proxy_results.json", "w") as f:
    json.dump({
        "test_n": len(test_df),
        "class_dist": dict(Counter(y_true)),
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
    }, f, indent=2)
print(f"\n[done] {OUT}/kemocon_bio_proxy_results.json")
