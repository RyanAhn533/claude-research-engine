"""
Agent IEMOCAP full test inference.
Ablation: {+cultural, -cultural} × {+prosody, -prosody}
Test subset: 20% stratified (seed 42) to match text baseline split.
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import (
    build_prompt, parse_output, LABEL_4, CULTURAL_PRIOR_KR,
)

IEMOCAP_PATH = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

# IEMOCAP label_4 map matches LABEL_4 order? Check:
# LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3} from exp_001
# LABEL_4 = ["angry", "happy", "neutral", "sad"] — same order. We'll map label 0-3.
LABEL_ID_TO_NAME = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

# Limit test size for Qwen batch=1 inference time. Use stratified subsample.
TEST_N = 400  # 100 per class stratified — tractable; full 1376 later if OK

def load_test_subset():
    df = pd.read_parquet(IEMOCAP_PATH)
    # stratified train/test mirroring exp_004 (seed 42, 80/20)
    idx = np.arange(len(df))
    _, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=df["label"].values)
    test = df.iloc[te_idx].copy()
    # stratified subsample to TEST_N
    pieces = []
    per = TEST_N // 4
    rng = np.random.default_rng(42)
    for lab in range(4):
        sub = test[test["label"] == lab]
        if len(sub) > per:
            sub = sub.iloc[rng.choice(len(sub), size=per, replace=False)]
        pieces.append(sub)
    out = pd.concat(pieces).reset_index(drop=True)
    return out

def run_agent(test_df, model, tokenizer, use_cultural, use_prosody):
    predictions = []
    reasonings = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        text = str(row.get("transcription", "") or "")
        prosody = None
        if use_prosody:
            prosody = {}
            for k in ["speaking_rate", "pitch_mean", "pitch_std", "rms"]:
                v = row.get(k)
                if v is not None and not pd.isna(v):
                    prosody[k] = float(v)
            if not prosody:
                prosody = None

        messages = build_prompt(text, prosody, use_cultural)
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=48, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        )
        label, reason = parse_output(raw)
        predictions.append(label)
        reasonings.append(reason)

        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(test_df)}] {elapsed:.1f}s, {elapsed/(i+1)*1000:.0f}ms/sample")
    return predictions, reasonings

# ===== Load =====
test_df = load_test_subset()
print(f"[info] test subset: {len(test_df)}  class dist: {Counter(test_df['label'].map(LABEL_ID_TO_NAME).tolist())}")

# Load model once
snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
print(f"[info] loading Qwen from {snap}")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] model loaded. VRAM used: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

# ===== Ablation =====
configs = [
    ("baseline (no cultural, no prosody)",   False, False),
    ("+cultural (no prosody)",               True,  False),
    ("+prosody (no cultural)",               False, True),
    ("full (+cultural +prosody)",            True,  True),
]

y_true = test_df["label"].map(LABEL_ID_TO_NAME).tolist()
results = {}

for name, uc, up in configs:
    print(f"\n=== {name} ===")
    preds, reasons = run_agent(test_df, model, tokenizer, uc, up)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", labels=LABEL_4, zero_division=0)
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    print(classification_report(y_true, preds, labels=LABEL_4, zero_division=0, digits=3))

    results[name] = {
        "acc": float(acc), "f1": float(f1),
        "use_cultural": uc, "use_prosody": up,
        "predictions": preds,
        "reasonings_sample": reasons[:5],
    }

with open(OUT / "agent_iemocap_results.json", "w") as f:
    json.dump({
        "test_n": len(test_df),
        "label_dist": dict(Counter(y_true)),
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in results.items()},
        "random_baseline_4class": 0.25,
    }, f, indent=2, ensure_ascii=False)

# Also save per-sample results
detail = test_df[["utt_id", "transcription", "label", "speaking_rate", "pitch_mean"]].copy() if "utt_id" in test_df.columns else test_df[["transcription", "label"]].copy()
for name, r in results.items():
    col = "pred_" + name.split(" (")[0].replace(" ", "_").replace("+", "p").replace("-", "m")
    detail[col] = r["predictions"]
detail.to_parquet(OUT / "agent_iemocap_perSample.parquet", index=False)

print(f"\n[done] results: {OUT}/agent_iemocap_results.json")
print(f"[done] per-sample: {OUT}/agent_iemocap_perSample.parquet")
