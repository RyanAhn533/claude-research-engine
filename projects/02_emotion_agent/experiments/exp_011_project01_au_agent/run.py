"""
Project_01 OpenGraphAU 41-AU intensity → Qwen agent.
FACS 표준 AU — LLM이 학습 data에서 접함 → semantic interpretation 가능 예상.
Cultural prior ablation on Korean data.
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
from agent.emotion_agent import CULTURAL_PRIOR_KR, parse_output

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry", "happy", "neutral", "sad"]
KR_TO_EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}

# FACS AU semantic labels (subset)
AU_DESC = {
    "AU1":  "inner brow raiser", "AU2":  "outer brow raiser",
    "AU4":  "brow lowerer",      "AU5":  "upper lid raiser",
    "AU6":  "cheek raiser",      "AU7":  "lid tightener",
    "AU9":  "nose wrinkler",     "AU10": "upper lip raiser",
    "AU12": "lip corner puller", "AU14": "dimpler",
    "AU15": "lip corner depressor", "AU17": "chin raiser",
    "AU20": "lip stretcher",     "AU23": "lip tightener",
    "AU25": "lips part",         "AU26": "jaw drop", "AU27": "mouth stretch",
}

KEY_AUS = list(AU_DESC.keys())


def au_to_text(row, threshold=50):
    """Convert AU intensity (0-100) to text, keeping only active AUs."""
    active = []
    for au in KEY_AUS:
        if au in row and not pd.isna(row[au]) and row[au] >= threshold:
            active.append(f"{au} ({AU_DESC[au]})={row[au]:.0f}")
    if not active:
        # list top-3 anyway
        scored = [(au, float(row[au])) for au in KEY_AUS if au in row and not pd.isna(row[au])]
        scored.sort(key=lambda x: -x[1])
        active = [f"{a} ({AU_DESC[a]})={v:.0f}" for a, v in scored[:3]]
    return ", ".join(active)


def build_prompt(au_text, use_cultural=False):
    sys_p = (
        "You are a facial emotion classifier using FACS action units (AUs). "
        "Given detected AU intensities (0-100 scale) of a Korean subject, "
        "infer emotion based on FACS prototypes: "
        "Happy=AU6+AU12; Sad=AU1+AU4+AU15; Angry=AU4+AU5+AU7+AU23; Neutral=low overall. "
        "Classify into: angry, happy, neutral, sad.\n"
        "Output format:\nLABEL: <label>\nREASON: <one sentence citing specific AU>"
    )
    if use_cultural:
        sys_p += "\n\n" + CULTURAL_PRIOR_KR
    user_p = f"Detected AUs:\n{au_text}\n\nClassify emotion."
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]


def load_balanced():
    df = pd.read_parquet(AU_PARQUET)
    # is_selected=0 (Yonsei clean) only
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    per = 100
    rng = np.random.default_rng(42)
    pieces = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        if len(sub) > per:
            sub = sub.iloc[rng.choice(len(sub), size=per, replace=False)]
        pieces.append(sub)
    return pd.concat(pieces).reset_index(drop=True)


def run_config(test_df, model, tokenizer, use_cultural):
    preds = []
    reasons = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        au_text = au_to_text(row)
        messages = build_prompt(au_text, use_cultural)
        ps = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        label, reason = parse_output(raw)
        preds.append(label)
        reasons.append(reason)
        if (i+1) % 50 == 0:
            el = time.time() - t0
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s  {el/(i+1)*1000:.0f}ms/sample")
    return preds, reasons


# ===== Main =====
test_df = load_balanced()
print(f"[info] N={len(test_df)}  dist: {Counter(test_df['emotion_en'].tolist())}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

y_true = test_df["emotion_en"].tolist()
results = {}
for name, uc in [("baseline (FACS prototype)", False), ("+cultural(Korean prior)", True)]:
    print(f"\n=== {name} ===")
    preds, reasons = run_config(test_df, model, tokenizer, uc)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0)
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    print(classification_report(y_true, preds, labels=LABELS, zero_division=0, digits=3))
    results[name] = {
        "acc": float(acc), "f1": float(f1),
        "use_cultural": uc, "preds": preds,
        "reasons_sample": reasons[:5],
    }

with open(OUT / "project01_au_agent_results.json", "w") as f:
    json.dump({
        "task": "Korean FER with OpenGraphAU 41-AU intensity, 4-class, FACS-aware prompt",
        "N": len(test_df),
        "dist": dict(Counter(y_true)),
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
    }, f, indent=2, ensure_ascii=False)
print(f"\n[done] {OUT}/project01_au_agent_results.json")
