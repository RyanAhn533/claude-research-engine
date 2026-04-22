"""
Project_01 FER asset + Qwen agent: Korean cultural prior 효과 검증.

Setup:
  - 237K Korean FER image → landmark 17-feature description (text) for each sample
  - Clean subset (Yonsei is_selected=0) 400 stratified (100/class)
  - 4-class: 기쁨(happy) / 분노(angry) / 슬픔(sad) / 중립(neutral)
  - Agent input: landmark feature description (text)
  - Ablation: baseline vs +cultural(Korean prior)

Thesis prediction:
  - If cultural prior works only when Korean data (thesis) → +cultural here should show effect
    (unlike IEMOCAP/MELD null)
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

FACE_FEATURES = Path("/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv")
YONSEI_CONSENSUS = Path("/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry", "happy", "neutral", "sad"]
KR_TO_EN = {"기쁨": "happy", "분노": "angry", "슬픔": "sad", "중립": "neutral"}
LANDMARK_COLS = ["ear_avg", "mar", "mouth_width", "brow_height_avg",
                 "brow_furrow", "nose_bridge", "cheek_raise_avg",
                 "lip_corner_angle", "face_aspect_ratio", "chin_length", "forehead_height"]


def feature_to_text(row):
    """Convert numeric landmark features to compact descriptive text."""
    parts = []
    for col in LANDMARK_COLS:
        v = row.get(col)
        if v is not None and not pd.isna(v):
            parts.append(f"{col}={float(v):.3f}")
    return "; ".join(parts)


def build_prompt(feat_text, use_cultural=False):
    sys_p = (
        "You are a facial emotion classifier for a Korean subject. "
        "Given geometric landmark features of the face (eye aspect ratio, mouth aspect ratio, "
        "mouth_width, brow_height, cheek_raise, lip_corner_angle, etc.), "
        "classify the emotion into: angry, happy, neutral, sad. "
        "Reason briefly about which features support the decision.\n"
        "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    )
    if use_cultural:
        sys_p += "\n\n" + CULTURAL_PRIOR_KR
    user_p = f"Observed landmark features:\n{feat_text}\n\nClassify emotion."
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]


def load_balanced():
    feat = pd.read_csv(FACE_FEATURES)  # has is_selected + emotion columns
    df = feat[feat["is_selected"] == 0].copy()  # clean subset (Yonsei consensus passed)
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    # stratified 100 per class
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
    t0 = time.time()
    for i, row in test_df.iterrows():
        ft = feature_to_text(row)
        messages = build_prompt(ft, use_cultural)
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
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s  {el/(i+1)*1000:.0f}ms/sample")
    return preds


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
for name, uc in [("baseline", False), ("+cultural(Korean prior)", True)]:
    print(f"\n=== {name} ===")
    preds = run_config(test_df, model, tokenizer, uc)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0)
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    print(classification_report(y_true, preds, labels=LABELS, zero_division=0, digits=3))
    results[name] = {"acc": float(acc), "f1": float(f1), "use_cultural": uc, "preds": preds}

with open(OUT / "project01_fer_agent_results.json", "w") as f:
    json.dump({
        "task": "Korean FER with landmark features, 4-class",
        "N": len(test_df),
        "dist": dict(Counter(y_true)),
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
    }, f, indent=2, ensure_ascii=False)
print(f"\n[done] {OUT}/project01_fer_agent_results.json")
