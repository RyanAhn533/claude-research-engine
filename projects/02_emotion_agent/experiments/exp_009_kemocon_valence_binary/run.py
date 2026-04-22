"""
K-EmoCon valence-binary agent (bio-proxy).
Task: high valence (>=3) vs low valence (<3) binary.
Agent receives arousal as bio-proxy hint (not valence — avoid leak).
This tests whether agent can infer valence from arousal + text-free bio cues.

ABLATION vs cultural prior injection for Korean data — thesis critical test.
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

KEMO_PATH = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_003_kemocon_meta/cache/kemocon_labels_full.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["positive", "negative"]

def build_prompt(arousal, use_cultural=False):
    sys_p = (
        "You are a bio-grounded emotion valence classifier. "
        "Given an observed AROUSAL level (1=calm, 5=very intense) of a Korean subject in a debate, "
        "and considering the subject's cultural context, "
        "predict whether valence is POSITIVE (positive affect) or NEGATIVE. "
        "Higher arousal does NOT always mean positive — the sign depends on culture and context.\n"
        "Output format:\nLABEL: <positive|negative>\nREASON: <one sentence>"
    )
    if use_cultural:
        sys_p += "\n\n" + CULTURAL_PRIOR_KR
    user_p = f"Observed:\n  arousal = {arousal} / 5\n\nPredict valence (positive or negative)."
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]


def parse_binary(raw):
    low = raw.lower()
    # search LABEL line first
    for line in raw.splitlines():
        if line.strip().upper().startswith("LABEL:"):
            val = line.split(":", 1)[1].strip().lower()
            if "pos" in val: return "positive"
            if "neg" in val: return "negative"
    # fallback
    if "positive" in low: return "positive"
    if "negative" in low: return "negative"
    return "positive"  # safer default


def load_balanced():
    df = pd.read_parquet(KEMO_PATH)
    # Use external (largest, most varied) annotation
    df = df[df["label_source"] == "external"].copy()
    df["valence_bin"] = df["valence"].apply(lambda v: "positive" if v >= 3 else "negative")
    # balanced 200/class
    pieces = []
    per = 200
    rng = np.random.default_rng(42)
    for lab in LABELS:
        sub = df[df["valence_bin"] == lab]
        n = min(per, len(sub))
        sub = sub.iloc[rng.choice(len(sub), size=n, replace=False)]
        pieces.append(sub)
    out = pd.concat(pieces).reset_index(drop=True)
    return out


# ===== Main =====
test_df = load_balanced()
print(f"[info] N={len(test_df)}  valence dist: {Counter(test_df['valence_bin'].tolist())}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

y_true = test_df["valence_bin"].tolist()
results = {}
for name, uc in [("baseline", False), ("+cultural", True)]:
    print(f"\n=== {name} ===")
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        messages = build_prompt(int(row["arousal"]), uc)
        ps = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        preds.append(parse_binary(raw))
        if (i+1) % 100 == 0:
            el = time.time() - t0
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s  {el/(i+1)*1000:.0f}ms/sample")
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, labels=LABELS, average="macro")
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    print(classification_report(y_true, preds, labels=LABELS, zero_division=0, digits=3))
    results[name] = {"acc": float(acc), "f1": float(f1), "use_cultural": uc, "preds": preds}

with open(OUT / "kemocon_valence_binary_results.json", "w") as f:
    json.dump({
        "task": "valence binary from arousal-only bio-proxy",
        "N": len(test_df),
        "label_dist": dict(Counter(y_true)),
        "random_baseline": 0.5,
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
    }, f, indent=2)
print(f"\n[done] {OUT}/kemocon_valence_binary_results.json")
