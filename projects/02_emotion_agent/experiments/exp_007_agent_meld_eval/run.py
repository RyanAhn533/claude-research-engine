"""
Agent MELD eval — 400 stratified from official test split.
Ablation: baseline / +cultural / +context (prev 2 utterances) / full
(prosody는 MELD parquet에 없으므로 제외)
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

MELD_PATH = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_002_meld_preproc/cache/meld_4class.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABEL_ID_TO_NAME = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

def build_prompt_meld(text, context_list=None, use_cultural=False):
    """MELD variant: optionally inject prev 2 utterances as context."""
    sys_p = (
        "You are a multimodal emotion recognition agent. "
        "Given a conversation utterance (optionally with context), "
        "classify into one of: angry, happy, neutral, sad. "
        "Output format:\nLABEL: <label>\nREASON: <one sentence>"
    )
    if use_cultural:
        sys_p += "\n\n" + CULTURAL_PRIOR_KR

    user_parts = []
    if context_list:
        user_parts.append("Previous context (last 2 utterances):")
        for c in context_list:
            user_parts.append(f"  - {c}")
    user_parts.append(f"\nCurrent utterance: \"{text}\"")
    user_parts.append("\nClassify emotion of current utterance.")

    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def load_test_subset():
    df = pd.read_parquet(MELD_PATH)
    test = df[df["split"] == "test"].copy()
    test = test.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    # Attach prev 2 utterances within same dialogue
    test["ctx1"] = test.groupby("dialogue_id")["text"].shift(1)
    test["ctx2"] = test.groupby("dialogue_id")["text"].shift(2)
    # Stratified 400 (100/class)
    pieces = []
    per = 100
    rng = np.random.default_rng(42)
    for lab in [0, 1, 2, 3]:
        sub = test[test["label_4"] == lab]
        if len(sub) > per:
            sub = sub.iloc[rng.choice(len(sub), size=per, replace=False)]
        pieces.append(sub)
    out = pd.concat(pieces).reset_index(drop=True)
    return out


def run_config(test_df, model, tokenizer, use_cultural, use_context):
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        text = str(row["text"])
        ctx = []
        if use_context:
            if isinstance(row.get("ctx2"), str):
                ctx.append(row["ctx2"])
            if isinstance(row.get("ctx1"), str):
                ctx.append(row["ctx1"])
        messages = build_prompt_meld(text, ctx if ctx else None, use_cultural)
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
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


# ===== Load =====
test_df = load_test_subset()
print(f"[info] MELD test subset: {len(test_df)}  dist: {Counter(test_df['label_4'].map(LABEL_ID_TO_NAME).tolist())}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

y_true = test_df["label_4"].map(LABEL_ID_TO_NAME).tolist()

configs = [
    ("baseline",          False, False),
    ("+cultural",         True,  False),
    ("+context(prev2)",   False, True),
    ("full(+cultural+context)", True, True),
]

results = {}
for name, uc, uctx in configs:
    print(f"\n=== {name} ===")
    preds = run_config(test_df, model, tokenizer, uc, uctx)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", labels=LABEL_4, zero_division=0)
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    print(classification_report(y_true, preds, labels=LABEL_4, zero_division=0, digits=3))
    results[name] = {"acc": float(acc), "f1": float(f1), "use_cultural": uc, "use_context": uctx, "preds": preds}

with open(OUT / "agent_meld_results.json", "w") as f:
    json.dump({
        "test_n": len(test_df),
        "configs": {k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in results.items()},
    }, f, indent=2)
print(f"\n[done] {OUT}/agent_meld_results.json")
