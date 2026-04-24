"""
exp_021: IEMOCAP cross-domain replication of 3-tier (T1 zero-shot + T2 ICL k=4).

Tests whether our Korean FER AU finding (3-tier + ICL gain) transfers to
English conversational text emotion (IEMOCAP transcription, 4-class).

Tier 3 (LoRA) skipped tonight (30min × 3 seeds = 105min) — do tomorrow if time.

3 seeds × 2 tiers × 400 samples ≈ 70min (tier 2 slower due to longer context).
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE

sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import parse_output

IEMOCAP_PARQUET = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

# IEMOCAP 4-class: angry/happy/neutral/sad
LABEL_MAP = {"ang": "angry", "hap": "happy", "exc": "happy", "neu": "neutral", "sad": "sad"}
LABELS = ["angry", "happy", "neutral", "sad"]

SEEDS = [42, 123, 777]
N_TEST_PER_CLASS = 100
N_EXEMPLARS_K4 = 4


SYS_BASE = (
    "You are an emotion classifier for spoken English utterances. "
    "Given a transcription, classify the speaker's emotion into: angry, happy, neutral, sad.\n"
)
OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"


def build_prompt(text, exemplars=None):
    sys_p = SYS_BASE
    if exemplars:
        sys_p += f"\nHere are {len(exemplars)} labeled examples:\n"
        for i, (t, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  Utterance: \"{t}\"\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += OUTPUT_FMT
    user_p = f"Utterance: \"{text}\"\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]


def prepare_data_seed(df, seed):
    df = df.copy()
    df["emotion_en"] = df["label_name"].map(LABEL_MAP)
    df = df[df["emotion_en"].isin(LABELS)].copy()

    rng = np.random.default_rng(seed)

    # Test set: stratified 100 per class
    test_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        n = min(N_TEST_PER_CLASS, len(sub))
        pick = sub.iloc[rng.choice(len(sub), size=n, replace=False)].index
        test_idx.extend(pick.tolist())
    test_df = df.loc[test_idx].copy()
    remaining = df.drop(index=test_idx)

    # Exemplar pool: 1 per class from remaining
    exemplar_idx = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        exemplar_idx.append(sub.iloc[rng.integers(0, len(sub))].name)
    ex_df = remaining.loc[exemplar_idx].copy()
    exemplars = [(r["transcription"], r["emotion_en"]) for _, r in ex_df.iterrows()]

    return test_df, exemplars


def run(test_df, exemplars, model, tokenizer):
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        text = row["transcription"]
        if pd.isna(text) or not text:
            preds.append("neutral")
            continue
        msgs = build_prompt(text, exemplars)
        ps = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        label, _ = parse_output(raw)
        preds.append(label)
        if (i+1) % 100 == 0:
            el = time.time() - t0
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s")
    return preds


df = pd.read_parquet(IEMOCAP_PARQUET)
print(f"[info] IEMOCAP N={len(df)}, label_name dist: {Counter(df['label_name'].tolist())}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

per_seed = {}
per_tier = {"T1_zero_shot": {"acc": [], "f1": []}, "T2_icl_k4": {"acc": [], "f1": []}}

for seed in SEEDS:
    print(f"\n###### SEED {seed} ######")
    test_df, exemplars = prepare_data_seed(df, seed)
    print(f"[info] test N={len(test_df)} dist: {Counter(test_df['emotion_en'].tolist())}")
    print(f"[info] exemplars:")
    for t, l in exemplars:
        print(f"   {l}: {t[:90]}")
    y_true = test_df["emotion_en"].tolist()
    seed_res = {}

    for tier_name, ex in [("T1_zero_shot", None), ("T2_icl_k4", exemplars)]:
        print(f"\n=== seed={seed} {tier_name} ===")
        preds = run(test_df, ex, model, tokenizer)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
        print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
        seed_res[tier_name] = {"acc": acc, "f1": f1}
        per_tier[tier_name]["acc"].append(acc)
        per_tier[tier_name]["f1"].append(f1)

    per_seed[seed] = seed_res

aggregate = {}
for name, d in per_tier.items():
    accs = np.array(d["acc"]); f1s = np.array(d["f1"])
    aggregate[name] = {
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)),
        "accs": accs.tolist(), "f1s": f1s.tolist(),
    }

with open(OUT / "iemocap_tiers_results.json", "w") as f:
    json.dump({
        "task": "IEMOCAP 3-tier cross-domain replication (T1+T2 only, T3 LoRA tomorrow)",
        "seeds": SEEDS, "N_per_seed": 400,
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "aggregate": aggregate,
        "reference_korean_FER_AU": {
            "T1_zero_shot_multiseed": 0.2908,
            "T2_icl_k4_multiseed":    0.4183,
            "T3_lora_multiseed":      0.5500,
        },
    }, f, indent=2, ensure_ascii=False)

print("\n[DONE]")
for name, a in aggregate.items():
    print(f"  {name}: acc={a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f}%  f1={a['f1_mean']:.3f}±{a['f1_std']:.3f}")
print(f"[saved] {OUT}/iemocap_tiers_results.json")
