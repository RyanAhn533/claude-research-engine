"""
Multi-seed k-shot sweep. SEEDS=[42,123,777] × k∈{0,2,4,8,16}.
Tests stability of inverted-U finding.
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

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry", "happy", "neutral", "sad"]
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

SEEDS = [42, 123, 777]
K_LIST = [0, 2, 4, 8, 16]


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


SYS_BASE = (
    "You are a facial emotion classifier using FACS action units (AUs). "
    "Given detected AU intensities (0-100) of a Korean subject, classify into: "
    "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23.\n"
)
OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"


def build_prompt(target_au_text, exemplars):
    sys_p = SYS_BASE
    if exemplars:
        sys_p += f"\nHere are {len(exemplars)} Korean examples with correct labels:\n"
        for i, (au_text, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  AUs: {au_text}\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += OUTPUT_FMT
    user_p = f"Target AUs:\n{target_au_text}\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]


def prepare_data_seed(df, seed):
    """Pool=16/class, test=100/class, both seed-dependent (disjoint)."""
    pool_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pool_idx.extend(sub.sample(n=4, random_state=seed).index.tolist())
    pool = df.loc[pool_idx].copy().reset_index(drop=True)
    remaining = df.drop(index=pool_idx)

    rng = np.random.default_rng(seed)
    pieces = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        sub = sub.iloc[rng.choice(len(sub), size=min(100, len(sub)), replace=False)]
        pieces.append(sub)
    test = pd.concat(pieces).reset_index(drop=True)

    def exemplars_for_k(k):
        if k == 0:
            return []
        per_class = max(1, k // len(LABELS))
        out = []
        for lab in LABELS:
            sub = pool[pool["emotion_en"] == lab].head(per_class)
            for _, r in sub.iterrows():
                out.append((au_to_text(r), r["emotion_en"]))
        return out[:k]

    return test, exemplars_for_k


def run(test_df, exemplars, model, tokenizer):
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        target = au_to_text(row)
        messages = build_prompt(target, exemplars)
        ps = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        label, _ = parse_output(raw)
        preds.append(label)
        if (i+1) % 200 == 0:
            el = time.time() - t0
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s {el/(i+1)*1000:.0f}ms/sample")
    return preds


# ===== Main =====
df = pd.read_parquet(AU_PARQUET)
df = df[df["is_selected"] == 0].copy()
df["emotion_en"] = df["emotion"].map(KR_TO_EN)
df = df[df["emotion_en"].notna()].copy()

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
per_k = {k: {"acc": [], "f1": []} for k in K_LIST}

for seed in SEEDS:
    print(f"\n###### SEED {seed} ######")
    test_df, exemplars_for_k = prepare_data_seed(df, seed)
    print(f"[info] test N={len(test_df)}  dist: {Counter(test_df['emotion_en'].tolist())}")
    y_true = test_df["emotion_en"].tolist()
    seed_res = {}
    for k in K_LIST:
        ex = exemplars_for_k(k)
        print(f"\n=== seed={seed} k={k} ({len(ex)} exemplars) ===")
        preds = run(test_df, ex, model, tokenizer)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
        print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
        seed_res[f"k={k}"] = {"acc": acc, "f1": f1}
        per_k[k]["acc"].append(acc)
        per_k[k]["f1"].append(f1)
    per_seed[seed] = seed_res

aggregate = {}
for k in K_LIST:
    accs = np.array(per_k[k]["acc"]); f1s = np.array(per_k[k]["f1"])
    aggregate[f"k={k}"] = {
        "k": k,
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)),
        "accs": accs.tolist(), "f1s": f1s.tolist(),
    }

with open(OUT / "kshot_sweep_multiseed_results.json", "w") as f:
    json.dump({
        "task": "k-shot sweep, multi-seed [42,123,777]",
        "seeds": SEEDS, "K_list": K_LIST,
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "aggregate": aggregate,
    }, f, indent=2)

print("\n[DONE]")
for k, a in aggregate.items():
    print(f"  {k}: acc={a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f}%  f1={a['f1_mean']:.3f}±{a['f1_std']:.3f}")
print(f"[saved] {OUT}/kshot_sweep_multiseed_results.json")
