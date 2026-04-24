"""
exp_020: Class-coverage control.

Dissociates: is anchor-variance effect driven by CLASS COVERAGE or ANCHOR ORIGIN?

exp_014 context:
  D_mixed (2 Kor + 2 Wes): 41.25 ± 0.87  — 3-class coverage (angry/happy/sad, no neutral)
  E_kor4  (4 Kor, 1 each):  41.00 ± 4.34  — 4-class coverage

Hypotheses:
  H_COVERAGE: 3-class coverage → low σ regardless of anchor origin.
  H_ORIGIN:    Western anchor → low σ regardless of class coverage.

New experiment (fixed k=4):
  H1_kor4_3class:    Korean only, 3-class coverage
                     [angry, happy, sad, sad_dup] (2nd Korean for sad)
                     → if H_COVERAGE correct: σ small (like D)
                     → if H_ORIGIN correct:   σ large (like E)

  H2_kor4_4class (replicate E for grid):  [angry, happy, neutral, sad] — sanity match

3 seeds × 2 configs × 400 samples ≈ 40 min.
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
    "angry, happy, neutral, sad.\n"
)
OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"


def build_prompt(target_au_text, exemplars, origin_label=None):
    sys_p = SYS_BASE
    if exemplars:
        if origin_label:
            sys_p += f"\nHere are {len(exemplars)} examples ({origin_label}) with correct labels:\n"
        else:
            sys_p += f"\nHere are {len(exemplars)} examples with correct labels:\n"
        for i, (au_text, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  AUs: {au_text}\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += OUTPUT_FMT
    user_p = f"Target AUs:\n{target_au_text}\n\nClassify emotion."
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]


def prepare_data_seed(df, seed):
    # Per class 2 Korean samples so we can duplicate for 3-class coverage H1
    pool_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pool_idx.extend(sub.sample(n=2, random_state=seed).index.tolist())
    pool = df.loc[pool_idx].copy().reset_index(drop=True)
    pool["au_text"] = pool.apply(au_to_text, axis=1)
    remaining = df.drop(index=pool_idx)

    rng = np.random.default_rng(seed)
    pieces = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        sub = sub.iloc[rng.choice(len(sub), size=min(100, len(sub)), replace=False)]
        pieces.append(sub)
    test = pd.concat(pieces).reset_index(drop=True)

    # Organize by label
    by_label = {}
    for _, r in pool.iterrows():
        by_label.setdefault(r["emotion_en"], []).append((r["au_text"], r["emotion_en"]))
    # by_label[l] = [sample1, sample2] for each l ∈ LABELS

    # H1: 3-class coverage (drop neutral, duplicate sad from 2 different Korean samples)
    H1 = [by_label["angry"][0], by_label["happy"][0], by_label["sad"][0], by_label["sad"][1]]

    # H2: 4-class coverage (replicate exp_014 E structure with 1 per class, using first sample each)
    H2 = [by_label[l][0] for l in LABELS]

    return test, {
        "H1_kor4_3class": H1,
        "H2_kor4_4class": H2,
    }


def run(test_df, exemplars, origin_label, model, tokenizer):
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        target = au_to_text(row)
        messages = build_prompt(target, exemplars, origin_label)
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
            print(f"    [{i+1}/{len(test_df)}] {el:.1f}s")
    return preds


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

CONFIG_NAMES = ["H1_kor4_3class", "H2_kor4_4class"]
per_seed = {}
per_config = {n: {"acc": [], "f1": []} for n in CONFIG_NAMES}

for seed in SEEDS:
    print(f"\n###### SEED {seed} ######")
    test_df, ex_packs = prepare_data_seed(df, seed)
    print(f"[info] test N={len(test_df)}  dist: {Counter(test_df['emotion_en'].tolist())}")
    y_true = test_df["emotion_en"].tolist()

    for name in CONFIG_NAMES:
        ex = ex_packs[name]
        origin = "Korean subjects"
        print(f"\n=== seed={seed} {name} (k={len(ex)}) ===")
        for t, l in ex:
            print(f"   {l}: {t[:80]}")
        preds = run(test_df, ex, origin, model, tokenizer)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
        print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
        per_seed.setdefault(seed, {})[name] = {"acc": acc, "f1": f1}
        per_config[name]["acc"].append(acc)
        per_config[name]["f1"].append(f1)

aggregate = {}
for name in CONFIG_NAMES:
    accs = np.array(per_config[name]["acc"]); f1s = np.array(per_config[name]["f1"])
    aggregate[name] = {
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)),
        "accs": accs.tolist(), "f1s": f1s.tolist(),
    }

with open(OUT / "class_coverage_results.json", "w") as f:
    json.dump({
        "task": "Class-coverage dissociation (3-class Korean-only vs 4-class Korean-only)",
        "seeds": SEEDS,
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "aggregate": aggregate,
        "reference_exp_014": {
            "D_mixed_2+2_3class": {"acc_mean": 0.4125, "acc_std": 0.0087},
            "E_kor4_4class":      {"acc_mean": 0.4100, "acc_std": 0.0434},
        },
        "interpretation_guide": {
            "H1_low_std_~0.87":  "class coverage is key (H_COVERAGE)",
            "H1_high_std_~4.34": "anchor origin is key (H_ORIGIN)",
        },
    }, f, indent=2, ensure_ascii=False)

print("\n[DONE]")
for name, a in aggregate.items():
    print(f"  {name}: acc={a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f}%  f1={a['f1_mean']:.3f}±{a['f1_std']:.3f}")
print(f"[saved] {OUT}/class_coverage_results.json")
