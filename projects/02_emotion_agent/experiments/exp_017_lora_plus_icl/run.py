"""
Test if ICL exemplars help ON TOP of LoRA fine-tuning.
Loads LoRA adapter from exp_015 (seed=42, single-seed best 56.75%),
evaluates same 400-sample test with and without in-context exemplars.

Configs:
  A: No exemplars (replicates exp_015 single-seed = 56.75% baseline)
  B: Korean k=2 Mixed (2+2) — matches exp_014 D config (41.25 at prompt-only level)
  C: Korean k=4 only (matches exp_014 E at prompt-only level)

Hypothesis: If LoRA embeds Korean distribution internally, ICL should
be redundant (no further gain). If ICL provides complementary signal
(e.g., test-sample-specific context), we'd see compound effect = 4-tier.
"""
import os, sys, json, glob, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import parse_output

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
ADAPTER_DIR = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_015_lora_korean_au/cache/adapter")
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

SEED = 42
N_TEST_PER_CLASS = 100


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


def format_au_dict(au_dict):
    parts = []
    for au, v in au_dict.items():
        if au in AU_DESC:
            parts.append(f"{au} ({AU_DESC[au]})={v}")
    return ", ".join(parts)


WESTERN_EKMAN = [
    (format_au_dict({"AU6": 80, "AU12": 85, "AU25": 60}), "happy"),
    (format_au_dict({"AU1": 70, "AU4": 65, "AU15": 75, "AU17": 55}), "sad"),
    (format_au_dict({"AU4": 80, "AU5": 65, "AU7": 78, "AU23": 72, "AU17": 60}), "angry"),
    (format_au_dict({"AU6": 10, "AU12": 15, "AU4": 12, "AU25": 20}), "neutral"),
]


# LoRA-trained system prompt (must match exp_015 training format for adapter use)
SYS_P_PLAIN = (
    "You are a facial emotion classifier using FACS action units (AUs). "
    "Given detected AU intensities (0-100) of a Korean subject, classify into: "
    "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23.\n"
    "Output format:\nLABEL: <label>\nREASON: <one sentence>"
)


def build_sys_with_exemplars(exemplars, origin=None):
    sys_p = (
        "You are a facial emotion classifier using FACS action units (AUs). "
        "Given detected AU intensities (0-100) of a Korean subject, classify into: "
        "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23.\n"
    )
    if exemplars:
        if origin:
            sys_p += f"\nHere are {len(exemplars)} examples ({origin}) with correct labels:\n"
        else:
            sys_p += f"\nHere are {len(exemplars)} examples with correct labels:\n"
        for i, (au_text, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  AUs: {au_text}\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
    return sys_p


def make_messages(au_text, exemplars=None, origin=None):
    sys_p = build_sys_with_exemplars(exemplars, origin) if exemplars else SYS_P_PLAIN
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": f"AUs: {au_text}\n\nClassify emotion."},
    ]


def prepare_splits():
    df = pd.read_parquet(AU_PARQUET)
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()

    rng = np.random.default_rng(SEED)
    test_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pick = sub.iloc[rng.choice(len(sub), size=N_TEST_PER_CLASS, replace=False)].index
        test_idx.extend(pick.tolist())
    test_df = df.loc[test_idx].copy()
    remaining = df.drop(index=test_idx)

    # Exemplar pool — one Korean per class from REMAINING (disjoint from test).
    # Uses random_state=SEED; consistent-ish with exp_014's pattern.
    pool_idx = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        pool_idx.extend(sub.sample(n=1, random_state=SEED).index.tolist())
    pool = remaining.loc[pool_idx].copy().reset_index(drop=True)
    pool["au_text"] = pool.apply(au_to_text, axis=1)

    # Korean exemplars in LABELS order: [angry, happy, neutral, sad]
    kor_all = [(r["au_text"], r["emotion_en"]) for _, r in pool.iterrows()]
    # exp_014 D config: Kor[angry, happy] + Wes[happy, sad]
    mixed_k4 = kor_all[:2] + WESTERN_EKMAN[:2]
    return test_df, {"korean_k4": kor_all, "mixed_k4_2+2": mixed_k4}


def run(test_df, exemplars, origin, model, tokenizer):
    model.eval()
    preds = []
    y_true = test_df["emotion_en"].tolist()
    t0 = time.time()
    for i, row in test_df.iterrows():
        au_text = au_to_text(row)
        msgs = make_messages(au_text, exemplars=exemplars, origin=origin)
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
            print(f"    {i+1}/{len(test_df)} {el:.0f}s")
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
    return acc, f1


test_df, ex_packs = prepare_splits()
print(f"[info] test N={len(test_df)} dist: {Counter(test_df['emotion_en'].tolist())}")
print(f"[info] exemplar packs: {list(ex_packs.keys())}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
tokenizer = AutoTokenizer.from_pretrained(snap)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
print(f"[info] adapter loaded from {ADAPTER_DIR}")
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

results = {}
configs = [
    ("A_lora_only",        None,                  None),
    ("B_lora_mixed_k4_2+2", ex_packs["mixed_k4_2+2"], "Korean + Western mix"),
    ("C_lora_korean_k4",    ex_packs["korean_k4"],    "Korean subjects"),
]

for name, ex, origin in configs:
    print(f"\n=== {name} (k={len(ex) if ex else 0}) ===")
    acc, f1 = run(test_df, ex, origin, model, tokenizer)
    print(f"  acc={acc*100:.2f}%  F1={f1:.3f}")
    results[name] = {"acc": acc, "f1": f1, "k": len(ex) if ex else 0}

with open(OUT / "lora_plus_icl_results.json", "w") as f:
    json.dump({
        "task": "LoRA + ICL combo test (seed=42, N=400)",
        "adapter": str(ADAPTER_DIR),
        "configs": results,
        "reference": {
            "exp_015_seed42_no_icl": 0.5675,
            "exp_014_D_mixed_no_lora_multiseed": 0.4125,
            "exp_014_E_korean_no_lora_multiseed": 0.4100,
        },
    }, f, indent=2, ensure_ascii=False)

print("\n[DONE]")
for name, r in results.items():
    print(f"  {name}: acc={r['acc']*100:.2f}%  F1={r['f1']:.3f}")
print(f"[saved] {OUT}/lora_plus_icl_results.json")
