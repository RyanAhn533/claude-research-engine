"""
LoRA fine-tune Qwen2.5-7B-Instruct on Korean FER AU (4-class).

Train: 10K balanced samples from Yonsei-clean pool.
Test: 400 stratified (seed=42, disjoint from train) — matches exp_012 so direct delta is readable.
Config: QLoRA (4-bit base + LoRA r=16 alpha=32), 1 epoch, bs=4, grad_accum=4, lr=2e-4.

Expected improvement over exp_012 fewshot_k4 (~42%): +10-15pp → target ~55%.
"""
import os, sys, json, glob, time, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import parse_output

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)
ADAPTER_DIR = OUT / "adapter"
ADAPTER_DIR.mkdir(exist_ok=True, parents=True)

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

N_TRAIN_PER_CLASS = 2500       # 10K total balanced
N_TEST_PER_CLASS = 100         # 400 total (matches exp_012)
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
MAX_LENGTH = 512


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


SYS_P = (
    "You are a facial emotion classifier using FACS action units (AUs). "
    "Given detected AU intensities (0-100) of a Korean subject, classify into: "
    "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23.\n"
    "Output format:\nLABEL: <label>\nREASON: <one sentence>"
)


def make_messages(au_text, answer=None):
    msgs = [
        {"role": "system", "content": SYS_P},
        {"role": "user", "content": f"AUs: {au_text}\n\nClassify emotion."},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def build_train_answer(label):
    return f"LABEL: {label}"


def prepare_splits():
    df = pd.read_parquet(AU_PARQUET)
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    print(f"[data] clean pool size: {len(df)}")

    rng = np.random.default_rng(SEED)

    test_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pick = sub.iloc[rng.choice(len(sub), size=N_TEST_PER_CLASS, replace=False)].index
        test_idx.extend(pick.tolist())
    test_df = df.loc[test_idx].copy()
    remaining = df.drop(index=test_idx)

    train_idx = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        n = min(N_TRAIN_PER_CLASS, len(sub))
        pick = sub.iloc[rng.choice(len(sub), size=n, replace=False)].index
        train_idx.extend(pick.tolist())
    train_df = remaining.loc[train_idx].copy().sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"[data] train={len(train_df)}  test={len(test_df)}")
    print(f"[data] train dist: {Counter(train_df['emotion_en'].tolist())}")
    print(f"[data] test  dist: {Counter(test_df['emotion_en'].tolist())}")
    return train_df, test_df


def tokenize_for_train(tokenizer, train_df):
    ids_list, attn_list, label_list = [], [], []
    for _, r in train_df.iterrows():
        au_text = au_to_text(r)
        answer = build_train_answer(r["emotion_en"])
        prompt_msgs = make_messages(au_text)
        full_msgs = make_messages(au_text, answer=answer)

        prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)

        if len(full_ids) > MAX_LENGTH:
            full_ids = full_ids[:MAX_LENGTH]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_prompt + list(full_ids[n_prompt:])
        if len(labels) < len(full_ids):
            labels += [-100] * (len(full_ids) - len(labels))
        labels = labels[:len(full_ids)]

        ids_list.append(full_ids)
        attn_list.append([1] * len(full_ids))
        label_list.append(labels)
    return Dataset.from_dict({"input_ids": ids_list, "attention_mask": attn_list, "labels": label_list})


def eval_on_test(model, tokenizer, test_df, tag=""):
    model.eval()
    preds = []
    y_true = test_df["emotion_en"].tolist()
    t0 = time.time()
    for i, row in test_df.iterrows():
        au_text = au_to_text(row)
        msgs = make_messages(au_text)
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
            print(f"    [eval {tag}] {i+1}/{len(test_df)} {el:.0f}s")
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
    print(f"[eval {tag}] acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    return acc, f1, preds


# ===== Main =====
train_df, test_df = prepare_splits()

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
tokenizer = AutoTokenizer.from_pretrained(snap)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    snap, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
print(f"[vram after LoRA attach] {torch.cuda.memory_allocated()/1024**2:.0f} MB")

# ===== Tokenize + train =====
print("[train] tokenizing...")
train_ds = tokenize_for_train(tokenizer, train_df)
print(f"[train] train_ds size={len(train_ds)}")

training_args = TrainingArguments(
    output_dir=str(OUT / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="no",
    bf16=False, fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, pad_to_multiple_of=8, padding=True, label_pad_token_id=-100,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)
trainer.train()

# Save adapter
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"[saved adapter] {ADAPTER_DIR}")

# ===== Eval after fine-tune =====
model.config.use_cache = True
acc_ft, f1_ft, preds_ft = eval_on_test(model, tokenizer, test_df, tag="finetuned")

with open(OUT / "lora_results.json", "w") as f:
    json.dump({
        "task": "LoRA fine-tune Qwen2.5-7B Korean AU 4-class",
        "config": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                   "epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                   "lr": LR, "n_train": len(train_df), "n_test": len(test_df), "seed": SEED},
        "finetuned": {"acc": acc_ft, "f1": f1_ft},
        "reference_fewshot_k4_seed42": 0.455,
        "reference_multiseed_mean": 0.4183,
    }, f, indent=2)
print(f"[DONE] acc={acc_ft*100:.2f}%  f1={f1_ft:.3f}  (ref fewshot multi-seed: 41.83±3.41%)")
print(f"[saved] {OUT}/lora_results.json")
