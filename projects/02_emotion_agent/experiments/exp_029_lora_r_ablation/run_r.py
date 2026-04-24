"""
exp_029: LoRA rank ablation on Korean FER AU.
Usage: python run_r.py <r_value>  (16 = baseline from exp_015 seed 42)

Tests whether our 55% LoRA ceiling is rank-limited.
r ∈ {8, 32} — one run each (seed 42), quick follow-up.
"""
import os, sys, json, glob, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import parse_output

LORA_R = int(sys.argv[1])
LORA_ALPHA = LORA_R * 2
SEED = 42

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path(__file__).parent / "cache" / f"r_{LORA_R}"
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

N_TRAIN_PER_CLASS = 2500
N_TEST_PER_CLASS = 100
EPOCHS = 1; BATCH_SIZE = 4; GRAD_ACCUM = 4; LR = 2e-4; MAX_LENGTH = 512

SYS_P = (
    "You are a facial emotion classifier using FACS action units (AUs). "
    "Given detected AU intensities (0-100) of a Korean subject, classify into: "
    "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23.\n"
    "Output format:\nLABEL: <label>\nREASON: <one sentence>"
)


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


def make_msgs(text, ans=None):
    m = [{"role":"system","content":SYS_P},
         {"role":"user","content":f"AUs: {text}\n\nClassify emotion."}]
    if ans is not None:
        m.append({"role":"assistant","content":ans})
    return m


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

    train_idx = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        n = min(N_TRAIN_PER_CLASS, len(sub))
        pick = sub.iloc[rng.choice(len(sub), size=n, replace=False)].index
        train_idx.extend(pick.tolist())
    train_df = remaining.loc[train_idx].copy().sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return train_df, test_df


def tokenize_train(tok, tdf):
    ids, attn, labs = [], [], []
    for _, r in tdf.iterrows():
        text = au_to_text(r)
        ans = f"LABEL: {r['emotion_en']}"
        prompt_ids = tok.apply_chat_template(make_msgs(text), tokenize=True, add_generation_prompt=True)
        full_ids = tok.apply_chat_template(make_msgs(text, ans), tokenize=True, add_generation_prompt=False)
        if len(full_ids) > MAX_LENGTH: full_ids = full_ids[:MAX_LENGTH]
        n_p = min(len(prompt_ids), len(full_ids))
        l = [-100]*n_p + list(full_ids[n_p:])
        if len(l) < len(full_ids): l += [-100]*(len(full_ids)-len(l))
        l = l[:len(full_ids)]
        ids.append(full_ids); attn.append([1]*len(full_ids)); labs.append(l)
    return Dataset.from_dict({"input_ids":ids, "attention_mask":attn, "labels":labs})


def eval_test(model, tok, tdf):
    model.eval(); preds = []
    y_true = tdf["emotion_en"].tolist(); t0 = time.time()
    for i, r in tdf.iterrows():
        text = au_to_text(r)
        ps = tok.apply_chat_template(make_msgs(text), tokenize=False, add_generation_prompt=True)
        inp = tok(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)
        lbl, _ = parse_output(raw); preds.append(lbl)
        if (i+1) % 100 == 0:
            print(f"    [eval r={LORA_R}] {i+1}/{len(tdf)} {time.time()-t0:.0f}s")
    return float(accuracy_score(y_true, preds)), float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))


train_df, test_df = prepare_splits()
print(f"[data r={LORA_R}] train={len(train_df)} test={len(test_df)}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(snap)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                             device_map={"":0}, torch_dtype=torch.float16)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

train_ds = tokenize_train(tok, train_df)
args = TrainingArguments(output_dir=str(OUT/"checkpoints"), num_train_epochs=EPOCHS,
                         per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                         learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.03,
                         logging_steps=50, save_strategy="no", bf16=False, fp16=True,
                         optim="paged_adamw_8bit", report_to="none", remove_unused_columns=False,
                         gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
                         seed=SEED, data_seed=SEED)
coll = DataCollatorForSeq2Seq(tokenizer=tok, pad_to_multiple_of=8, padding=True, label_pad_token_id=-100)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=coll)
trainer.train()

model.config.use_cache = True
acc, f1 = eval_test(model, tok, test_df)
print(f"[DONE r={LORA_R}] acc={acc*100:.2f}%  f1={f1:.3f}")

with open(OUT / "result.json", "w") as f:
    json.dump({"r":LORA_R,"alpha":LORA_ALPHA,"seed":SEED,
               "epochs":EPOCHS,"bs":BATCH_SIZE,"grad_accum":GRAD_ACCUM,"lr":LR,
               "finetuned":{"acc":acc,"f1":f1}}, f, indent=2)
print(f"[saved] {OUT}/result.json")
