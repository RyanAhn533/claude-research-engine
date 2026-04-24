"""
exp_023: IEMOCAP k-shot sweep (k=0,4,8). Cross-domain saturation check.
Mirrors exp_013 Korean FER k-sweep but on English dialog text.
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
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)

LABEL_MAP = {"ang":"angry","hap":"happy","exc":"happy","neu":"neutral","sad":"sad"}
LABELS = ["angry","happy","neutral","sad"]
SEEDS = [42, 123, 777]
K_LIST = [0, 4, 8]
N_TEST_PER_CLASS = 100

SYS_BASE = (
    "You are an emotion classifier for spoken English utterances. "
    "Given a transcription, classify the speaker's emotion into: angry, happy, neutral, sad.\n"
)
OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"


def build_prompt(text, exemplars):
    sys_p = SYS_BASE
    if exemplars:
        sys_p += f"\nHere are {len(exemplars)} labeled examples:\n"
        for i, (t, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  Utterance: \"{t}\"\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += OUTPUT_FMT
    return [{"role":"system","content":sys_p},
            {"role":"user","content":f"Utterance: \"{text}\"\n\nClassify emotion."}]


def prepare_data_seed(df, seed):
    df = df.copy()
    df["emotion_en"] = df["label_name"].map(LABEL_MAP)
    df = df[df["emotion_en"].isin(LABELS)].copy()
    rng = np.random.default_rng(seed)

    pool_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pool_idx.extend(sub.sample(n=4, random_state=seed).index.tolist())
    pool = df.loc[pool_idx].copy()
    remaining = df.drop(index=pool_idx)

    test_idx = []
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        n = min(N_TEST_PER_CLASS, len(sub))
        pick = sub.iloc[rng.choice(len(sub), size=n, replace=False)].index
        test_idx.extend(pick.tolist())
    test_df = remaining.loc[test_idx].copy()

    def exemplars_for_k(k):
        if k == 0: return []
        per_class = max(1, k // len(LABELS))
        out = []
        for lab in LABELS:
            sub = pool[pool["emotion_en"] == lab].head(per_class)
            for _, r in sub.iterrows():
                out.append((r["transcription"], r["emotion_en"]))
        return out[:k]

    return test_df, exemplars_for_k


def run(test_df, exemplars, model, tokenizer):
    preds = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        text = row["transcription"]
        if pd.isna(text) or not text:
            preds.append("neutral"); continue
        msgs = build_prompt(text, exemplars)
        ps = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        label, _ = parse_output(raw)
        preds.append(label)
        if (i+1) % 200 == 0:
            print(f"    [{i+1}/{len(test_df)}] {time.time()-t0:.1f}s")
    return preds


df = pd.read_parquet(IEMOCAP_PARQUET)
snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                             device_map={"":0}, torch_dtype=torch.float16)
model.eval()

per_seed = {}
per_k = {k: {"acc":[], "f1":[]} for k in K_LIST}

for seed in SEEDS:
    print(f"\n###### SEED {seed} ######")
    test_df, exemplars_for_k = prepare_data_seed(df, seed)
    y_true = test_df["emotion_en"].tolist()
    seed_res = {}
    for k in K_LIST:
        ex = exemplars_for_k(k)
        print(f"\n=== seed={seed} k={k} ({len(ex)} exemplars) ===")
        preds = run(test_df, ex, model, tokenizer)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
        print(f"  acc={acc*100:.2f}%  F1={f1:.3f}")
        seed_res[f"k={k}"] = {"acc":acc, "f1":f1}
        per_k[k]["acc"].append(acc); per_k[k]["f1"].append(f1)
    per_seed[seed] = seed_res

agg = {}
for k in K_LIST:
    accs = np.array(per_k[k]["acc"]); f1s = np.array(per_k[k]["f1"])
    agg[f"k={k}"] = {"k":k, "acc_mean":float(accs.mean()), "acc_std":float(accs.std(ddof=1)),
                     "f1_mean":float(f1s.mean()), "f1_std":float(f1s.std(ddof=1)),
                     "accs":accs.tolist(), "f1s":f1s.tolist()}

with open(OUT / "iemocap_kshot_results.json", "w") as f:
    json.dump({"task":"IEMOCAP k-shot sweep (cross-domain)", "seeds":SEEDS, "K_list":K_LIST,
               "per_seed":{str(s):per_seed[s] for s in SEEDS}, "aggregate":agg}, f, indent=2)

print("\n[DONE]")
for k, a in agg.items():
    print(f"  {k}: acc={a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f}%  f1={a['f1_mean']:.3f}±{a['f1_std']:.3f}")
