"""
exp_032: Prompt ablation on IEMOCAP + MELD text — cross-domain confirmation.

Tests whether IEMOCAP/MELD T1 zero-shot is also sub-optimal baseline.
3 prompts × 3 seeds × 2 datasets × 400 samples ≈ 90min.
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
sys.path.insert(0, "/home/ajy/claude-research-engine/projects/02_emotion_agent/src")
from agent.emotion_agent import parse_output

IEMOCAP = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet")
MELD = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_002_meld_preproc/cache/meld_4class.parquet")
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry","happy","neutral","sad"]
IEM_MAP = {"ang":"angry","hap":"happy","exc":"happy","neu":"neutral","sad":"sad"}
MELD_MAP = {"ang":"angry","hap":"happy","neu":"neutral","sad":"sad"}
SEEDS = [42, 123, 777]
N_TEST_PER_CLASS = 100

OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
PROMPTS = {
    "P1_baseline": "You are an emotion classifier for spoken English utterances. Given a transcription, classify the speaker's emotion into: angry, happy, neutral, sad.",
    "P2_minimal": "Classify the emotion. Options: angry, happy, neutral, sad.",
    "P3_rich": "You are an emotion classifier for English dialog. Analyze the utterance for tone, word choice, and emotional content. Consider context and speaker intent. Classify into: angry (frustrated/aggressive), happy (joyful/positive), neutral (calm/factual), sad (downcast/melancholic).",
}


def build_prompt(text, sys_base):
    sys_p = sys_base + OUTPUT_FMT
    return [{"role":"system","content":sys_p},
            {"role":"user","content":f"Utterance: \"{text}\"\n\nClassify emotion."}]


def prepare_test(df, seed, label_col, text_col, label_map):
    df = df.copy()
    df["emotion_en"] = df[label_col].map(label_map)
    df = df[df["emotion_en"].isin(LABELS) & df[text_col].notna() & (df[text_col].str.len() > 0)].copy()
    rng = np.random.default_rng(seed)
    pieces = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        n = min(N_TEST_PER_CLASS, len(sub))
        pieces.append(sub.iloc[rng.choice(len(sub), size=n, replace=False)])
    return pd.concat(pieces).reset_index(drop=True)


def run(test_df, sys_base, model, tok, text_col, tag):
    preds = []; t0 = time.time()
    y_true = test_df["emotion_en"].tolist()
    for i, row in test_df.iterrows():
        text = row[text_col]
        if pd.isna(text) or not text:
            preds.append("neutral"); continue
        msgs = build_prompt(text, sys_base)
        ps = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)
        lbl, _ = parse_output(raw); preds.append(lbl)
        if (i+1) % 200 == 0:
            print(f"  [{tag}] {i+1}/{len(test_df)} {time.time()-t0:.0f}s")
    return (float(accuracy_score(y_true, preds)),
            float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0)))


snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                             device_map={"":0}, torch_dtype=torch.float16)
model.eval()
print(f"[info] VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

datasets = {
    "IEMOCAP": (pd.read_parquet(IEMOCAP), "label_name", "transcription", IEM_MAP),
    "MELD": (pd.read_parquet(MELD), "label_4_name", "text", MELD_MAP),
}

all_results = {}
for ds_name, (df, lc, tc, lm) in datasets.items():
    print(f"\n########## DATASET: {ds_name} ##########")
    per_seed = {}
    per_prompt = {k: {"acc":[], "f1":[]} for k in PROMPTS}
    for seed in SEEDS:
        print(f"\n###### SEED {seed} ######")
        test_df = prepare_test(df, seed, lc, tc, lm)
        seed_res = {}
        for name, sys_p in PROMPTS.items():
            tag = f"{ds_name}_{name}_s{seed}"
            print(f"\n=== {tag} ===")
            acc, f1 = run(test_df, sys_p, model, tok, tc, tag)
            print(f"  acc={acc*100:.2f}%  F1={f1:.3f}")
            seed_res[name] = {"acc":acc, "f1":f1}
            per_prompt[name]["acc"].append(acc); per_prompt[name]["f1"].append(f1)
        per_seed[seed] = seed_res
    agg = {}
    for name in PROMPTS:
        accs = np.array(per_prompt[name]["acc"]); f1s = np.array(per_prompt[name]["f1"])
        agg[name] = {"acc_mean":float(accs.mean()), "acc_std":float(accs.std(ddof=1)),
                     "f1_mean":float(f1s.mean()), "f1_std":float(f1s.std(ddof=1))}
    all_results[ds_name] = {"per_seed":{str(s):per_seed[s] for s in SEEDS}, "aggregate":agg}

with open(OUT / "text_prompt_ablation_results.json", "w") as f:
    json.dump({"task":"IEMOCAP+MELD prompt ablation 3 seeds","seeds":SEEDS,"prompts":{k:v[:200] for k,v in PROMPTS.items()},
               "results":all_results}, f, indent=2)

print("\n[DONE]")
for ds, dat in all_results.items():
    print(f"\n--- {ds} ---")
    for name, a in dat["aggregate"].items():
        print(f"  {name}: acc={a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f}%  F1={a['f1_mean']:.3f}±{a['f1_std']:.3f}")
