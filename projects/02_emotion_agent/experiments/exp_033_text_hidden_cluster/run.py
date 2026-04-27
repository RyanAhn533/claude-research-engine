"""
exp_033: Hidden-cluster analysis on IEMOCAP text (cross-domain mechanism check).

Mirrors exp_028 (Korean FER AU) on IEMOCAP transcription.
Compares T1 (best zero-shot prompt) vs T2 (ICL k=4) hidden representations.

Hypothesis: since IEMOCAP ICL gain is essentially nil (+1.42pp), the
hidden-rep compression mechanism observed on Korean FER (D 22% compact)
should also be absent on IEMOCAP. Confirms that compression mechanism
correlates with output-variance benefit.
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE

IEMOCAP = Path("/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet")
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)

LABEL_MAP = {"ang":"angry","hap":"happy","exc":"happy","neu":"neutral","sad":"sad"}
LABELS = ["angry","happy","neutral","sad"]
SEED = 42
N_SAMPLES = 80

OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
SYS_BASE = ("You are an emotion classifier for spoken English utterances. "
            "Given a transcription, classify the speaker's emotion into: angry, happy, neutral, sad.")


def build_prompt(text, exemplars=None):
    sys_p = SYS_BASE
    if exemplars:
        sys_p += f"\nHere are {len(exemplars)} labeled examples:\n"
        for i, (t, lab) in enumerate(exemplars):
            sys_p += f"\nExample {i+1}:\n  Utterance: \"{t}\"\n  LABEL: {lab}\n"
        sys_p += "\nNow classify the target."
    sys_p += OUTPUT_FMT
    return [{"role":"system","content":sys_p},
            {"role":"user","content":f"Utterance: \"{text}\"\n\nClassify emotion."}]


def prepare():
    df = pd.read_parquet(IEMOCAP)
    df["emotion_en"] = df["label_name"].map(LABEL_MAP)
    df = df[df["emotion_en"].isin(LABELS)].copy()
    df = df[df["transcription"].notna() & (df["transcription"].str.len() > 0)].copy()
    rng = np.random.default_rng(SEED)

    # exemplar pool: 1 per class
    pool_idx = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pool_idx.append(sub.iloc[rng.integers(0, len(sub))].name)
    pool = df.loc[pool_idx].copy()
    exemplars = [(r["transcription"], r["emotion_en"]) for _, r in pool.iterrows()]
    remaining = df.drop(index=pool_idx)

    # 20 per class test
    pieces = []
    per_class = N_SAMPLES // 4
    for lab in LABELS:
        sub = remaining[remaining["emotion_en"] == lab]
        pieces.append(sub.iloc[rng.choice(len(sub), size=per_class, replace=False)])
    test = pd.concat(pieces).reset_index(drop=True)
    return test, exemplars


def extract_hidden(model, tokenizer, test_df, exemplars, tag):
    hidden = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        text = row["transcription"]
        msgs = build_prompt(text, exemplars)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float().cpu().numpy()
        hidden.append((h, row["emotion_en"]))
        del out
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            print(f"  [{tag}] {i+1}/{len(test_df)} {time.time()-t0:.1f}s")
    return hidden


def cluster_stats(hidden, labels=LABELS):
    by_label = defaultdict(list)
    for h, l in hidden:
        by_label[l].append(h)
    centroids = {l: np.mean(by_label[l], axis=0) for l in labels if by_label[l]}
    within = {}
    for l, vecs in by_label.items():
        if not vecs: continue
        c = centroids[l]
        within[l] = float(np.mean([np.linalg.norm(v - c) for v in vecs]))
    between = {}
    for l in labels:
        if l not in centroids: continue
        others = [centroids[l2] for l2 in labels if l2 != l and l2 in centroids]
        if not others: continue
        between[l] = float(np.mean([np.linalg.norm(centroids[l] - o) for o in others]))
    tightness = {l: (between[l]/within[l] if within[l]>0 else 0.0) for l in labels if l in within and l in between}
    return {"within_class_dist": within, "between_class_dist": between, "tightness_ratio": tightness,
            "within_overall": float(np.mean(list(within.values()))),
            "between_overall": float(np.mean(list(between.values()))),
            "tightness_overall": float(np.mean(list(tightness.values())))}


test_df, exemplars = prepare()
print(f"[info] N={len(test_df)} exemplars={len(exemplars)}")

snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                             device_map={"":0}, torch_dtype=torch.float16)
model.eval()
print(f"[VRAM] {torch.cuda.memory_allocated()/1024**2:.0f} MB")

results = {}
for name, ex in [("T1_zero_shot", None), ("T2_icl_k4", exemplars)]:
    print(f"\n=== {name} ===")
    hidden = extract_hidden(model, tok, test_df, ex, name)
    stats = cluster_stats(hidden)
    results[name] = stats
    print(f"  within={stats['within_overall']:.3f} between={stats['between_overall']:.3f} tightness={stats['tightness_overall']:.4f}")

tT1 = results["T1_zero_shot"]["tightness_overall"]
tT2 = results["T2_icl_k4"]["tightness_overall"]
wT1 = results["T1_zero_shot"]["within_overall"]
wT2 = results["T2_icl_k4"]["within_overall"]
print(f"\n=== SUMMARY (IEMOCAP) ===")
print(f"T1 within={wT1:.3f} / T2 within={wT2:.3f}  (T2/T1={wT2/wT1:.3f})")
print(f"T1 tightness={tT1:.4f} / T2 tightness={tT2:.4f}")

with open(OUT / "iemocap_hidden_cluster.json", "w") as f:
    json.dump({"task":"IEMOCAP T1 vs T2 hidden cluster (cross-domain mechanism)",
               "N":len(test_df), "results":results,
               "comparison":{"T1_within":wT1, "T2_within":wT2, "T2_over_T1": float(wT2/wT1) if wT1>0 else None}}, f, indent=2)
print(f"[saved] {OUT}/iemocap_hidden_cluster.json")
