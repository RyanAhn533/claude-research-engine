"""
exp_030b: Prompt ablation MULTI-SEED (3 seeds) for Korean FER AU zero-shot.
Validates exp_030 single-seed surprising finding (P4 cultural 39.5% > FACS 29.25%).
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

AU_PARQUET = Path("/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet")
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)

LABELS = ["angry","happy","neutral","sad"]
KR_TO_EN = {"기쁨":"happy","분노":"angry","슬픔":"sad","중립":"neutral"}
AU_DESC = {
    "AU1":"inner brow raiser","AU2":"outer brow raiser","AU4":"brow lowerer",
    "AU5":"upper lid raiser","AU6":"cheek raiser","AU7":"lid tightener",
    "AU9":"nose wrinkler","AU10":"upper lip raiser","AU12":"lip corner puller",
    "AU14":"dimpler","AU15":"lip corner depressor","AU17":"chin raiser",
    "AU20":"lip stretcher","AU23":"lip tightener","AU25":"lips part",
    "AU26":"jaw drop","AU27":"mouth stretch",
}
KEY_AUS = list(AU_DESC.keys())
SEEDS = [42, 123, 777]
N_TEST_PER_CLASS = 100

OUTPUT_FMT = "\nOutput format:\nLABEL: <label>\nREASON: <one sentence>"
PROMPTS = {
    "P1_facs_baseline": (
        "You are a facial emotion classifier using FACS action units (AUs). "
        "Given detected AU intensities (0-100) of a Korean subject, classify into: "
        "angry, happy, neutral, sad. FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23."),
    "P2_no_facs": (
        "You are a facial emotion classifier. Given detected facial action unit (AU) "
        "intensities (0-100) of a Korean subject, classify into: angry, happy, neutral, sad."),
    "P3_minimal": (
        "Classify the emotion from these AU intensities. Options: angry, happy, neutral, sad."),
    "P4_cultural": (
        "You are a facial emotion classifier using FACS action units (AUs). The subject is Korean; "
        "cultural factors may influence expression. Classify into: angry, happy, neutral, sad. "
        "FACS prototypes: Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23."),
    "P5_dimensional": (
        "You are a facial emotion classifier. Analyze the AU intensities (0-100) along arousal "
        "(activation level) and valence (positive/negative) dimensions, then classify the subject's "
        "emotion into: angry (high-arousal negative), happy (high-arousal positive), "
        "neutral (low-arousal neutral), sad (low-arousal negative)."),
}


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


def build_prompt(text, sys_base):
    sys_p = sys_base + OUTPUT_FMT
    return [{"role":"system","content":sys_p},
            {"role":"user","content":f"AUs: {text}\n\nClassify emotion."}]


def prepare_test(seed):
    df = pd.read_parquet(AU_PARQUET)
    df = df[df["is_selected"] == 0].copy()
    df["emotion_en"] = df["emotion"].map(KR_TO_EN)
    df = df[df["emotion_en"].notna()].copy()
    rng = np.random.default_rng(seed)
    pieces = []
    for lab in LABELS:
        sub = df[df["emotion_en"] == lab]
        pieces.append(sub.iloc[rng.choice(len(sub), size=N_TEST_PER_CLASS, replace=False)])
    return pd.concat(pieces).reset_index(drop=True)


def run(test_df, sys_base, model, tok, tag):
    preds = []; t0 = time.time()
    y_true = test_df["emotion_en"].tolist()
    for i, row in test_df.iterrows():
        text = au_to_text(row)
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

per_seed = {}
per_prompt = {k: {"acc":[], "f1":[]} for k in PROMPTS}

for seed in SEEDS:
    print(f"\n###### SEED {seed} ######")
    test_df = prepare_test(seed)
    seed_res = {}
    for name, sys_p in PROMPTS.items():
        print(f"\n=== seed={seed} {name} ===")
        acc, f1 = run(test_df, sys_p, model, tok, f"{name}_s{seed}")
        print(f"  acc={acc*100:.2f}%  F1={f1:.3f}")
        seed_res[name] = {"acc":acc, "f1":f1}
        per_prompt[name]["acc"].append(acc); per_prompt[name]["f1"].append(f1)
    per_seed[seed] = seed_res

agg = {}
for name in PROMPTS:
    accs = np.array(per_prompt[name]["acc"]); f1s = np.array(per_prompt[name]["f1"])
    agg[name] = {"acc_mean":float(accs.mean()), "acc_std":float(accs.std(ddof=1)),
                 "f1_mean":float(f1s.mean()), "f1_std":float(f1s.std(ddof=1)),
                 "accs":accs.tolist(), "f1s":f1s.tolist()}

with open(OUT / "prompt_ablation_multiseed_results.json", "w") as f:
    json.dump({"task":"Prompt ablation multi-seed (3 seeds)","seeds":SEEDS,
               "per_seed":{str(s):per_seed[s] for s in SEEDS},
               "aggregate":agg}, f, indent=2)

print("\n[DONE]")
for name, a in agg.items():
    print(f"  {name}: acc={a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f}%  F1={a['f1_mean']:.3f}±{a['f1_std']:.3f}")
