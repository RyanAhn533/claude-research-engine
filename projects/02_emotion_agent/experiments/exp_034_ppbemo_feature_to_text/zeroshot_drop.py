"""
exp_034 zero-shot + modality drop (kill criteria check).

Kill criterion 1: zero-shot acc > random (7-class = ~14%) on PPB-Emo.
Kill criterion 2: modality drop changes accuracy in a non-trivial way.

If either fails clearly, this paper direction is dead → stop.
"""
import os, sys, json, time, glob, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE

CSV = "/home/ajy/DMS_PPBEMO_driving_behavior/PPB-EMO_codes/ppb-emo/features_all.csv"
OUT = Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True, parents=True)

LABELS_RAW = ["ND", "HD", "FD", "SD", "AD", "DD", "SAD"]
LABELS_EN  = ["neutral", "happy", "fear", "sad", "angry", "disgust", "sadness_strong"]
LABEL_MAP  = dict(zip(LABELS_RAW, LABELS_EN))
INV_MAP    = {v: k for k, v in LABEL_MAP.items()}

df = pd.read_csv(CSV)
df["label_en"] = df["category"].map(LABEL_MAP)
print(f"Loaded {len(df)} rows. Class dist: {df['category'].value_counts().to_dict()}")

eeg_cols = [c for c in df.columns if c.startswith("EEG_NEBAND")]
body_cols = [c for c in df.columns if c.startswith("BODY")]
dbd_cols = [c for c in df.columns if c.startswith("DBD")]


def aggregate_eeg(row):
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    out = {}
    for b in bands:
        cols = [c for c in eeg_cols if f"NEBAND_{b}-" in c and c.endswith("_mean")]
        if cols:
            vals = row[cols].values.astype(float)
            out[f"EEG_{b}_mean_avg"] = float(vals.mean())
            out[f"EEG_{b}_mean_std"] = float(vals.std())
    return out


def aggregate_body(row):
    out = {}
    cm = [c for c in body_cols if c.endswith("_mean")]
    cs = [c for c in body_cols if c.endswith("_std")]
    if cm: out["BODY_position_mean"] = float(row[cm].values.astype(float).mean())
    if cs: out["BODY_movement_std"] = float(row[cs].values.astype(float).mean())
    return out


def aggregate_dbd(row):
    out = {}
    keys = ["Acceleration", "Gas pedal", "Brake", "Steering",
            "X axis position", "Y axis position", "Z axis position", "Speed"]
    for k in keys:
        m = f"DBD_{k}_mean"; s = f"DBD_{k}_std"
        if m in df.columns: out[f"DBD_{k}_mean"] = float(row[m])
        if s in df.columns: out[f"DBD_{k}_std"] = float(row[s])
    return out


def build_text(feats):
    eeg_str = ", ".join(f"{k.replace('EEG_','')}={v:.3f}" for k, v in feats.items() if k.startswith("EEG_"))
    body_str = ", ".join(f"{k.replace('BODY_','')}={v:.3f}" for k, v in feats.items() if k.startswith("BODY_"))
    dbd_str = ", ".join(f"{k.replace('DBD_','')}={v:.3f}" for k, v in feats.items() if k.startswith("DBD_"))
    parts = []
    if eeg_str: parts.append(f"EEG (frequency-band activity per region): {eeg_str}")
    if body_str: parts.append(f"BODY (posture / movement): {body_str}")
    if dbd_str: parts.append(f"DRIVING (vehicle behavior): {dbd_str}")
    return "\n".join(parts) if parts else "[no features available]"


def row_to_text(row, drop=None):
    feats = {}
    if drop != "eeg":  feats.update(aggregate_eeg(row))
    if drop != "body": feats.update(aggregate_body(row))
    if drop != "dbd":  feats.update(aggregate_dbd(row))
    return build_text(feats)


SYS_P = (
    "You are an emotion classifier for car-driving scenarios. Given multimodal "
    "features (EEG band activity, body posture, driving behavior) of a driver, "
    "classify into one of: neutral, happy, fear, sad, angry, disgust, sadness_strong.\n"
    "Output format:\nLABEL: <label>\nREASON: <one sentence>"
)

LABEL_RE = re.compile(r"LABEL\s*:\s*(\w+)", re.IGNORECASE)


def parse_label(text):
    m = LABEL_RE.search(text)
    if not m: return "neutral"
    raw = m.group(1).lower()
    for lab in LABELS_EN:
        if lab.startswith(raw[:3]) or raw.startswith(lab[:3]) or raw == lab:
            return lab
    return "neutral"


snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, quantization_config=bnb,
                                             device_map={"":0}, torch_dtype=torch.float16)
model.eval()
print(f"[VRAM] {torch.cuda.memory_allocated()/1024**2:.0f} MB")


def run_zero_shot(eval_df, drop=None, tag="full"):
    preds = []; t0 = time.time()
    y_true = eval_df["label_en"].tolist()
    for i, (_, row) in enumerate(eval_df.iterrows()):
        text = row_to_text(row, drop=drop)
        msgs = [{"role":"system","content":SYS_P},
                {"role":"user","content":f"Driver multimodal features:\n{text}\n\nClassify the driver's emotion."}]
        ps = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(ps, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)
        preds.append(parse_label(raw))
        if (i+1) % 60 == 0:
            print(f"  [{tag}] {i+1}/{len(eval_df)} {time.time()-t0:.0f}s")
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, labels=LABELS_EN, average="macro", zero_division=0))
    return acc, f1, preds


# Use full 240 (zero-shot, no train)
results = {}
for tag, drop in [("full", None), ("drop_eeg", "eeg"), ("drop_body", "body"), ("drop_dbd", "dbd")]:
    print(f"\n=== {tag} ===")
    acc, f1, preds = run_zero_shot(df, drop=drop, tag=tag)
    print(f"  acc={acc*100:.2f}%  macro-F1={f1:.3f}")
    results[tag] = {"acc": acc, "f1": f1, "preds": preds}

# Random baseline reminders
n_classes = 7
print(f"\nRandom baseline (7-class): {100/n_classes:.2f}%")
print(f"Majority class baseline: {df['category'].value_counts().iloc[0]/len(df)*100:.2f}%")

# Kill criterion check
full_acc = results["full"]["acc"] * 100
random_acc = 100 / n_classes
majority_acc = df['category'].value_counts().iloc[0] / len(df) * 100

print(f"\n=== KILL CRITERIA ===")
print(f"1. Zero-shot vs random ({random_acc:.2f}%): full={full_acc:.2f}% → {'PASS' if full_acc > random_acc + 3 else 'FAIL'}")
print(f"   Zero-shot vs majority ({majority_acc:.2f}%): {'PASS' if full_acc > majority_acc else 'FAIL or tied'}")

drop_accs = [results[k]["acc"] * 100 for k in ["drop_eeg", "drop_body", "drop_dbd"]]
max_drop_diff = max(abs(full_acc - x) for x in drop_accs)
print(f"2. Modality drop variability: full={full_acc:.2f}% / drops={drop_accs} / max diff={max_drop_diff:.2f}pp")
print(f"   {'PASS (LLM responsive to modality)' if max_drop_diff > 2 else 'WEAK (LLM ignores modality changes)'}")

with open(OUT / "zeroshot_drop_results.json", "w") as f:
    json.dump({
        "task": "PPB-Emo zero-shot + modality drop (kill criteria)",
        "random_baseline_pct": random_acc,
        "majority_baseline_pct": majority_acc,
        "results": {k: {"acc": v["acc"], "f1": v["f1"]} for k, v in results.items()},
    }, f, indent=2)

# Save predictions for failure analysis later
preds_df = df[["segment_id","participant","category","label_en"]].copy()
for k, v in results.items():
    preds_df[f"pred_{k}"] = v["preds"]
preds_df.to_csv(OUT / "predictions.csv", index=False)
print(f"\n[saved] {OUT}/zeroshot_drop_results.json + predictions.csv")
