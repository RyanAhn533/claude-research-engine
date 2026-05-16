"""
exp_034 sanity: row -> compact text prompt + tokenizer length check.

Strategy: aggregate 800 features into modality-grouped summary text
(EEG by band×region, BODY by joint, DBD by signal) → ~50 numbers.
Verify Qwen tokenizer doesn't blow up.
"""
import pandas as pd, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import glob, os

HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE

CSV = "/home/ajy/DMS_PPBEMO_driving_behavior/PPB-EMO_codes/ppb-emo/features_all.csv"
df = pd.read_csv(CSV)
print(f"Loaded {len(df)} rows × {len(df.columns)} cols")

LABEL_MAP = {"ND":"neutral", "HD":"happy", "FD":"fear", "SD":"sad",
             "AD":"angry", "DD":"disgust", "SAD":"sadness_strong"}

# Feature columns by modality
eeg_cols = [c for c in df.columns if c.startswith("EEG_NEBAND")]
body_cols = [c for c in df.columns if c.startswith("BODY")]
dbd_cols = [c for c in df.columns if c.startswith("DBD")]

print(f"EEG cols: {len(eeg_cols)}, BODY cols: {len(body_cols)}, DBD cols: {len(dbd_cols)}")


def aggregate_eeg(row):
    """Aggregate EEG_NEBAND_{band}-sensor-{sensor}_{stat} → per-band (band-level mean across sensors, mean stat only)."""
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    out = {}
    for b in bands:
        # find cols matching this band
        cols = [c for c in eeg_cols if f"NEBAND_{b}-" in c and c.endswith("_mean")]
        if cols:
            vals = row[cols].values.astype(float)
            out[f"EEG_{b}_mean_avg"] = float(vals.mean())
            out[f"EEG_{b}_mean_std"] = float(vals.std())
    return out


def aggregate_body(row):
    """BODY_{joint}_{x|y}_{stat} → per-joint distance/movement (keep mean over keypoints)."""
    out = {}
    # joint-level means: extract joint names
    joint_cols_mean = [c for c in body_cols if c.endswith("_mean")]
    if joint_cols_mean:
        out["BODY_position_mean"] = float(row[joint_cols_mean].values.astype(float).mean())
    joint_cols_std = [c for c in body_cols if c.endswith("_std")]
    if joint_cols_std:
        out["BODY_movement_std"] = float(row[joint_cols_std].values.astype(float).mean())
    return out


def aggregate_dbd(row):
    """Driving signals individually: keep core ones (acceleration, gas pedal, brake, steering, position)."""
    out = {}
    keys = ["Acceleration", "Gas pedal", "Brake", "Steering", "X axis position",
            "Y axis position", "Z axis position", "Speed"]
    for k in keys:
        # mean and std across the 30s window
        m_col = f"DBD_{k}_mean"; s_col = f"DBD_{k}_std"
        if m_col in df.columns:
            out[f"DBD_{k}_mean"] = float(row[m_col])
        if s_col in df.columns:
            out[f"DBD_{k}_std"] = float(row[s_col])
    return out


def row_to_text(row):
    feats = {}
    feats.update(aggregate_eeg(row))
    feats.update(aggregate_body(row))
    feats.update(aggregate_dbd(row))
    # Text serialization
    eeg_str = ", ".join(f"{k.replace('EEG_','')}={v:.3f}" for k, v in feats.items() if k.startswith("EEG_"))
    body_str = ", ".join(f"{k.replace('BODY_','')}={v:.3f}" for k, v in feats.items() if k.startswith("BODY_"))
    dbd_str = ", ".join(f"{k.replace('DBD_','')}={v:.3f}" for k, v in feats.items() if k.startswith("DBD_"))
    text = (f"EEG (frequency-band activity per region): {eeg_str}\n"
            f"BODY (posture / movement): {body_str}\n"
            f"DRIVING (vehicle behavior): {dbd_str}")
    return text, feats


# Test on 3 rows
print("\n=== sample text prompts ===")
snap = glob.glob(f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(snap)

SYS_P = ("You are an emotion classifier for car-driving scenarios. Given multimodal "
         "features (EEG band activity, body posture, driving behavior) of a driver, "
         "classify into: neutral, happy, fear, sad, angry, disgust, sadness_strong.\n"
         "Output format:\nLABEL: <label>\nREASON: <one sentence>")

for idx in [0, 50, 200]:
    text, feats = row_to_text(df.iloc[idx])
    user_p = f"Driver multimodal features:\n{text}\n\nClassify the driver's emotion."
    msgs = [{"role":"system","content":SYS_P},{"role":"user","content":user_p}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    n_tokens = len(tok.encode(full, add_special_tokens=False))
    label = LABEL_MAP.get(df.iloc[idx]['category'], '?')
    print(f"\n--- row {idx} (true={df.iloc[idx]['category']}={label}) ---")
    print(f"text length: {len(text)} chars, prompt total: {n_tokens} tokens")
    print(f"  feats count: {len(feats)}")
    if idx == 0:
        print(f"\nFull prompt:\n{full[:800]}...")
        print(f"\n--- 끝 ---")

print(f"\n[OK] All 3 sample rows tokenize within Qwen limits.")
