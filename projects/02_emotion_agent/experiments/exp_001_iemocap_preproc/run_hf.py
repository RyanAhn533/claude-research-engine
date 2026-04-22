"""
IEMOCAP preprocessing via HuggingFace AbstractTTS/iemocap (pre-segmented).
Local cache: /mnt/hdd/huggingface_cache/datasets/AbstractTTS___iemocap
"""
import os, json
from pathlib import Path
from collections import Counter

os.environ["HF_HOME"] = "/mnt/hdd/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/hdd/huggingface_cache/datasets"

from datasets import load_dataset
import pandas as pd
import numpy as np

OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True)

print("[info] loading AbstractTTS/iemocap from cache...")
try:
    ds = load_dataset("AbstractTTS/IEMOCAP", cache_dir="/mnt/hdd/huggingface_cache/datasets")
except Exception as e:
    # fallback: try with underscore
    print(f"[warn] first load failed: {e}")
    ds = load_dataset("AbstractTTS___iemocap", cache_dir="/mnt/hdd/huggingface_cache/datasets")

print(f"[info] splits: {list(ds.keys())}")
for split, d in ds.items():
    print(f"  {split}: {len(d)} rows, cols={d.column_names}")

# Take the train split (or all concatenated)
full = ds["train"] if "train" in ds else list(ds.values())[0]
print(f"\n[info] sample row keys:", full[0].keys())
print(f"[info] sample (minus audio):")
sample = {k: v for k, v in full[0].items() if k != "audio"}
print(f"  {sample}")

# Find label column
LABEL_CANDIDATES = ["label", "emotion", "major_emotion", "class"]
label_col = None
for c in LABEL_CANDIDATES:
    if c in full.column_names:
        label_col = c
        break
if label_col is None:
    # Inspect first few unique values of each non-audio column
    for c in full.column_names:
        if c == "audio":
            continue
        vals = set()
        for i in range(min(100, len(full))):
            vals.add(str(full[i][c]))
        print(f"  [{c}] unique_first_100={list(vals)[:10]}")
    raise RuntimeError("Cannot find label column")

print(f"\n[info] label column: {label_col}")
labels = [full[i][label_col] for i in range(len(full))]
print(f"[info] label distribution: {Counter(labels).most_common()}")

# Build dataframe (keep audio path or array metadata)
rows = []
for i in range(len(full)):
    r = full[i]
    audio_info = r.get("audio", {})
    rows.append({
        "idx": i,
        "label_raw": r[label_col],
        "audio_path": audio_info.get("path") if isinstance(audio_info, dict) else None,
        "sr": audio_info.get("sampling_rate") if isinstance(audio_info, dict) else None,
        **{k: r[k] for k in full.column_names if k not in ("audio", label_col)},
    })

df = pd.DataFrame(rows)
print(f"\n[info] df shape: {df.shape}")
print(df.head())

# Apply 4-class MERC standard (ang/hap+exc/neu/sad)
def norm(lab):
    lab = str(lab).lower()
    if lab in ("exc", "excited"):
        return "hap"
    if lab in ("happy", "hap", "angry", "ang", "sad", "neu", "neutral"):
        return lab[:3]
    return None

df["label_name"] = df["label_raw"].apply(norm)
mask4 = df["label_name"].isin(["ang", "hap", "neu", "sad"])
df4 = df[mask4].copy()
LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
df4["label"] = df4["label_name"].map(LABEL_MAP)

print(f"\n[info] 4-class kept: {len(df4)} / {len(df)}")
print(df4["label_name"].value_counts().to_dict())

# Save
df4.to_parquet(OUT / "iemocap_4class_hf.parquet", index=False)

stats = {
    "source": "HuggingFace AbstractTTS/iemocap",
    "total_raw": int(len(df)),
    "total_4class": int(len(df4)),
    "per_class": {k: int(v) for k, v in df4["label_name"].value_counts().items()},
    "label_raw_dist": {str(k): int(v) for k, v in Counter(labels).items()},
}
with open(OUT / "iemocap_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[done] {OUT}/iemocap_4class_hf.parquet")
print(f"[done] {OUT}/iemocap_stats.json")
