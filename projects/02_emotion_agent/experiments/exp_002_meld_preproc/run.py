"""
MELD preprocessing — 7-class + 4-class unified w/ video path.

MERC standard uses 7-class emotion. We also provide 4-class subset (joy/ang/sad/neu)
matching IEMOCAP so that agent can be evaluated in both modes.
"""
import os, json
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path("/data/heartlab/datasets/MELD/MELD.Raw")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

# Split → (csv, video_dir_name)
SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev":   ("dev_sent_emo.csv",   "dev_splits_complete"),
    "test":  ("test_sent_emo.csv",  "output_repeated_splits_test"),
}

EMO_7 = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
EMO_7_MAP = {e: i for i, e in enumerate(EMO_7)}

# 4-class aligned with IEMOCAP (ang/hap/neu/sad)
EMO_4_MAP_NAME = {"anger": "ang", "joy": "hap", "sadness": "sad", "neutral": "neu"}
EMO_4_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}


all_rows = []
for split, (csv_name, vid_dir) in SPLITS.items():
    csv_path = ROOT / csv_name
    vid_root = ROOT / vid_dir
    print(f"[info] split={split}  csv={csv_path.exists()}  vid_dir={vid_root.exists()}")
    df = pd.read_csv(csv_path)
    df["split"] = split
    df["video_path"] = df.apply(
        lambda r: str(vid_root / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4"),
        axis=1
    )
    # Check video existence for first 5
    exist_flags = df["video_path"].head(5).apply(lambda p: Path(p).exists()).tolist()
    print(f"  first-5 video exists: {exist_flags}  len={len(df)}")
    all_rows.append(df)

df = pd.concat(all_rows, ignore_index=True)
print(f"\n[info] total rows: {len(df)}")
print(f"[info] per split:\n{df['split'].value_counts()}")
print(f"\n[info] emotion dist:\n{df['Emotion'].value_counts()}")
print(f"\n[info] sentiment dist:\n{df['Sentiment'].value_counts()}")

# Rename cleanly
df = df.rename(columns={
    "Sr No.": "sr_no",
    "Utterance": "text",
    "Speaker": "speaker",
    "Emotion": "emotion_7",
    "Sentiment": "sentiment",
    "Dialogue_ID": "dialogue_id",
    "Utterance_ID": "utterance_id",
    "Season": "season",
    "Episode": "episode",
    "StartTime": "start_time",
    "EndTime": "end_time",
})

df["label_7"] = df["emotion_7"].map(EMO_7_MAP)
df["label_4_name"] = df["emotion_7"].map(EMO_4_MAP_NAME)
df["label_4"] = df["label_4_name"].map(EMO_4_MAP)

# Check video existence (full sample)
import random
sample_ids = random.Random(42).sample(range(len(df)), 200)
missing_count = sum(1 for i in sample_ids if not Path(df.iloc[i]["video_path"]).exists())
print(f"\n[sanity] sample 200 / missing videos: {missing_count}")

# Save
df.to_parquet(OUT / "meld_full.parquet", index=False)
df_4 = df[df["label_4_name"].notna()].copy()
df_4.to_parquet(OUT / "meld_4class.parquet", index=False)

stats = {
    "total": int(len(df)),
    "per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
    "emotion_7_dist": {k: int(v) for k, v in df["emotion_7"].value_counts().items()},
    "sentiment_dist": {k: int(v) for k, v in df["sentiment"].value_counts().items()},
    "emotion_4_total": int(len(df_4)),
    "emotion_4_dist": {k: int(v) for k, v in df_4["label_4_name"].value_counts().items()},
    "video_missing_sample": missing_count,
    "video_sample_size": 200,
    "columns": list(df.columns),
}
with open(OUT / "meld_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[done] {OUT}/meld_full.parquet  (n={len(df)})")
print(f"[done] {OUT}/meld_4class.parquet (n={len(df_4)})")
print(f"[done] {OUT}/meld_stats.json")
