"""
K-EmoCon metadata snapshot — bio + Korean multimodal asset.
Full preprocessing은 JY SGMT pipeline에서 이미 수행. 여기선 agent 프로젝트에 필요한 metadata만 index.
"""
import os, json
from pathlib import Path
from collections import Counter
import pandas as pd

SRC = Path("/home/ajy/Jetson_thor/data/precessed_data/kemocon")
LABEL_CSV = SRC / "label-segments_index_clean.csv"
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(LABEL_CSV)
print(f"[info] total rows: {len(df)}  cols: {len(df.columns)}")
print(f"[info] pid range: {sorted(df['pid'].unique())[:5]}... (n={df['pid'].nunique()})")
print(f"[info] annotator sources: {df['label_source'].value_counts().to_dict()}")

# arousal/valence ranges
print(f"\n[info] arousal dist (1-5 scale):\n{df['arousal'].value_counts().sort_index()}")
print(f"\n[info] valence dist (1-5 scale):\n{df['valence'].value_counts().sort_index()}")

# discrete label
print(f"\n[info] label_discrete top:\n{df['label_discrete'].value_counts().head(15)}")

# 4-class mapping (ang/hap/neu/sad) for IEMOCAP alignment
# Based on common K-EmoCon discrete → basic emotion mapping
MAP = {
    "cheerful": "hap", "happy": "hap", "delight": "hap", "pride": "hap",
    "angry": "ang", "frustration": "ang", "contempt": "ang",
    "sad": "sad", "sorrow": "sad", "dejection": "sad",
    "concentration": "neu", "none_1": "neu", "none_2": "neu",
}
df["label_4_name"] = df["label_discrete"].map(MAP)
mask = df["label_4_name"].notna()
LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
df["label_4"] = df["label_4_name"].map(LABEL_MAP)

# Valence-Arousal quadrant (for regression baseline)
df["va_quadrant"] = df.apply(
    lambda r: ("HV" if r["valence"] >= 3 else "LV") + "_" + ("HA" if r["arousal"] >= 3 else "LA"),
    axis=1,
)

# Aggregated external annotation subset only (most reliable)
agg = df[df["label_source"] == "aggregated_external"].copy()
print(f"\n[info] aggregated_external subset: {len(agg)} segments, {agg['pid'].nunique()} participants")

# Save
df.to_parquet(OUT / "kemocon_labels_full.parquet", index=False)
agg.to_parquet(OUT / "kemocon_aggregated.parquet", index=False)

stats = {
    "source": str(LABEL_CSV),
    "total_rows": int(len(df)),
    "unique_participants": int(df["pid"].nunique()),
    "annotator_sources": {k: int(v) for k, v in df["label_source"].value_counts().items()},
    "arousal_dist": {int(k): int(v) for k, v in df["arousal"].value_counts().items()},
    "valence_dist": {int(k): int(v) for k, v in df["valence"].value_counts().items()},
    "va_quadrant_dist": {k: int(v) for k, v in df["va_quadrant"].value_counts().items()},
    "label_discrete_top10": {k: int(v) for k, v in df["label_discrete"].value_counts().head(10).items()},
    "label_4class_mapped": int(mask.sum()),
    "label_4class_dist": {k: int(v) for k, v in df[mask]["label_4_name"].value_counts().items()},
    "aggregated_external_rows": int(len(agg)),
    "bio_root": str(SRC / "e4_data"),
    "bio_participants": sorted([int(d.name) for d in (SRC / "e4_data").iterdir() if d.is_dir() and d.name.isdigit()]),
}
with open(OUT / "kemocon_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[done] {OUT}/kemocon_labels_full.parquet  ({len(df)} rows)")
print(f"[done] 4-class subset: {mask.sum()}")
print(f"[done] aggregated_external: {len(agg)}")
print(f"[done] bio participants: {stats['bio_participants']}")
