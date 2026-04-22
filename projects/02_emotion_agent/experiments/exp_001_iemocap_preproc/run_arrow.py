"""
Direct arrow file read — bypass datasets library cache locking.
"""
import os, json
from pathlib import Path
from collections import Counter

import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

ARROW_DIR = Path("/mnt/hdd/huggingface_cache/datasets/AbstractTTS___iemocap/default/0.0.0/9f1696a135a65ce997d898d4121c952269a822ca")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True)

shards = sorted(ARROW_DIR.glob("iemocap-train-*-of-*.arrow"))
print(f"[info] {len(shards)} shards")

tables = []
for sh in shards:
    print(f"  reading {sh.name}...")
    try:
        with ipc.open_stream(sh) as reader:
            t = reader.read_all()
    except Exception:
        with ipc.open_file(sh) as reader:
            t = reader.read_all()
    print(f"    rows={t.num_rows}  cols={t.column_names}")
    tables.append(t)

full = pa.concat_tables(tables)
print(f"\n[info] total: {full.num_rows}  cols={full.column_names}")

# Non-audio columns to pandas
cols_light = [c for c in full.column_names if c not in ("audio", "array")]
df_light = full.select(cols_light).to_pandas()
print("\n[info] sample (light cols):")
print(df_light.head(3))
print("\n[info] dtypes:")
print(df_light.dtypes)

# Label detection
LABEL_CANDIDATES = ["emotion", "label", "major_emotion", "class", "emo"]
label_col = None
for c in LABEL_CANDIDATES:
    if c in df_light.columns:
        label_col = c
        break
if label_col is None:
    for c in df_light.columns:
        print(f"  [{c}] sample: {df_light[c].iloc[0]}")
    raise RuntimeError("no label col found")

print(f"\n[info] label column: {label_col}")
print(df_light[label_col].value_counts())

# 4-class mapping
def norm(lab):
    lab = str(lab).lower().strip()
    if lab in ("exc", "excited"):
        return "hap"
    if lab.startswith("happy") or lab == "hap":
        return "hap"
    if lab.startswith("angry") or lab == "ang":
        return "ang"
    if lab.startswith("sad"):
        return "sad"
    if lab.startswith("neu"):
        return "neu"
    return None

df_light["label_name"] = df_light[label_col].apply(norm)
mask4 = df_light["label_name"].isin(["ang", "hap", "neu", "sad"])
df4 = df_light[mask4].copy()
LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
df4["label"] = df4["label_name"].map(LABEL_MAP)

print(f"\n[info] 4-class kept: {len(df4)} / {len(df_light)}")
print(df4["label_name"].value_counts().to_dict())

df4.to_parquet(OUT / "iemocap_4class_hf.parquet", index=False)

stats = {
    "source": "HF AbstractTTS/IEMOCAP (direct arrow read)",
    "shard_count": len(shards),
    "total_raw": int(len(df_light)),
    "total_4class": int(len(df4)),
    "per_class": {k: int(v) for k, v in df4["label_name"].value_counts().items()},
    "label_col": label_col,
    "label_raw_dist": {str(k): int(v) for k, v in df_light[label_col].value_counts().items()},
    "columns": list(df_light.columns),
}
with open(OUT / "iemocap_stats.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n[done] {OUT}/iemocap_4class_hf.parquet")
print(f"[done] {OUT}/iemocap_stats.json")
