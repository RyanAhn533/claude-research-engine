"""
IEMOCAP preprocessing — 4-class MERC standard split.

Output:
  cache/iemocap_4class.parquet  — columns: path, session, speaker, text, label, duration, sr
  cache/iemocap_stats.json      — class/length stats
"""
import os, sys, json, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path("/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_full_release/IEMOCAP_full_release")
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True, parents=True)

# 4-class MERC standard: ang/hap(+exc)/neu/sad, fru excluded
LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
LABEL_NAMES = ["angry", "happy", "neutral", "sad"]


def parse_emoevaluation(session_dir: Path):
    """Parse EmoEvaluation/*.txt for (utt_id, label, start, end, vad).
    IEMOCAP EmoEvaluation line format:
      [start - end]  utt_id  label  [val act dom]
    """
    records = []
    evo_dir = session_dir / "dialog" / "EmoEvaluation"
    if not evo_dir.exists():
        return records
    for txt in sorted(evo_dir.glob("*.txt")):
        with open(txt, "r", encoding="latin-1") as f:
            for line in f:
                if not line.startswith("["):
                    continue
                # e.g.
                # [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                m = re.match(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\S+)", line)
                if not m:
                    continue
                start, end, uttid, lab = m.groups()
                records.append({
                    "utt_id": uttid,
                    "label_raw": lab,
                    "start": float(start),
                    "end": float(end),
                })
    return records


def find_wav(session_dir: Path, utt_id: str):
    """utt_id like 'Ses01F_impro01_F000' → session/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"""
    dialog = "_".join(utt_id.split("_")[:2])  # Ses01F_impro01
    wav = session_dir / "sentences" / "wav" / dialog / f"{utt_id}.wav"
    return wav if wav.exists() else None


def find_transcription(session_dir: Path, utt_id: str):
    """sentences/transcription/Ses01F_impro01.txt contains multiple lines:
       Ses01F_impro01_F000 [start-end]: text
    """
    dialog = "_".join(utt_id.split("_")[:2])
    trans = session_dir / "dialog" / "transcriptions" / f"{dialog}.txt"
    if not trans.exists():
        return ""
    with open(trans, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(utt_id):
                # e.g. "Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me."
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return ""


def norm_label(lab: str):
    """Map exc → hap; drop 'xxx', 'oth', 'fru', 'sur', 'fea', 'dis' (not in 4-class)."""
    if lab == "exc":
        return "hap"
    return lab


# === Main ===
print(f"[info] root: {ROOT}")
assert ROOT.exists(), f"IEMOCAP root missing: {ROOT}"

rows = []
drops = Counter()

for s in range(1, 6):
    session_dir = ROOT / f"Session{s}"
    if not session_dir.exists():
        print(f"[warn] {session_dir} missing, skip")
        continue
    print(f"[info] Session{s}: scanning EmoEvaluation...")
    records = parse_emoevaluation(session_dir)
    print(f"  {len(records)} raw records")

    for r in records:
        lab = norm_label(r["label_raw"])
        if lab not in LABEL_MAP:
            drops[lab] += 1
            continue
        uttid = r["utt_id"]
        wav = find_wav(session_dir, uttid)
        if wav is None:
            drops["no_wav"] += 1
            continue
        text = find_transcription(session_dir, uttid)
        speaker = uttid.split("_")[-1][0]  # F / M
        rows.append({
            "utt_id": uttid,
            "path": str(wav),
            "session": s,
            "speaker": speaker,
            "text": text,
            "label_name": lab,
            "label": LABEL_MAP[lab],
            "start": r["start"],
            "end": r["end"],
            "duration": r["end"] - r["start"],
            "sr": 16000,  # IEMOCAP standard
        })

df = pd.DataFrame(rows)
print(f"\n[info] kept {len(df)} utterances")
print(f"[info] dropped by label (not in 4-class / no wav):")
for k, v in drops.most_common():
    print(f"  {k:10s}  {v}")

print("\n[class dist]")
print(df["label_name"].value_counts().reindex(["angry", "happy", "neutral", "sad"]).fillna(0))

print("\n[duration stats]")
print(df["duration"].describe())

# Save parquet + json stats
df.to_parquet(OUT / "iemocap_4class.parquet", index=False)

stats = {
    "total": int(len(df)),
    "per_class": {str(k): int(v) for k, v in df["label_name"].value_counts().to_dict().items()},
    "per_session": {int(k): int(v) for k, v in df["session"].value_counts().to_dict().items()},
    "per_speaker": {str(k): int(v) for k, v in df["speaker"].value_counts().to_dict().items()},
    "duration_mean_sec": float(df["duration"].mean()),
    "duration_median_sec": float(df["duration"].median()),
    "duration_max_sec": float(df["duration"].max()),
    "dropped": dict(drops),
}
with open(OUT / "iemocap_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n[done] saved: {OUT}/iemocap_4class.parquet")
print(f"[done] stats:  {OUT}/iemocap_stats.json")

# Sanity: load 1 wav
try:
    import soundfile as sf
    import random
    sample = df.sample(1, random_state=0).iloc[0]
    data, sr = sf.read(sample["path"])
    print(f"\n[sanity] {sample['utt_id']} label={sample['label_name']} sr={sr} shape={data.shape} duration_metadata={sample['duration']:.2f}")
except Exception as e:
    print(f"[sanity] wav load failed: {e}")
