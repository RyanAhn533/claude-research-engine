"""
Phase 0 Gate 4: 주요 dataset load sanity check.
"""
import os, json
from pathlib import Path

OUT = Path(__file__).parent / "logs"
OUT.mkdir(exist_ok=True, parents=True)

paths = {
    "IEMOCAP_root": "/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_full_release/IEMOCAP_full_release",
    "IEMOCAP_session1": "/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_full_release/IEMOCAP_full_release/Session1",
    "MELD_data": "/data/heartlab/datasets/MELD",
    "MELD_raw_alt": "/mnt/hdd/BG/private_paper/datasets/MELD.Raw",
    "CMU_MOSEI": "/data/heartlab/datasets/CMU_MOSEI",
    "CMU_MOSI": "/data/heartlab/datasets/CMU_MOSI",
    "DEAP_code": "/mnt/hdd/BG/MER/DEAP_code",
    "AU_embeddings_project01": "/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings",
    "Yonsei_consensus_csv": "/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv",
    "Korean_FER_237K": "/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea",
    "Qwen25_7B_cache": "/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",
    "Qwen25_VL_cache": "/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct",
}

result = {}
for name, p in paths.items():
    pp = Path(p)
    result[name] = {
        "path": p,
        "exists": pp.exists(),
        "is_dir": pp.is_dir() if pp.exists() else None,
        "n_items_top": len(list(pp.iterdir())) if pp.exists() and pp.is_dir() else None,
    }

# Spot check files
import pandas as pd
try:
    df = pd.read_csv(paths["Yonsei_consensus_csv"])
    result["Yonsei_consensus_csv"]["rows"] = len(df)
    result["Yonsei_consensus_csv"]["cols"] = df.columns.tolist()
    result["Yonsei_consensus_csv"]["load_ok"] = True
except Exception as e:
    result["Yonsei_consensus_csv"]["load_ok"] = False
    result["Yonsei_consensus_csv"]["error"] = str(e)

# IEMOCAP: check Session1 WAV
try:
    ses1 = Path(paths["IEMOCAP_session1"])
    wavs = list(ses1.rglob("*.wav"))[:3]
    result["IEMOCAP_session1"]["wav_sample_count_first3"] = len(wavs)
    result["IEMOCAP_session1"]["wav_sample_names"] = [w.name for w in wavs]
except Exception as e:
    result["IEMOCAP_session1"]["error"] = str(e)

# MELD: check csv
try:
    for base in [paths["MELD_data"], paths["MELD_raw_alt"]]:
        bp = Path(base)
        if bp.exists():
            csvs = list(bp.rglob("*.csv"))[:5]
            result[f"MELD_csv_at_{bp.name}"] = {"samples": [c.name for c in csvs]}
except Exception as e:
    result["MELD_scan_error"] = str(e)

with open(OUT / "data_sanity.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# Summary print
print("=== Data sanity ===")
for name, r in result.items():
    if isinstance(r, dict):
        status = "OK " if r.get("exists") else "MISS"
        print(f"  [{status}] {name:30s}  {r.get('path','')[:70]}")
print(f"\nJSON saved: {OUT / 'data_sanity.json'}")
