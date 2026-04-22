# exp_001 — IEMOCAP Preprocessing PASS

## Outcome
- Source: HuggingFace `AbstractTTS/IEMOCAP` (local cache, 3-shard arrow)
- Raw rows: 10,039
- **4-class MERC 표준 kept: 6,877**
  - happy (exc merged): 2,632
  - neutral: 1,726
  - angry: 1,269
  - sad: 1,250
- Non-audio features kept: gender, transcription, speaking_rate, pitch_mean/std, rms, relative_db

## Files
- `cache/iemocap_4class_hf.parquet` — 4-class metadata + features
- `cache/iemocap_stats.json` — counts per class/column

## Lessons learned
- BG 랩원 IEMOCAP original은 **dialog-level wav만 존재**, utterance segmentation 불완전 → 재사용 X
- HF `AbstractTTS/IEMOCAP` arrow 3-shard = **pre-segmented utterance + label + prosody features**. 정석 소스.
- `datasets.load_dataset` 으로 열면 cache lock permission 이슈 → **arrow 파일 직접 `pyarrow.ipc` 로 읽기**가 안정적
- Label: 'exc' → 'hap' merge가 MERC 표준 (Emotion-LLaMA, M3NET 동일)

## Direction recorded
- `EMA-D002`: HF arrow 직접 read 패턴 (cache permission 우회)

## Next
- Audio 실제 load sanity (arrow 안 audio 컬럼)
- MELD + DEAP preprocessing (exp_002 / exp_003)
