---
exp_id: exp_001_iemocap_preproc
phase: 1
status: in_progress
date_planned: 2026-04-22
category: data
depends_on: exp_000 (env ready)
---

# IEMOCAP preprocessing

## 목적
IEMOCAP 원본 Session1-5 → 4-class emotion split + audio + text triples preprocessed pickle.
Week 1 Gate: MERC 표준 protocol에서 IEMOCAP 로드 sanity 확보.

## 왜 필요한가
Emotion-LLaMA/M3NET 등 top-tier MER 논문이 모두 IEMOCAP 4-class (hap/ang/neu/sad, exc → hap merge) 동일 protocol 사용. **우리 agent와 공정 비교**하려면 동일 split/label map 필수.

## 방법
- 참조: `/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_data_preprocessing.ipynb` (BG 랩원 코드)
- `torchaudio.datasets.IEMOCAP` 활용 (Session 1-5 자동 indexing)
- Label: `{ang: 0, hap: 1, neu: 2, sad: 3}` — 4-class (MERC 표준, "exc" → "hap" merge, "fru" 제외)
- Audio: 16kHz mono로 resample
- Text: 각 utterance transcription (IEMOCAP dataset metadata)

## 제약
- GPU 불필요 (CPU preprocessing)
- BrandSpace 영향 없음
- 저장 경로: `/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_001_iemocap_preproc/cache/`

## 산출물
- `cache/iemocap_4class.parquet` — 한 row당 (path, session, speaker, text, label, duration)
- `cache/iemocap_stats.json` — class 분포, 길이 통계
- `summary.md` — 요약 보고

## Gate
- Total 4-class sample ≈ 5500 (IEMOCAP 표준 숫자)
- Class 분포: hap>neu>ang>sad 순 (or similar balanced)
- Audio load 1 sample sanity

## Fallback
- `torchaudio.datasets.IEMOCAP`이 API 변경으로 실패 → 직접 Session 디렉토리 scan
