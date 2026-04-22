# exp_002 — MELD preprocessing PASS

## Outcome
- Source: `/data/heartlab/datasets/MELD/MELD.Raw/`
- Total: **13,708** utterances
- Splits: train 9989 / dev 1109 / test 2610 (official)
- Video presence: 200/200 sample OK
- 7-class emotion: neutral 6436, joy 2308, surprise 1636, anger 1607, sadness 1002, disgust 361, fear 358
- 3-class sentiment: neutral 6436, negative 4184, positive 3088
- **4-class (IEMOCAP-aligned)**: 11,353 (neutral 6436, joy→hap 2308, anger→ang 1607, sadness→sad 1002)

## Files
- `cache/meld_full.parquet` — 13708 rows w/ 7-class + sentiment + video_path
- `cache/meld_4class.parquet` — 11353 rows (ang/hap/neu/sad)
- `cache/meld_stats.json`

## Schema
`sr_no, text, speaker, emotion_7, sentiment, dialogue_id, utterance_id, season, episode, start_time, end_time, split, video_path, label_7, label_4, label_4_name`

## Note
- Video = mp4 per utterance at `{split}_splits/dia{X}_utt{Y}.mp4` 패턴
- 전체 video 약 13708 mp4 파일 존재 (spot check)
- IEMOCAP 4-class subset과 호환 → 두 dataset 공통 label space로 실험 가능
