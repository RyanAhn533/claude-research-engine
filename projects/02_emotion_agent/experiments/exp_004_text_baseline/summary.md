# exp_004 — Text-only baseline (IEMOCAP + MELD)

## Results

| Dataset | Split | Acc | Macro-F1 | vs Random (25%) |
|---------|-------|-----|----------|----------|
| IEMOCAP | 3-seed 80/20 | **66.33% ± 0.84** | **0.640 ± 0.007** | +41%p |
| MELD | official train→test | **59.75%** | **0.312** | +35%p but class imbalance dominant |

Random baseline (4-class uniform): 25.0%

## Method
- Preprocessing: 4-class subset (ang/hap/neu/sad), text (transcription 또는 Utterance)
- Vectorizer: TF-IDF word 1-2gram, 50k features, sublinear_tf
- Classifier: LogisticRegression (liblinear)
- Seeds: 42, 123, 777

## Observations
- IEMOCAP text-only 66% = SOTA MERC (~70-75% multimodal) 대비 −5%p. Reasonable.
- MELD text-only 60% = class imbalance (neutral 57% 지배). macro-F1 0.31이 실제 성능.
- Multimodal (+ audio + video) 추가 시 MELD 대폭 향상 기대 (SOTA ~65-68%).

## Purpose in Q1 논문
- §4.1 ablation: "Text-only baseline" 참고값 (multimodal 대비 기여 증명)
- §5 limitations: MELD text-only의 class imbalance pitfall

## Next
- Day 5: Emotion-LLaMA repo clone + 구조 파악
- Week 2: SOTA reproduce (GPU 여유 시)
