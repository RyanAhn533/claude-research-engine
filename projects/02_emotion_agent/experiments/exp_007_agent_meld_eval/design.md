---
exp_id: exp_007_agent_meld_eval
phase: week2_day4
status: planned
---

# Agent MELD test eval (official split)

## 목적
MELD official test (2211 utt 중 stratified 400) 에서 agent baseline vs +cultural vs +prosody vs full ablation.
exp_006 (IEMOCAP) 과 동일 format — 두 dataset 결과 비교 table 완성.

## Notes
- MELD는 **대화 맥락** 중요 (conversation context). 현재 prototype은 utterance-level. Context 추가 옵션 실험 가능
- Class imbalance 심함 → weighted eval + per-class F1 필수
- Prosody feature MELD parquet에는 없음 (IEMOCAP만 있음). **prosody-only variant skip**
- Cultural prior는 "Korean" 특화인데 MELD는 영어 TV show. Cultural 효과 drop 예상

## Variants to test
1. baseline (no cultural, no context)
2. +cultural prior (Korean)
3. +conversation context (previous 2 utterances in prompt)
4. full (+cultural +context)

## Expected
- MELD text baseline (exp_004) 59.75% → agent prompt-only ~55-62% 예상
- Context 추가 시 +3-5%p 기대
- Cultural prior는 MELD에서 효과 낮음 (Korean 가정 불일치) — **ablation 정직하게 보고**

## Execution
- 400 samples × 4 configs = 1600 calls
- Qwen 4-bit, ~400ms/sample → ~11분
