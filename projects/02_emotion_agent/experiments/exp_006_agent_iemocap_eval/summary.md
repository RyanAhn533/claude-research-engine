# exp_006 — Agent IEMOCAP eval (400 samples, 4 configs)

## Setup
- Model: Qwen2.5-7B-Instruct 4-bit (VRAM 5.4GB)
- Test: 400 stratified (100/class) from IEMOCAP 4-class (seed 42)
- Inference: batch 1, ~600-750 ms/sample
- Total: 4 configs × 400 = 1600 calls, ~18 min

## Results

| Config | Acc | Macro-F1 |
|--------|-----|----------|
| baseline (no cultural, no prosody) | **48.75%** | 0.484 |
| +cultural (Korean prior) | 48.50% | 0.485 |
| +prosody only | 48.00% | 0.479 |
| +cultural +prosody (full) | 48.25% | 0.485 |

Random baseline: 25%.
Text-only TF-IDF+LogReg (exp_004, 3-seed 80/20): **66.33% ± 0.84** (학습 있는 baseline).

## Interpretation

1. **Zero-shot Qwen ~48%** (Random +23%p). 학습 없는 LLM prompt-only의 한계.
2. **Cultural prior (Korean)가 IEMOCAP에서는 invariant (±0.25%p)**. 예상대로 — IEMOCAP은 English TV-actor dataset, Korean 맥락 아님.
3. **Prosody injection도 효과 없음 또는 약간 감소**. Dict→text 단순 표현이 LLM reasoning에 충분 활용 안 됨.

## Why this is actually good for the paper

- Cultural prior가 **cultural-match 조건에서만** 효과 있다는 예측. IEMOCAP(English)에서 null effect = 우리 thesis 간접 지지.
- **Korean data (K-EmoCon)에서 cultural prior effect 검증이 결정적**. 이걸 exp_008에서 진행.
- Text-only TF-IDF+LogReg보다 낮은 zero-shot 수치는 정직하게 limitation으로 reporting (fine-tune 여유 있으면 상회 기대).

## Next
- exp_007: MELD (English 예상, 유사 null effect 예상)
- exp_008: **K-EmoCon (Korean)** — cultural prior effect 결정적 검증
- Week 3: fine-tune 여유 되면 LoRA (GPU 48GB 확보 필요 — BrandSpace kill)
