# Insights — Project 02 Emotion Agent

## Week 1 (preprocessing) — PASS
- IEMOCAP 6877 4-class (HF arrow direct)
- MELD 13708 utt / 4-class 11353 (official split, video present)
- K-EmoCon 41654 segments / 4187 aggregated external (text absent, bio raw)
- Qwen2.5-7B 4-bit: 5.4GB VRAM PASS

## Week 2 (agent prototype + ablation) — MIXED/HONEST NEGATIVE

### 핵심 finding 표 (cultural prior Δ)

| Dataset | Task | Baseline | +Cultural | Δ | Interpretation |
|---------|------|----------|-----------|---|----|
| IEMOCAP (English) | 4-cls utterance | 48.75% | 48.50% | −0.25 | null (expected; English ≠ Korean) |
| MELD (English) | 4-cls | 55.00% | 55.75% | +0.75 | weak positive |
| **MELD + context** | | 55.00% | **58.50%** | **+3.50** | synergy w/ context |
| K-EmoCon (Korean) | 4-cls 4187 agg | 9.57% | 5.22% | — | invalid (sad=0, neu=0) |
| K-EmoCon | valence binary | 50% | 50% | 0 | info-poor (arousal only) |
| Korean FER (landmark) | 4-cls | 25.50% | 25.25% | −0.25 | numeric features → LLM 의미불명 |
| **Korean FER (AU FACS)** | 4-cls | **34.25%** | **27.25%** | **−7.00** | agent 혼란 (negative) |

### 먹히는 패턴 ✅
- Qwen 4-bit 안정 (5.4GB, inference 600-700ms/sample)
- MELD context injection +3.5%p (conversation 맥락은 작동)
- Agent macro-F1 > TF-IDF F1 on MELD (0.58 vs 0.31, class imbalance robust)
- AU 기반 input이 landmark raw보다 훨씬 나음 (34% vs 25%) — FACS standard는 LLM이 학습 data에서 접함

### 안 먹히는 패턴 ❌
- Prompt-only cultural prior는 inconsistent — 7 variants 중 2개만 positive
- Raw numeric features → LLM: semantic interpretation 불가
- Zero-shot LLM 성능 < trained text baseline (IEMOCAP: 48 < 66)
- K-EmoCon는 text 없음 + class imbalance → current agent 적용 부적합
- Cultural prior 길이 긴 prompt = negative effect 가능 (Korean FER AU에서 -7%p)

### Thesis 수정 (honest)
원래: "Korean cultural prior injection으로 agent 성능 개선"
수정: **"Prompt-only cultural prior는 unreliable. Fine-tune 기반 cultural alignment 필요"**

### Week 3 방향 (3 options)
A. **LoRA fine-tune** on Korean 4-class FER AU data — BrandSpace off + 48GB 필요
B. **Few-shot in-context** Korean examples — 별도 fine-tune 없음
C. **Retrieval-augmented** (RAG with Korean emotion exemplars)

### Blind spot recognized
- Korean text+emotion dataset 부재가 큰 제약
- IEMOCAP/MELD는 English (cultural prior 직접 적용 못 함)
- Thesis "Korean-specific agent" 검증이 제한적
- Project_01 FER (이미지 데이터)만으로는 cultural prior 효과 분리 힘듦
