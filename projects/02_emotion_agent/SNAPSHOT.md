# Snapshot — 02_emotion_agent (2026-04-23 14:55 KST)

> JY 깨어나서 한 눈에 파악용. 세부: `Q1_WORKING.md`, `state/insights.md`, 각 `experiments/exp_NNN/summary.md`.

## 현재 상태: **3-tier story 완성 (prompt 29% → ICL 42% → LoRA 57%)**

### 17 iterations (14 single-seed + 3 multi-seed replication)
| # | Exp | 결과 (핵심) |
|---|-----|-----------|
| 0 | env setup | Qwen 4-bit 5.4GB VRAM PASS |
| 1-3 | preproc | IEMOCAP 6877, MELD 13708/11353, K-EmoCon 41654 |
| 4 | text baseline | IEMOCAP 66%, MELD F1 0.31 |
| 5-7 | agent eval | IEMOCAP 48.75%, MELD 58.5% |
| 8-10 | K-EmoCon/FER null | class imbalance / info-poor / numeric features |
| 11 | Korean FER AU (FACS) | 34.25% baseline, cultural abstract −7%p |
| 12 | Few-shot k=4 (single-seed) | 41.75% (+12%p) |
| 13 | k-sweep (single-seed) | inverted-U, k=2 optimal 42.25% |
| 14 | Cross-cultural (single-seed) | Ekman=25.25, real Korean +15-17%p, mixed=44.25% |
| **15** | **exp_012 multi-seed** | baseline **31.83 ± 1.70** / fewshot_k4 **41.83 ± 3.41** / Δ=+10pp |
| **16** | **exp_013 multi-seed** | k=0 **29.08±0.76** / k=2 **44.25±2.82** / k=4 42.75±5.02 / k=8 43.50±1.52 / k=16 42.33±1.84 |
| **17** | **exp_014 multi-seed** | **Ekman 25.42±0.38** (random, very stable) / **Mixed 41.25±0.87** (BEST, stable) / Kor 41.00±4.34 |
| **18** | **exp_015 QLoRA r=16** (seed=42) | **56.75% / F1 0.573** (seed=42 baseline). Adapter 170MB. 30min 훈련. |
| **19** | **exp_015 multi-seed LoRA** | **55.00 ± 2.61% / F1 0.555 ± 0.026** (seeds 42/123/777: 56.75/52.00/56.25). 3-tier ROBUST. |
| **20** | **exp_016 anchor-ratio** | **Two-effect finding**: Kor presence → MEAN (k=1 충분), Wes ≥2 → VARIANCE (std 4-6 → <1). Paper §4.4 novel. |

## 🎯 논문 §4 key findings (multi-seed robust)

### (i) 3-level cultural grounding — **reviewer-defensible**

| Level | Method | Korean FER AU acc (3-seed mean±std) |
|-------|--------|------|
| Abstract | cultural prompt text | −7 pp (single-seed from exp_011) |
| **Prototype** | Ekman FACS textbook | **25.42 ± 0.38%** — indistinguishable from random (25%) |
| **Distribution** | Korean real k=4 exemplars | **41.00 ± 4.34%** |
| **Anchored** | Korean 2 + Western 2 mix | **41.25 ± 0.87%** (lowest variance, highest mean) |

→ Δ (Ekman → Mixed) = **+15.83 pp**, combined σ tiny — statistically significant.
→ **"Cultural grounding requires actual distribution samples, not textbook prototypes"** ROBUST.

### (ii) Exemplar-scaling: **saturation, not inverted-U**

Multi-seed reveals original "k=16 drop" (36%) was seed artifact. True picture:

| k | mean ± std | vs zero-shot (29.08%) |
|---|-----------|-----|
| 0 | 29.08 ± 0.76 | — |
| 2 | 44.25 ± 2.82 | +15.17 pp |
| 4 | 42.75 ± 5.02 | +13.67 pp (noisy) |
| 8 | 43.50 ± 1.52 | +14.42 pp |
| 16 | 42.33 ± 1.84 | +13.25 pp |

→ **REVISED claim**: "k≥2 exemplars give +13–15 pp robust gain; saturation beyond k=2 (not collapse)". Paper §4.2 needs rewording.

### (iii) 3-tier adaptation hierarchy — **multi-seed robust §4.3**

| Tier | Method | Acc (mean±std, n=3) | F1 | Δ |
|------|--------|---------------------|-----|-------|
| 1. Prompt only | zero-shot FACS | 29.08 ± 0.76% | 0.194 ± 0.011 | — |
| 2. ICL | fewshot k=4 Korean | 41.83 ± 3.41% | 0.364 ± 0.040 | +12.75 pp |
| **3. LoRA** | **QLoRA r=16, 10K, 1 ep** | **55.00 ± 2.61%** | **0.555 ± 0.026** | **+13.17 pp** |
| total | zero-shot → LoRA | | | **+25.92 pp** |

Per-seed LoRA: 42→56.75, 123→52.00, 777→56.25. σ=2.61, lower bound (52.4) > ICL upper (45.2) → **tiers are statistically separated**.

→ Thesis: **cultural grounding benefits compound across adaptation levels**. Each tier gives +13pp robust gain. Total headroom vs zero-shot = **+25.92pp**.
→ Paper §4.3 headline chart ready.

### (iv) Korean-Western anchor ratio — **§4.4 novel finding** (exp_014 + exp_016 combined)

Full 5-point ratio sweep (k=4 total, 3 seeds each):

| k=Wes (of 4) | Acc mean ± std | σ regime |
|--------------|----------------|----------|
| 0 (4+0 Korean only) | 41.00 ± **4.34** | HIGH |
| 1 (3 Kor + 1 Wes) | 41.50 ± **6.00** | HIGH |
| 2 (2+2 Mixed) | 41.25 ± **0.87** | LOW |
| 3 (1 Kor + 3 Wes) | 40.08 ± **0.80** | LOW |
| 4 (Western only, Ekman) | 25.42 ± 0.38 | LOW (but low mean) |

**Two independent effects observed**:
1. **MEAN ∝ Korean presence** — even a single Korean exemplar gives +15pp over zero-shot (41.5% at 3+1, 40.1% at 1+3). Korean count within [1,4] doesn't matter much.
2. **VARIANCE ∝ Western count (threshold k_W ≥ 2)** — std collapses from 4-6pp to <1pp when at least 2 Western anchors are present.

**Paper §4.4 claim**: "Effective cultural grounding requires dual design — target-culture exemplars (even k=1) for mean accuracy, plus ≥2 diverse anchors for stability across exemplar sampling."

Unique vs existing ICL literature: most work treats exemplars as monolithic. This decouples the roles of target-culture vs anchor-culture examples.

## GPU 현황
- A6000 1× 48GB, **현재 BrandSpace serve.py (~28.9GB) 상주**
- 우리 agent Qwen 4-bit: 5.4GB 사용 (budget 7.7GB 내 OK)
- LoRA fine-tune (24-30GB)은 **BrandSpace off 필요**

## 다음 할 것 (JY 판단 필요)

| 옵션 | 설명 | 시간 | GPU | 상태 |
|-----|-----|-----|-----|------|
| A. Multi-seed replication | exp_012/013/014 3-seed 재측정 | 3h 실제 | OK | ✅ DONE |
| B. MELD cross-cultural | Korean+English exemplar mix on MELD | 30분 | OK | pending |
| C. Qwen2.5-VL image | face image 직접 input | 20분 | 6-8GB | pending |
| D. LoRA fine-tune (Korean AU data) | exp_015 draft ready (QLoRA r=16) | 2-4h | 48GB free ✓ | **다음** |
| E. Emotion-LLaMA 재현 | NeurIPS 2024 SOTA comparison | 4-6h | 24GB | pending |
| F. Paper writing | §3, §4 multi-seed로 재기술 | - | 0 | ongoing |

**다음 실행**: D (LoRA, peft 0.19.1 + accelerate 1.13.0 설치 완료, exp_015/run.py ready)

## Repo 상태
- https://github.com/RyanAhn533/claude-research-engine (private)
- 17 leaderboard entries (multi-seed 3개 추가)
- exp_012/013/014 multi-seed 모두 commit, exp_014 push 대기

## 파일 위치 (자주 쓸 것)
```
projects/02_emotion_agent/
├── Q1_WORKING.md                 ← 논문 drafting
├── ROADMAP.md                    ← 4주 plan
├── SNAPSHOT.md                   ← 이 파일
├── state/leaderboard.jsonl       ← 모든 수치
├── state/insights.md             ← Week별 회고
├── experiments/exp_NNN/          ← 각 실험 결과 + summary.md
└── src/agent/emotion_agent.py    ← agent 코드
```
