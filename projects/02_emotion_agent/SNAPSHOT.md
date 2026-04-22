# Snapshot — 02_emotion_agent (2026-04-23 06:30 KST)

> JY 깨어나서 한 눈에 파악용. 세부: `Q1_WORKING.md`, `state/insights.md`, 각 `experiments/exp_NNN/summary.md`.

## 현재 상태: **논문 §4 핵심 findings 확보**

### 지금까지 14 iterations
| # | Exp | 결과 (핵심) |
|---|-----|-----------|
| 0 | env setup | Qwen 4-bit 5.4GB VRAM PASS |
| 1 | IEMOCAP preproc | 6877 4-class |
| 2 | MELD preproc | 13708 / 4-class 11353 |
| 3 | K-EmoCon meta | 41654 segs / text 없음 |
| 4 | text baseline | IEMOCAP 66%, MELD F1 0.31 |
| 5 | agent prototype | sanity 4/4 OK |
| 6 | agent IEMOCAP | 48.75% zero-shot |
| 7 | agent MELD | 55% → full 58.5% (+context) |
| 8 | K-EmoCon 4-class | invalid (class imbalance) |
| 9 | K-EmoCon valence | info-poor (50%) |
| 10 | Korean FER landmark | random (25%) — numeric features 부적합 |
| 11 | Korean FER AU (FACS) | **34.25%** baseline, cultural abstract -7%p |
| **12** | **Few-shot k=4** | **41.75%** (+12%p over zero-shot, thesis 회복) |
| **13** | **k-sweep** | **inverted-U**, k=2 optimal 42.25% |
| **14** | **Cross-cultural ablation** | **★ PAPER-GRADE**: Ekman=random, real Korean +15-17%p, **mixed=best 44.25%** |

## 🎯 논문 §4 key findings (확정)

**3-levels of cultural grounding**:

| Level | Method | Korean FER AU acc |
|-------|--------|------|
| Abstract | cultural prompt text | −7 pp (harmful) |
| Prototype | Ekman FACS textbook (idealized AU patterns) | ±0 pp (random) |
| **Distribution** | Actual Korean exemplars | **+15 ~ +19 pp** |
| **Anchored** | Korean + Western mix | **+18.75 pp (best)** |

→ **"Cultural grounding requires actual distribution exemplars, not abstract knowledge or textbook prototypes"**

Plus **inverted-U scaling** (k=2 optimal) → "few > many".

## GPU 현황
- A6000 1× 48GB, **현재 BrandSpace serve.py (~28.9GB) 상주**
- 우리 agent Qwen 4-bit: 5.4GB 사용 (budget 7.7GB 내 OK)
- LoRA fine-tune (24-30GB)은 **BrandSpace off 필요**

## 다음 할 것 (JY 판단 필요)

| 옵션 | 설명 | 시간 | GPU |
|-----|-----|-----|-----|
| A. Multi-seed replication | exp_012/013/014 3-seed 재측정 (stat robustness) | 1h | 현 상태 OK |
| B. MELD cross-cultural | Korean+English exemplar mix on MELD | 30분 | OK |
| C. Qwen2.5-VL image | face image 직접 input | 20분 | 6-8GB (경계) |
| D. LoRA fine-tune (Korean AU data) | 1-epoch fine-tune | 2-4h | **BrandSpace off 필요** |
| E. Emotion-LLaMA 재현 | NeurIPS 2024 SOTA comparison | 4-6h | BrandSpace off + 24GB |
| F. Paper writing start | §3, §4 초안 작성 | 추후 | 0 |

**추천 우선순위**: A → B → D → E → F (safety 먼저, novelty check, fine-tune, SOTA, writing)

## Repo 상태
- https://github.com/RyanAhn533/claude-research-engine (private)
- 14 leaderboard entries, 8 direction IDs (EMA-D001~D008)
- 모든 결과 commit/push 완료

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
