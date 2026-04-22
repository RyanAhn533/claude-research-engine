# Project 02 — Emotion Agent (Q1)

> **Culturally-aware multimodal emotion agent with bio-grounded labeling.**
> Q1 논문 (JY 1저자) · 연세대 공저 · Moon 교수 교신
> Direction ID prefix: `EMA-D###`

## Status (2026-04-22)

- Phase: **Pre-setup** (Q1 agent pivot confirmed)
- Iterations: 0 (Week 1 Day 1 대기)
- Best metric: — (baseline 미수립)
- Active goal: **4주 내 reproduction + agent prototype 완성**

## Quick links

| 목적 | 경로 |
|-----|-----|
| **전체 Plan (step-by-step)** | [`ROADMAP.md`](ROADMAP.md) |
| 환경 setup (GPU/LLM/env) | [`setup/SETUP.md`](setup/SETUP.md) |
| Reference repo 매핑 | [`references/REPOS.md`](references/REPOS.md) |
| Target metric | `rules/target_metrics.md` |
| Constraints | `rules/constraints.md` |
| Leaderboard | `state/leaderboard.jsonl` |
| Experiments | `experiments/exp_NNN/` |
| Research log | `research_log/` |

## 3-축 Novelty

1. **Bio-grounded labeling agent** (JY ongoing on another server)
   — bio signal → pseudo-label for multimodal training

2. **Cultural priors** (Jack 2012 + 연세대 298-person consensus)
   — 한국인 cultural context를 agent reasoning에 주입

3. **LLM reasoning loop** (Qwen2.5-7B / Qwen2.5-VL-7B local)
   — perception → reasoning → cultural-conditioned decision

## Target venues

| Venue | IF | 확률 추정 (after Week 4) |
|------|----|----------------------|
| IEEE TAFFC | 11 | 25-35% |
| ICMI / ACII | 5-7 | 40-50% |
| Sensors (safety) | 3.4 | 70%+ |
| NHB/PNAS (dream, 연세대 공저 강화 시) | 21/9.4 | 10-15% |

## Project 01에서 상속받은 자산

| 자산 | 본 프로젝트에서의 활용 |
|-----|--------------------|
| 237K Korean FER + Triplet 87.56% | Perception baseline §Method |
| Jack 2012 multi-layer counter-evidence | Cultural prior 근거 §Discussion |
| Yonsei 298-person consensus filter | Reliability signal §Method |
| Demographic fairness gap | Limitation framing §Limitations |
| SGMT / S-PACE pipeline (bio+behavior fusion) | Perception backbone |

## 저자 구조 (계획)

- 1저자: **JY**
- 공저: **연세대 컨소시엄 심리학자** (데이터 권한 + psych interpretation)
- 교신: **Yeon-Kug Moon** (Sejong University Heart Lab)
