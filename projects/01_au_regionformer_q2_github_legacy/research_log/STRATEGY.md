# AU-RegionFormer Q2 — Strategy summary

> 본 프로젝트의 전략 요약. 상세 행동 지침은 `../README.md` 참조.

**Last updated**: 2026-04-22
**Status**: Draft-ready. 후배 1저자로 인계 준비 완료.

---

## Thesis (한 줄)

> **"한국인 FER에서 Region + Geometric landmark + kNN graph의 multi-view orthogonal fusion과 298-person social-consensus-aware filtering으로 training-free 87.56%를 달성한다."**

## Novelty 2축

1. **Social-consensus-aware filtering** — 연세대 컨소시엄 298명 검증 라벨을 label noise 신호로 활용 → +2.78%p
2. **Multi-view orthogonal fusion** — AU intensity는 Region embedding과 redundant (+0.06%p), Landmark geometric은 orthogonal (+2.56%p). 명시적 ablation으로 입증.

### 보조 finding
- Multi-layer Jack 2012 counter-evidence (Region/AU/Landmark 3 layer 일관된 mouth 우위) — reviewer 방어

## Target venue

| Venue | IF | 확률 |
|------|----|-----|
| **ESWA** | 7.5 | 40-55% (primary) |
| **Pattern Recognition Letters** | 5.1 | 50-60% (fallback) |
| Sensors / IEEE Access | 3.4 | 70%+ (safety) |

## 저자 구조

| 역할 | 담당 |
|-----|-----|
| 1저자 | 석사 후배 |
| 2저자 | JY |
| 공저 | 연세대 컨소시엄 측 (데이터 권한) |
| 교신 | Yeon-Kug Moon (Sejong Univ Heart Lab) |

## 현재까지 완료

- 11 iterations (Phase 0.1 ~ Phase 0.3 triplet fusion)
- Leaderboard 15 entries
- Best: triplet (Region+Landmark+kNN) 87.56% linear probe clean subset
- 논문 §3-§5 표/그림 자료 `../results/phase0/` 완비

## 다음 (후배 action)

1. Writing — `../README.md` §3 mapping 참조
2. 연세대 공저 offer — Moon 교수 경유
3. Submit — ESWA primary, PRL fallback

## 참고 맥락 (historical)

본 프로젝트는 초기에 "AI vs Psychology gap + Korean agent" Q1 angle로 기획됐다.
그 중 **FER-specific multi-view fusion + social consensus 부분은 본 Q2**로 확정.
**Agent + cultural priors + bio-grounded labeling 부분은 `projects/02_emotion_agent/`로 분리**.

## 핵심 수치

```
Random baseline:          25.0%
Region POOLED (clean):    84.32%   (exp_002, iter 2)
+ kNN graph:              84.97%   (+0.65, exp_006)
+ Landmark (triplet):     87.56%   (+2.59, exp_009)  ⭐ best
vs v1 trained model (79.7%): +7.86%p
Per-class F1 (triplet):  기쁨 0.949 / 분노 0.840 / 슬픔 0.835 / 중립 0.869
```
