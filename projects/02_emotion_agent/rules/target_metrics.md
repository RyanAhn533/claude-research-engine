# Target Metrics — Emotion Agent Q1

## 최종 목표
IEEE TAFFC (IF 11) submit 가능 수준 / stretch NHB/PNAS.

## Primary metric (이걸 올려야 iteration 성공)

| Rank | Metric | 현재 best | 목표 | 논문 section |
|-----|--------|---------|-----|-----------|
| 1 | **Macro F1 on IEMOCAP** (agent vs SOTA) | — | SOTA ±2%p 이상 | §4.1 |
| 2 | **Macro F1 on MELD** | — | SOTA ±2%p | §4.1 |
| 3 | **Cultural prior ablation effect** | — | 통계 유의 (p<0.05), ≥ +1%p | §4.3 (핵심 novelty) |
| 4 | **Bio-grounded label agreement** (human κ) | — | κ > 0.5 | §4.2 |

## Secondary metrics

| Metric | 의미 |
|-------|-----|
| Per-class F1 | class balance |
| Calibration (ECE) | agent confidence 신뢰도 |
| Per-demographic fairness gap | §Limitations 방어 |
| Reasoning quality (qualitative sample review) | LLM agent 고유 |
| Inference latency per sample | 실용성 / Jetson 배포성 |

## "개선" 기준
- Primary rank 1 or 2: 절대값 **+1%p 이상 AND p < 0.05** (3-seed mean)
- Secondary 악화 ≤ 2%
- Constraint 전부 통과

## 종료 조건
- **target_reached**: Primary 1 + 2 + 3 모두 도달
- **plateau**: 20 iter 연속 개선 없음 → `state/plateau.md`
- **pivot**: Week 3 gate 미달 3회 → scope 축소 (4 benchmark → 2)
