# Target Metrics — Q1 논문 관점

**최종 목표**: IEEE TAFFC (IF 11) 수용 / NHB dream. Thesis = "AI vs Psychology gap in Korean facial emotion".

**자동화 루프가 올리려는 것**: 논문 §4 Results의 수치 품질. 단순 accuracy가 아님.

---

## Primary metric (이걸 올려야 iteration 성공)

| Rank | Metric | 현재 best | 목표 | 논문 section |
|-----|--------|---------|-----|-----------|
| 1 | **per-class Macro F1** (4-class) on Phase 0.1c linear probe POOLED | 0.815 (ConvNeXt) | ≥0.83 | §4.1 |
| 2 | **per-class F1 floor** (최악 class; 현재는 미측정) | TBD | ≥0.70 | §4.1 |
| 3 | **Jack 2012 reversal effect size** (mouth-AU − eye-AU acc %p) | +1.2%p (AU), +13.5%p (region) | ≥+10%p (AU level) | §4.3 핵심 Figure |
| 4 | **FACS canonical reproduction** (Happy=AU6+12 top-rank 재현, Sad=AU1+4+15 재현) | Happy OK / Sad 부분 | 3 emotion 모두 top-rank | §4.4 |

---

## Secondary metrics (놓치지 않게 같이 보기)

| Metric | 의미 |
|--------|-----|
| Calibration (ECE) | confidence 정확도 |
| Per-demographic fairness gap | 성별/나이별 성능 차이 (Section §4.5 fairness) |
| AU detector validation on KUFEC-II | OpenGraphAU 한국인 정확도 |
| 298-person 검증 샘플 accuracy | 논문 §4.5 사회적 합의 signal |

---

## 명확히 "개선"으로 인정하는 기준

- Primary rank 1 metric이 **절대값 +0.01 이상 증가 AND p < 0.05** (bootstrap 1000, 3-seed mean)
- Secondary metric 악화 ≤ 2% (Pareto 악화 금지)
- Constraint 전부 통과 (`rules/constraints.md`)

"개선 아님"으로 보는 경우:
- 절대값 증가했으나 p ≥ 0.05
- Primary 개선 대신 Secondary 악화 (trade-off로 분류)
- 데이터 split 변경 (평가 프로토콜 변경) → **자동 실격**

---

## 종료 조건

- **target_reached**: Primary 1 + 2 + 3 모두 목표 도달
- **plateau**: 20 iteration 연속 개선 없음 → JY review_queue로 push
- **pivot triggered**: Primary 3 (Jack reversal)이 2 iteration 연속 음수 → thesis 재검토
