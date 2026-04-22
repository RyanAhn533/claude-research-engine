# Report Format — 8-Section Standard

모든 실험 문서 (`experiments/exp_NNN/design.md` 또는 완료 후 문서)는 이 포맷.

```
0. 이전 단계와의 연결
   이전 실험의 어떤 질문에 답하는가? 왜 이 실험이 독립적으로 필요한가?

1. 목적
   - Primary question (yes/no 또는 quantity로 답 가능)
   - Paper section 매핑 (§X.X)
   - 성공 조건 (numeric threshold + 다음 액션)

2. 방법
   - Data (규모, 출처, preprocessing)
   - Metric (primary, secondary — 계산 방법 + 해석 범위)
   - Baseline (Random, 선행, Oracle)
   - Reproducibility (seed, CV, sample)

3. 결과 (명확한 수치)
   - 핵심 표 (config × metric + p-value)
   - 그림 경로

4. 해석
   - Finding 1, 2, 3
   - 선행연구와의 관계 (일치/불일치)
   - 예상 vs 실제 비교

5. 판정
   - GO / MARGINAL / NO-GO
   - 다음 실험 justification

6. Risks / Caveats (Risk × Impact × Mitigation)

7. Paper section으로 이관
   - §X.X Draft (논문용 한 단락)
   - Figure/Table 지정

8. Claude direction 평가
   - Direction ID, Logic chain ID
   - hindsight_score (1주 후)
   - lesson
```
