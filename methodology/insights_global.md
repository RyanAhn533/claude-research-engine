# Global Insights — Cross-project patterns

프로젝트 횡단 Claude 방법론 패턴. 매 프로젝트 3 iter 이상 진행 후 업데이트.

## 확정 패턴 (2026-04-22 기준, 1 프로젝트만 활성)

### ✅ 먹히는 패턴
1. **Linear probe가 embedding 평가 gold standard** (silhouette, cos sim 단독 X)
2. **Random baseline 대비 %p 보고** (0.X 절대값보다 "random 25% 대비 +Y%p" 해석 쉬움)
3. **3인 페르소나 토론 (Method Skeptic + Data-Centric + Strategic)**
4. **WebSearch 병렬 선행연구 조사** — 실험 결과를 선행연구와 대조하면 thesis 강해짐
5. **Subset filter 비교** (raw vs clean vs rejected) — ablation + interpretation 동시
6. **Per-group matrix** (category × condition) — paper table/figure 직접 생성
7. **Negative finding도 기록** (redundancy, gap 없음) → 다음 Phase 방향 결정

### ❌ 안 먹히는 패턴
1. **고차원 raw feature에 직접 silhouette/cosine distance** (고차원 저주 + L2 norm 누락)
2. **Path/data 검증 미실시 시 class imbalance 방치**
3. **Python env dependency 순차 설치** (user-level 충돌 놓침 — PYTHONNOUSERSITE=1 + --force-reinstall 첫시도부터)
4. **도메인 용어 층위 혼용** (예: AU vs face region)
5. **"문서 만들어" 시 목적 미확인** (snapshot HTML vs live working doc 구분 필요)
6. **단순 concat으로 ensemble 시도** — orthogonal 아닌 feature는 redundant

### 공통 원칙
- **절대값 + 상대값 동시 보고**
- **실수는 즉시 admit** (변명 X)
- **논문 section 매핑 없는 실험 취소**
- **직관이 "이상하다"고 할 때 metric 재검토**

## Venue 판단 메타-패턴

- JY의 "충분해?" 질문 = dream pitch 거부 신호
- 매 설계 변경마다 % 체크포인트 제공
- 현실/도전/드림 구분 명시
- 드림 venue 집착 시 reject 리스크 경고

## 프로젝트별 link

- [01 AU-RegionFormer Q1](../projects/01_au_regionformer_q1/state/insights.md)
