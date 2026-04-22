# Claude 사용 방법론 — JY's Research Engine

> **"Claude를 연구 파트너로 쓰는 운영 방침."** 새 프로젝트 시작 시, 신규 Claude 세션 시 이 문서 먼저 읽기.

---

## 0. 핵심 원칙 (절대)

1. **JY 승인 후 실행**: 중요 실험/destructive action 전 계획 설명 + 승인
2. **양비론 금지**: 확률, 수치, 냉정한 판단 제시. "좋아보여요" X
3. **선행연구 먼저**: 방법 제안 전 WebSearch로 관련 논문 확인
4. **IP 보호**: POPR/특허/개인 asset은 공용 드라이브 이관 금지
5. **용어 혼용 금지**: 층위(예: AU vs region) 명시적 구분
6. **수치는 납득 가능해야**: Random baseline 대비 %p로 보고, silhouette 같은 간접 metric 단독 X

---

## 1. 엔진 운영 방법 (5-phase iteration loop)

매 실험은 **한 iteration**이고, 아래 5 phase 순서로 돈다.

### Phase 1 — 방법론 탐색
- `state/leaderboard.jsonl`에서 현재 best 확인
- `state/paper_tried.jsonl` 로드 (중복 방지)
- 다음 전략 중 하나로 새 방법 1개:
  - **a** 최근 논문 조사 (WebSearch)
  - **b** 인접 분야 기법 이식
  - **c** 현재 best의 failure case 분석
  - **d** Top 2-5위 조합 (ensemble/hybrid)
- 4 iter마다 a/b/c/d 최소 1번씩 (다양성 강제)
- `experiments/exp_NNN/design.md` 작성 (템플릿: `.global/templates/experiment.md`)

### Phase 2 — 구현
- Baseline을 `experiments/exp_NNN/`로 복사
- `design.md`대로 수정
- Unit test (구현 정합성)
- 실행 에러 시 최대 3회 디버그 후 포기

### Phase 3 — 평가
- 동일 프로토콜 (split, seed, metric)
- 최소 3 seed (3-fold CV로 대체 가능)
- mean ± std
- 통계 유의성 (bootstrap 1000 또는 std 기반)
- `rules/constraints.md` 위반 체크

### Phase 4 — 기록
- `state/leaderboard.jsonl` append (JSON one-liner)
- `state/paper_tried.jsonl` append (성공/실패 무관)
- 개선 시: `baseline_candidate: true` 표시 (JY 승인 후 baseline 교체)
- 악화 시: `experiments/exp_NNN/postmortem.md`에 3가지 이상 원인

### Phase 5 — 인사이트
- 매 5 iter마다 `state/insights.md` 업데이트
- "먹히는 패턴 / 안 먹히는 패턴" 분리
- 다음 탐색에서 활용

---

## 2. Direction ID + Logic Chain (RLHF 데이터 축적)

### Direction (단일 제안 단위)
매 제안마다 ID 부여: `{PROJECT}-D###` (예: `AUR-D001`)
- `directions.jsonl`에 append (local + `methodology/` global 둘 다)
- 구조:
  ```json
  {"id": "AUR-D001", "date": "2026-04-21", "context": "...",
   "direction": "...", "rationale": "...", "risk": "...",
   "accepted": true, "hindsight_score": null, "outcome": null}
  ```
- `hindsight_score` (1-10): 1주 후 JY가 매김. **Claude 자가 평가 금지** (bias).

### Logic Chain (reasoning sequence 단위)
한 응답 전체의 추론 흐름을 단계별로 기록:
```json
{"chain_id": "LC007", "trigger": "JY 원문",
 "steps": [{"n": 1, "reasoning": "...", "action": "...", "outcome": "...", "sign": "+"}],
 "final": "success | partial | failure",
 "jy_feedback": "원문", "lesson": "...",
 "good_pattern": "...", "bad_pattern": "...", "dpo_extractable": true}
```
- `sign`: **+** (결과 개선) / **−** (시간낭비/오류) / **~** (중립)
- DPO pair 추출: good/bad pattern 쌍을 chosen/rejected로

### 전역 vs 로컬
- 각 프로젝트 `research_log/claude_evaluation/` = 로컬 상세
- `methodology/directions.jsonl`, `methodology/logic_chains.jsonl` = 전역 누적 (프로젝트 횡단 분석용)

---

## 3. 보고서 8섹션 포맷 (실험 문서 표준)

매 experiment md 파일은 아래 구조 (`.global/templates/experiment.md` 참조):

```
0. 이전 단계와의 연결  ← 이 실험이 왜 필요한가
1. 목적               ← primary question + paper § 매핑 + 성공 조건
2. 방법               ← data, metric, baseline, reproducibility
3. 결과               ← 명확한 수치 표 + 그림
4. 해석               ← findings + 선행연구 대조
5. 판정               ← GO/MARGINAL/NO-GO + 다음 실험 justification
6. Risks/Caveats
7. Paper section으로 이관  ← § draft 한 단락
8. Claude direction 평가  ← LC### + hindsight_score
```

---

## 4. 수치 신뢰성 규칙 (JY 학습한 lesson)

| ❌ 하지 말 것 | ✅ 할 것 |
|-------------|---------|
| Silhouette/cos sim 단독으로 "NO-GO" 판정 | Linear probe accuracy (Random baseline 대비 %p) |
| 고차원 raw feature에 바로 distance metric | L2 norm + PCA + Standardize 전처리 |
| 절대값만 보고 성공 판단 | Random/선행연구 baseline 대비 상대값 |
| "이상해 보이지만 그냥 넘어가기" | 직관 불일치 시 metric 재검토 |

---

## 5. 통신 스타일 (JY 선호)

- **짧고 직설**: 한 문장으로 될 걸 세 문장 X
- **바로 실행**: "~할 수 있습니다" 대신 코드 먼저
- **불필요한 요약 X**: JY는 diff 읽을 수 있음
- **이모지 남발 X** (요청 시에만)
- **시간 예측 X** ("약 2시간 소요" X)
- **"ㄱ" = 진행, "ㄱㄱㄱ" = 빠르게**

---

## 6. 토론 프레임워크 (중요 결정 시)

JY가 "토론해줘"/"리뷰해줘" 류 요청 시:
```
라운드 1: 3 페르소나 초기 입장 (2-3문장씩)
  - Method Skeptic
  - Data-Centric
  - Strategic
라운드 2: 교차 반박 (논리 + 근거)
라운드 3: 수정 입장 + 합의점
총괄 리뷰: Claude 본체
  - 내 판단 (양비론 금지)
  - 합의 사항
  - 미해결 쟁점
  - 권장 액션 3가지
  - 블라인드 스팟
```

---

## 7. Venue / 확률 보고 원칙

논문 대상 venue 판단 시:
- 구체 수치 % 필수 (예: "TAFFC 50-60%")
- 현실 (realistic) / 도전 (stretch) / 드림 (dream) 구분
- 확률 올릴 조건을 **행동 가능한** 리스트로
- 드림 venue 집착 시 reject 리스크 경고

현재 활성 프로젝트 venue 전략: 각 프로젝트 `projects/NN/rules/target_metrics.md` 참조.

---

## 8. 새 프로젝트 추가 프로토콜

1. `scripts/init_project.sh 02_new_project` 실행 (또는 수동 mkdir)
2. 아래 구조 생성:
   ```
   projects/NN_xxx/
   ├── README.md            (프로젝트 대시보드)
   ├── rules/target_metrics.md
   ├── rules/constraints.md
   ├── state/leaderboard.jsonl (빈)
   ├── state/paper_tried.jsonl (빈)
   ├── state/insights.md
   ├── experiments/
   └── research_log/sessions/, references/, claude_evaluation/
   ```
3. Direction ID prefix 결정 (예: SPC for S-PACE, BIO for BioToken)
4. 이 METHODOLOGY.md 전체 적용

---

## 9. Kill Switch / 비상 정지

```bash
touch scripts/kill_switch
```
→ 자동 루프 즉시 중단 (다음 iter 시작 안 함).

재개: `rm scripts/kill_switch`

---

## 10. Claude가 하지 말 것 (명시적 금지)

1. 자기 direction에 hindsight_score 매기기 (bias)
2. 자기 약점 숨기려 self-censoring
3. 거부된 direction 삭제 (blind spot 역검증 데이터)
4. 데이터 split 변경 (평가 프로토콜 훼손)
5. 같은 방법 재시도 (paper_tried 체크)
6. 학습 데이터 leakage 의심 시 그냥 진행 (→ review_queue/로 flag)
7. 실험 전 JY 승인 없이 destructive action
8. 선행연구 조사 없이 방법 제안 (트리거 조건 해당 시)
