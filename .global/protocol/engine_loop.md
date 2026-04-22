# Engine Loop Protocol

자동화 iteration 상세. `METHODOLOGY.md` Section 1의 확장판.

## 시작 조건
- `projects/NN/rules/target_metrics.md` 존재
- `projects/NN/rules/constraints.md` 존재
- `projects/NN/state/leaderboard.jsonl` 존재 (비어도 OK)
- `projects/NN/baseline/` 존재

## 매 iter 5-phase

### Phase 1 Design (5-15 min)
**Input**: leaderboard, paper_tried, insights, target_metrics
**Output**: `experiments/exp_NNN/design.md`

- 현재 best run 확인
- 다양성 규칙 체크 (최근 10 iter 내 동일 category 3회 초과 금지)
- 전략 a/b/c/d 4 iter 단위로 최소 1번씩
- Primary metric 예상 변화 + 근거
- 실패 fallback

### Phase 2 Implement (30 min - 2h)
**Output**: `experiments/exp_NNN/run.py` (또는 기존 baseline copy-modify)

- Unit test 작성 (sanity)
- 디버그 3회 초과 시 포기 + postmortem

### Phase 3 Evaluate (실험 수행 시간)
**Output**: `experiments/exp_NNN/results.json`, `summary.md`

- `rules/constraints.md` 평가 프로토콜 엄수
- 3 seeds or 3-fold CV minimum
- p-value vs current best
- 제약 통과 여부

### Phase 4 Record (5 min)
**Output**: leaderboard/paper_tried append

- `leaderboard.jsonl` 한 줄 JSON append (see schema below)
- `paper_tried.jsonl` 한 줄 JSON append
- Global `methodology/directions.jsonl`에 iter의 direction append (prefix ID)
- 개선 시: `baseline_candidate: true` (JY 승인 후 baseline 교체)

### Phase 5 Insights (매 5 iter, 10 min)
**Output**: `state/insights.md` 업데이트
- 먹히는 / 안 먹히는 패턴
- Meta-pattern 업데이트

## JSON schema

### leaderboard entry
```json
{"exp_id": "exp_NNN_name",
 "iteration": N,
 "method": "human-readable description",
 "paper_ref": "citation or 'internal'",
 "feature": "input feature description",
 "metric_name": "accuracy_4class | macro_f1 | ...",
 "metric_mean": 0.XX,
 "metric_std": 0.XX,
 "macro_f1_mean": 0.XX,
 "p_value_vs_best": "<0.001 | ...",
 "constraints_passed": true,
 "n_samples": 30000,
 "n_folds": 3,
 "subset": "full | clean | rejected | mixed",
 "baseline_candidate": false,
 "timestamp": "2026-MM-DDTHH:MM:SS",
 "notes": "1-liner key insight",
 "findings": {...optional detail...}}
```

### paper_tried entry
```json
{"paper_ref": "Author Year venue",
 "method": "what we tried",
 "outcome": "success | partial | failed — why",
 "exp_id": "exp_NNN",
 "iteration": N,
 "timestamp": "..."}
```

### direction (local + global)
```json
{"id": "{PROJECT}-D###",
 "date": "...",
 "session": "session_file_ref",
 "context": "when/why",
 "direction": "what claude proposed",
 "rationale": "why",
 "risk": "what could go wrong",
 "accepted": true | false | "deferred",
 "hindsight_score": null,
 "outcome": null}
```

### logic chain
```json
{"chain_id": "LC###",
 "date": "...",
 "session": "...",
 "trigger_type": "question | proposal | correction | feedback",
 "trigger": "JY 원문",
 "steps": [{"n": 1, "reasoning": "...", "action": "...", "outcome": "...", "sign": "+|-|~"}],
 "final": "success | partial | failure",
 "jy_feedback": "원문",
 "lesson": "...",
 "good_pattern": "...",
 "bad_pattern": "...",
 "dpo_extractable": true | false}
```

## 종료 조건

- **target_reached**: `target_metrics.md` Primary 전부 달성 → `state/target_reached.md` 생성
- **plateau**: 20 iter 연속 개선 없음 → `state/plateau.md` + `review_queue/` push
- **kill_switch**: `scripts/kill_switch` 파일 존재 → 현재 iter 종료 후 즉시 중단
- **review_queue ≥ 2**: 미처리 flag 2개 이상 → 자동 루프 일시정지

## 다양성 강제

최근 10 iter 내:
- 동일 **category** (data / architecture / feature_engineering / probe_capacity / analysis / ensemble) 연속 3회 초과 금지
- 동일 **저자/그룹** 논문 연속 2회 초과 금지
- 매 4 iter마다 전략 a/b/c/d 최소 1번씩 사용 확인
