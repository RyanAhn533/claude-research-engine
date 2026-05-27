---
description: CONDITIONAL_SELF_ATTACK — paper claim-affecting 실험에만 실행
allowed-tools: Read, Bash(python:*)
argument-hint: [project-name] [exp-id-suffix]
model: opus
---

## 트리거 검사 먼저

`projects/$1/state/leaderboard.jsonl`에서 `exp_$2` 행을 읽고, 그 `linked_claims`에 해당하는 claim 행을 `claim_registry.jsonl`에서 찾는다.

다음 중 **하나라도** 참이면 SELF_ATTACK 실행. 아니면 즉시 종료 ("not required").

- `claim.required_for_submission == true`
- `experiment.baseline_candidate == true`  
- `experiment.method` 가 "first to ..." 같은 novelty 주장 포함
- reviewer_risk_score >= medium (JY가 design.md에 명시)
- 직전 narrative와 모순되는 결과

## 3 페르소나 병렬 실행

3개 subagent **반드시** 본 컨텍스트에서 격리 (디자인에 오염되지 않게):

### Subagent 1: Method Skeptic
- 입력: `experiments/exp_$2/design.md` + `results.json`
- 임무: hyperparameter 선택, baseline 선정, 통계 처리, 데이터 분할의 모든 의심점 공격
- thinking budget: **max**
- 출력: < 2K tokens, structured concerns 리스트

### Subagent 2: Reviewer Simulator
- 입력: 위와 동일 + 해당 claim 문장
- 임무: ICLR/NeurIPS reviewer 페르소나로 critique. severity별 분류. expected `review_score / 10` 추정.
- thinking budget: **max**
- 출력: severity high/medium/low로 분류된 concerns

### Subagent 3: Novelty Critic
- 입력: 위와 동일 + 최근 paper (WebSearch 가능)
- 임무: "X et al. (2024)" 패턴으로 선행연구 끌어와서 novelty 주장 반박
- thinking budget: **max**
- 출력: 인용 후보 + 어디서 novelty가 약한지

## 결과 처리

세 subagent 출력을 `state/gate_log.jsonl`에 task 행으로 추가 (`gate: "D"`, `is_advisory: true`).

`leaderboard.jsonl`의 해당 exp 행은 **수정하지 말고**, `self_attack_concerns` 필드를 채운 새 행을 `supersedes`로 추가.

## 절대 금지
SELF_ATTACK은 advisory만 한다. **차단 권한 없음**. 어떤 verdict가 나와도 LOGGING은 그대로 진행됨. 다만 JY가 보고 baseline_candidate 승격 여부를 결정.
