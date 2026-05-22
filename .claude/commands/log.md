---
description: Phase 5 LOGGING — leaderboard, paper_tried, claim, direction 동시 append
allowed-tools: Read, Bash(python:*), Bash(git:*)
argument-hint: [project-name] [exp-id-suffix]
---

실험 `exp_$2`가 끝났다. 다음을 순서대로 append-only로 기록한다.

## 사전 조건 (이게 없으면 LOGGING 진입 금지)
1. `projects/$1/reproducibility_manifests/exp_$2.yaml` 존재
2. `experiments/exp_$2/results.json` 존재
3. Gate B (leakage audit + smoke test) 통과
4. Gate C (stage에 맞는 통계) verdict 결정됨

## 행 추가 순서

### 1) leaderboard.jsonl
- schema `experiment.schema.json`
- `paired_delta_ci_95`, `cohen_d_paired`, `wilcoxon_p` 채우기 (paper_ready 단계면 필수)
- `linked_claims` 채우기 — 없으면 어느 paper claim에도 연결 안 된 실험이라는 뜻
- `manifest_ref` = `reproducibility_manifests/exp_$2.yaml`

### 2) paper_tried.jsonl
- schema `paper_tried_entry.schema.json`
- `method_id` + `config_fingerprint` 쌍이 중복이면 거부됨 (의도된 동작)
- 실패면 `failure_category` 정확히 (syntactic/semantic/dynamics/protocol)
- `method_globally_blocked: true`는 **절대 자동 설정 금지** — JY만 가능

### 3) hypothesis_registry.jsonl (해당 hypothesis_id의 outcome 필드 업데이트는 **새 행 append**)
- 원래 행을 수정하지 말고, `supersedes`를 채운 새 행 추가
- `outcome.result`: supported / rejected / neutral
- `outcome.observed_delta`: 실제 측정값
- `outcome.tost_passed`: TOST가 통과했는지

### 4) claim_registry.jsonl
- 해당 claim의 `supporting_exps` 또는 `contradicting_exps`에 exp_id 추가
- 마찬가지로 새 행 + supersedes

### 5) methodology/directions.jsonl
- prefix는 `rules/direction_prefix.txt`에서 읽음
- 이번 사이클에 너가 제안했던 모든 direction 기록
- `hindsight_score`는 **절대 너가 채우지 마** (JY 전용)

## 마지막
```bash
python -m engine.cli.jy verify-chain --project $1 --log leaderboard
python -m engine.cli.jy verify-chain --project $1 --log paper_tried
```
체인 깨지면 즉시 보고.

## git
JY 명시적 동의 없이 `git push` 금지. local commit은 OK.
