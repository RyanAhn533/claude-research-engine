---
description: v2 엔진 한 iteration 실행 (HUMAN_APPROVAL까지 진행 후 정지)
allowed-tools: Read, Edit, Write, Bash(python:*), Bash(git:*), Grep, Glob, WebSearch
argument-hint: [project-name] [exp-id-suffix]
model: opus
---

프로젝트 `projects/$1`에서 실험 `exp_$2` 한 사이클을 다음 순서로 진행해라.

## Phase 진행

### 0) STATUS_SYNC
- `python -m engine.cli.jy status --project $1` 출력 읽기
- 실제 파일 시스템 상태가 HANDOFF.md와 모순되면 **파일 상태를 신뢰**
- `kill_switch_present=true` → 즉시 중단

### 1) METHOD_SEARCH (Plan mode 필수)
- Plan mode로 진입 (read-only)
- 5개 strategy 중 최소 3개를 subagent로 병렬 실행:
  - a. WebSearch — 최근 논문 (최우선)
  - b. 인접 도메인 transplant
  - c. failure-case 분석 (`negative_results/` 읽기)
  - d. top-k hybrid
  - e. first-principles redesign
- `paper_tried.jsonl`을 `(method_id, config_fingerprint)` 쌍으로 dedup
- 후보 3개를 `experiments/exp_$2/design.md`로 출력

### 1.5) HYPOTHESIS_REGISTRATION
- 각 후보마다 `hypothesis_registry.jsonl`에 행 추가:
  - `primary`, `null_hypothesis`, `success_criterion.delta_threshold` 필수
  - `falsifiability_check`는 너 스스로 PASS/FAIL 판정
- FAIL 후보는 HUMAN_APPROVAL 진입 금지

### 2) HUMAN_APPROVAL — **여기서 멈춰라**
- 3 후보 + 각각의 가설을 JY에게 제시
- JY 응답 대기. 자동 진행 금지.

## 권한 규칙
`engine/core/permission_policy.py` 정책표를 따른다:
- `auto_safe`: 즉시 실행
- `human_gate_required`: JY에게 명시적으로 묻기
- `hard_block`: 거부

특히 `git push`, `rm`, `overwrite`, `baseline_replacement`는 절대 자동 실행 금지.

## 로깅
이번 phase에서 만든 Direction은 `methodology/directions.jsonl`에 `prefix-D###` ID로 추가.
ID는 `projects/$1/rules/direction_prefix.txt`에서 읽음.
