---
description: schema 셀프 검증 + 프로젝트 jsonl 해시체인 무결성 검증
allowed-tools: Bash(python:*)
argument-hint: [project-name (optional)]
---

```bash
python -m engine.cli.jy validate ${1:+--project $1}
```

이 명령은:
1. `engine/schemas/*.json` 8개를 draft-07 self-validate
2. `engine/tests/fixtures/*.{valid,invalid}.json` 16개로 positive/negative 검증
3. 프로젝트 지정 시 모든 `state/*.jsonl`을 해시체인 끝까지 walk

하나라도 실패 → exit 1. **실패하면 절대 다음 명령으로 넘어가지 마.** 깨진 chain은 즉시 알려야 함.
