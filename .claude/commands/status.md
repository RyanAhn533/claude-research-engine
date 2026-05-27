---
description: 현재 프로젝트의 엔진 상태 확인
allowed-tools: Bash(python:*)
argument-hint: [project-name]
---

```bash
python -m engine.cli.jy status --project $1
```

읽은 후:
- `phase`가 무엇인지 확인
- `pending_human_gate`가 비어있지 않으면 그 게이트가 무엇인지 JY에게 보고
- `kill_sw`가 true면 **즉시 중단**하고 사용자에게 알림
- 마지막 leaderboard 행과 마지막 hypothesis_registry 행을 한 줄씩 요약
