# Claude 레버리지 2026 — 엔진 운영 보강

> **METHODOLOGY.md 보조 문서.** Anthropic 2026 best practice + research-agent 논문에서 추린 것 중, 이 엔진에 실제로 적용 가치 있는 것만 남김.

작성일: 2026-05-03
근거: Anthropic Claude Code Docs (2026), Agentic Coding Trends Report 2026, "Why LLMs Aren't Scientists Yet" (arxiv 2601.03315), Agent Laboratory (arxiv 2501.04227)

---

## 0. 왜 이 문서가 필요한가

`METHODOLOGY.md`는 **연구 process 운영 룰**(5-phase loop, direction ID, leaderboard 무결성)에 집중한다. 그런데 2026년 들어 Claude 자체의 사용 방식이 크게 바뀌었다 — Plan mode, subagent, skills, hooks, adaptive thinking, prompt caching. 이 변화를 엔진에 반영하지 않으면 같은 돈/시간으로 절반의 성과가 난다.

이 문서는 **"Claude 사용 도구 레이어"의 룰**이다. METHODOLOGY가 *무엇을 할지* 정한다면, 이 문서는 *어떻게 할지*의 도구 선택을 정한다.

---

## 1. 2026 트렌드 요약 (이 엔진과 관련된 것만)

### 1-1. "Step-by-step"은 죽었다

Claude 4.6/4.7은 답하기 전에 이미 reasoning한다. "let's think step by step"은 더 이상 hidden reasoning을 unlock하지 않고, 오히려 적응형 사고(adaptive thinking)의 budget 결정을 방해한다.

**엔진 영향**: Phase 1(방법론 탐색) 프롬프트에서 "단계별로 생각해"류 표현 제거. 대신 effort budget만 명시.

### 1-2. Adaptive thinking — budget만 주고 Claude가 결정

Opus 4.6/4.7과 Sonnet 4.6은 effort budget(low/medium/high/max)을 받으면 task 복잡도에 따라 reasoning 양을 자동 조절한다. 단순 질문엔 thinking을 건너뛰고, 복잡한 결정엔 깊이 들어간다.

**엔진 영향**:
| 작업 | effort | 이유 |
|-----|--------|-----|
| 코드 fix, log 읽기 | low | overkill 회피 |
| Phase 1 방법 탐색 | high | 대안 비교 필요 |
| 토론 프레임워크 (3 페르소나) | max | 양비론 회피, 깊은 반박 |
| Phase 5 insight 갱신 | high | 패턴 추출 |
| Hindsight score 후 분석 | high | DPO pair 추출 |

### 1-3. Plan mode — read-only 사전 분석

큰 변화 전 read-only mode로 codebase를 훑고 plan을 먼저 세우는 패턴. Plan mode 안에서는 Claude가 어떤 mutation도 못 하므로, JY 승인 룰과 자연스럽게 맞물린다.

**엔진 영향**: METHODOLOGY §0의 "JY 승인 후 실행" 룰 강화. **Phase 1에서 새 방법 탐색 시 plan mode 권장**.

### 1-4. Subagent — context 격리 + 병렬

Side task(논문 검색, log 분석, leaderboard 통계)가 main 컨텍스트를 오염시키는 게 가장 흔한 비효율. Subagent에게 그 작업을 위임하면 main에는 summary만 돌아온다.

**엔진 영향**: 아래 패턴 권장.

| 작업 | Subagent 권장? | 이유 |
|-----|--------------|-----|
| 논문 N개 비교 (Phase 1-a) | ✅ | 본문 다 읽으면 main 30K+ 토큰 소모 |
| Leaderboard.jsonl 통계 분석 | ✅ | 100+ 라인 raw data |
| 단일 design.md 작성 | ❌ | main에서 직접 |
| 실험 코드 디버깅 | ❌ (보통) | 반복 수정 필요 |

### 1-5. Skills, Hooks — 반복 룰의 코드화

- **Skills**: 자주 쓰는 워크플로우를 markdown으로 정의 (`/distill`, `/debate` 등)
- **Hooks**: lifecycle event에 deterministic script (예: 매 commit 전 leaderboard.jsonl 검증)

`~/agents/`(JY의 bash 기반)와 `~/knowledge/distill.sh`가 이미 비슷한 역할을 하므로, 단기에 마이그레이션 강제할 필요 없음. **장기 옵션**으로만 기록.

### 1-6. Prompt caching — stable 컨텍스트는 캐시

`CLAUDE.md`, `METHODOLOGY.md`, recent `leaderboard.jsonl`은 매 세션마다 같은 내용이 들어간다. 캐시하면 재요청 시 토큰비 ~10% 수준.

**엔진 영향**: API로 호출할 때 (예: `~/agents/agent.sh critic ...`)는 system prompt 부분을 cache_control 표시. Claude Code 사용 시는 자동으로 처리되므로 별도 작업 불필요.

---

## 2. Research-agent 논문에서 가져올 lesson

### 2-1. "Why LLMs Aren't Scientists Yet" (arxiv 2601.03315)

4번의 end-to-end 자율 연구 시도 중 **3번이 implementation 또는 evaluation 단계에서 실패**. 즉 ideation/literature review는 LLM이 잘 하지만, **코드 디버깅 + 평가 신뢰성** 단계가 약하다.

**엔진 영향**:
- Phase 2(구현)의 "3회 디버그 후 포기" 룰은 이 패턴과 정확히 맞물림 — 유지
- Phase 3(평가)의 random baseline + p-value + 3 seed 룰도 이 패턴 방어 — 유지
- **추가**: Phase 2 실패 시 단순 retry 말고 **subagent로 root cause 분석**한 후 재시도

### 2-2. Agent Laboratory (arxiv 2501.04227)

3-stage(literature/experiment/report) + human feedback at each stage. **o1-preview로 best 결과**. Human-in-the-loop이 self-loop보다 우수.

**엔진 영향**: 우리 엔진이 이미 human(JY) ↔ Claude 반복 구조라 사상 일치. 다만 **"hindsight_score를 1주 후 매김" 룰**은 Agent Laboratory보다 진일보 — 즉시 feedback이 아닌 **결과 기반 retrospective**라 더 강한 신호.

---

## 3. METHODOLOGY.md와의 매핑

이 문서는 METHODOLOGY를 대체하지 않고 보강한다. 매핑:

| METHODOLOGY 섹션 | 이 문서의 보강 |
|-----------------|---------------|
| §0 핵심 원칙 1 (JY 승인 후 실행) | §1-3 Plan mode 권장 |
| §1 Phase 1 (방법 탐색 a/b/c/d) | §1-2 effort=high, §1-4 논문 조사는 subagent |
| §1 Phase 2 (구현, 3회 디버그) | §2-1 lesson — 단순 retry X, root cause 분석 |
| §1 Phase 3 (평가) | §2-1 maintained |
| §1 Phase 5 (insight) | §1-2 effort=high, §1-4 leaderboard 통계는 subagent |
| §6 토론 프레임워크 | §1-2 effort=max |
| §10 Claude 금지사항 | §1-5 hooks로 강제 가능 (장기) |

---

## 4. 즉시 적용 (작업 0개) vs 장기 옵션

### 즉시 적용 (별도 코드 작업 없음, 이 룰만 따르면 됨)
1. Phase 1 방법 탐색 시 — plan mode 사용
2. Phase 1-a 논문 N개 비교 시 — subagent에 위임
3. Phase 5 insight 갱신 — effort=high
4. 토론 프레임워크 — effort=max
5. "단계별로 생각해" 류 prompt 표현 사용 안 함

### 장기 옵션 (작업 필요, 임팩트 중간)
1. `~/agents/distill.sh, debate.sh` → Claude Code Skills로 마이그레이션
2. `pre-commit` hook으로 leaderboard.jsonl 형식 자동 검증
3. `methodology/directions.jsonl` append 시 ID 충돌 검사 hook

장기 옵션은 **현재 plan 우선순위(Phase 6 Stage 16 등)에 영향 없을 때만** 진행.

---

## 5. 참조

- [Best Practices for Claude Code](https://code.claude.com/docs/en/best-practices) — Anthropic 공식
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic 연구 글
- [How Anthropic Teams Use Claude Code](https://www-cdn.anthropic.com/58284b19e702b49db9302d5b6f135ad8871e7658.pdf) — 내부 사례 PDF
- [2026 Agentic Coding Trends Report](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf) — Anthropic 트렌드 보고서
- [Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking) — Claude API docs
- [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) — Claude API docs
- [Plan mode](https://www.mejba.me/ai-school/claude-code-mastery-2026-agentic-engineering-bootcamp/lessons/context-engineering-project-setup/leveraging-plan-mode-for-risk-free-design) — 2026 가이드
- [Why LLMs Aren't Scientists Yet](https://arxiv.org/abs/2601.03315) — 자율 연구 실패 lessons
- [Agent Laboratory](https://arxiv.org/abs/2501.04227) — LLM 연구 보조 프레임워크
- [Subagents Complete Guide](https://ofox.ai/blog/claude-code-hooks-subagents-skills-complete-guide-2026/) — 2026 통합 가이드

---

## 6. 한 문단 요약

2026년의 Claude는 더 이상 "step-by-step 시켜야 사고하는 모델"이 아니다 — Adaptive thinking, Plan mode, Subagent, Skills, Hooks, Prompt caching이 표준이 됐다. 이 엔진의 5-phase loop에 가장 임팩트 큰 즉시 적용 룰은 (1) Phase 1 탐색 시 Plan mode + 논문 조사 subagent 위임, (2) effort budget 명시(평소 high, 토론 max, 단순 작업 low), (3) "단계별로 생각해" 류 표현 제거다. Skills/Hooks 마이그레이션은 장기 옵션이며 현재 plan 우선순위가 끝난 후에만 검토한다. 자율 연구 실패 패턴(arxiv 2601.03315)은 implementation/evaluation 단계에 집중되므로, METHODOLOGY의 "3회 디버그 후 포기 + random baseline + p<0.05 + 3 seed" 룰을 더 엄격히 지키는 게 가장 큰 방어다.
