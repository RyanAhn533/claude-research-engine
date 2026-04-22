# claude-research-engine

> **JY's Claude-driven autonomous research engine.**
> Q1 논문 워크스페이스 + iteration loop + RLHF 축적 + 방법론 누적.

**Visibility**: private
**Owner**: RyanAhn533 · **Primary user**: JY
**Created**: 2026-04-22

---

## 🎯 목적 3축

| # | 축 | 구현 |
|---|-----|-----|
| 1 | **Q1 논문 작성** (TAFFC/NHB) | `projects/NN_xxx/` 별 완결된 research workspace |
| 2 | **Claude 자동화 연구 엔진** | iteration design → run → leaderboard → insights 루프 |
| 3 | **Claude 방법론 축적** | `methodology/` 전역 RLHF 데이터 (directions, logic chains) |

---

## 📂 구조

```
claude-research-engine/
├── README.md                      이 파일 — 전체 인덱스
├── CLAUDE.md                      전역 Claude 설정 (모든 하위 프로젝트 적용)
├── METHODOLOGY.md                 "Claude를 어떻게 쓸 것인가" 방법론 (범용)
│
├── .global/                       엔진 공통 자산
│   ├── rules_template/            target_metrics/constraints 템플릿
│   ├── templates/                 session/experiment/logic_chain/direction_eval 템플릿
│   └── protocol/                  engine_loop, report_format 등 protocol
│
├── methodology/                   ⭐ 프로젝트 횡단 RLHF 데이터 누적
│   ├── directions.jsonl           전역 direction 로그 (ID prefix: AUR-D###)
│   ├── logic_chains.jsonl         전역 논리 체인
│   └── insights_global.md         프로젝트 간 공통 패턴
│
├── projects/                      개별 프로젝트 (각자 독립 완결)
│   └── 01_au_regionformer_q1/     한국인 감정 AI-psych gap (Q1 논문)
│       ├── README.md              프로젝트 대시보드
│       ├── Q1_WORKING.md          논문 drafting + progress tracker
│       ├── PROJECT_STATUS.md      데이터/실험 현황
│       ├── rules/                 target_metrics, constraints
│       ├── state/                 leaderboard, paper_tried, insights
│       ├── experiments/           exp_NNN/ per iteration
│       ├── research_log/          sessions, references, claude_evaluation
│       ├── src/analysis/phase0/   분석 코드
│       └── results/phase0/        작은 결과물 (summary, figure)
│
└── scripts/
    ├── kill_switch                자동화 중단 스위치
    └── init_project.sh            새 프로젝트 initialize helper
```

## 🚦 활성 프로젝트

| # | Name | Status | Target | Best metric |
|---|------|--------|--------|------------|
| 01 | **au_regionformer_q1** | active (9 iter) | TAFFC (IF 11) | 87.56% (exp_009 triplet fusion) |
| 02 | *(reserved for s_pace)* | — | — | — |
| 03 | *(reserved for biotoken)* | — | — | — |

## 🧭 새 세션 시작 순서

1. 이 README
2. `CLAUDE.md` — 전역 Claude 원칙
3. `METHODOLOGY.md` — 엔진 운영 방법
4. `projects/NN_xxx/README.md` — 해당 프로젝트 상세
5. `projects/NN_xxx/state/leaderboard.jsonl` — 현재 best
6. `projects/NN_xxx/research_log/sessions/` 최신 파일

## ⚡ 자동화 엔진 간단 프로토콜

매 iteration:
1. **Phase 1 Design**: `projects/NN/experiments/exp_NNN/design.md` 작성
2. **Phase 2 Implement**: 코드 수정
3. **Phase 3 Evaluate**: 3-fold CV, Random baseline 대비 %p
4. **Phase 4 Record**: `leaderboard.jsonl` append
5. **Phase 5 (매 5 iter)**: `insights.md` 업데이트

상세: `.global/protocol/engine_loop.md`

## 🛡️ 규칙 (요약)

- 평가 프로토콜 고정 (데이터 split, seed, metric)
- p < 0.05 미달은 개선으로 간주 X
- 같은 방법 재시도 금지 (`paper_tried.jsonl` 체크)
- `scripts/kill_switch` 존재 시 즉시 중단
- 최근 10 iter 내 동일 category 연속 3회 초과 금지

## 🔒 IP / 라이선스

Private. JY 개인/랩 자산 포함. 연세대 298명 raw data 외부 공개 금지.
