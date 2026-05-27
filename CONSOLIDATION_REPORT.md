# Claude Research Engine — 통합 전/후 비교

> 작성: 2026-05-27 · 이 세션에서 엔진/저장소에 한 일을 전후로 비교.

---

## 한 줄 요약

엔진 복사본이 **3개로 흩어져 있던 걸** → **`/home/ajy/CLAUDE_RESEARCH_ENGINE/` 하나로 합치고 GitHub 백업**까지 했다. 데이터 손실 0, 전부 되돌리기 가능.

---

## 1. 전(BEFORE) — 흩어진 상태

```
① /home/ajy/CLAUDE_RESEARCH_ENGINE/        ← 우리가 작업한 곳
     - v2 엔진 코드 구현체 (zip 설치본, 5/22)
     - 우리 Q2 작업 (exp_001~003) + 새 02_affective_grounding
     - git remote 없음 ❌ (백업 안 됨)
     - .global / methodology / scripts / 02_emotion_agent 없음 ❌

② /home/ajy/06_research_infra/04_claude-research-engine/   ← GitHub clone
     - 로컬 체크아웃이 옛날(5/16, d776dd0) — 엔진 코드 비어있던 상태
     - 옛 01_(9실험) + 02_emotion_agent(35실험, 206MB) 있음
     - .global / methodology / scripts 있음

③ GitHub 원본 (RyanAhn533/claude-research-engine)  ← 진짜 원격
     - 실제로는 v2.2까지 가있음 (외부리뷰 6버그 수정본)
     - 우리 Q2 작업 / affective_grounding 없음 ❌
```

**문제**: 어디가 "진짜"인지 불명 + 우리 작업이 백업 안 된 로컬에만 있었음.

---

## 2. 후(AFTER) — 하나로 통합

```
/home/ajy/CLAUDE_RESEARCH_ENGINE/   = 단일 소스
     - 엔진 코드: A의 구현체 (※ GitHub v2.2와 거의 동급, 아래 주의 참고)
     - .global / methodology / scripts  ← B에서 가져옴 ✅
     - git 연결됨 ✅ (B의 history 62커밋 + origin remote 이식)
     - GitHub에 백업됨 ✅ (브랜치 backup-zip-consolidation)
     projects/
       ├─ 01_au_regionformer_q2          ← 우리 Q2 (exp_001~003) 그대로
       ├─ 01_au_regionformer_q2_github_legacy  ← B의 옛 9실험 보존
       ├─ 02_affective_grounding         ← 새 Agent 논문 (MASTER/ENVIRONMENT PLAN)
       └─ 02_emotion_agent               ← B의 LoRA 본작업(35실험) 가져옴 ✅
```

---

## 3. 항목별 전/후 표

| 항목 | 전 | 후 |
|---|---|---|
| 엔진 위치 | 3곳 분산 | **1곳 통합** |
| GitHub 백업 | ❌ (A는 remote 없음) | ✅ 브랜치로 push됨 |
| 우리 Q2 작업(exp_001~003) | A 로컬에만 | A + GitHub 백업브랜치 |
| 02_emotion_agent (LoRA 35실험) | B에만 | **A로 합침** |
| .global / methodology / scripts | A에 없음 | **A로 합침** |
| 02_affective_grounding (Agent 논문) | 막 만든 빈 폴더 | MASTER_PLAN + ENVIRONMENT_PLAN |
| 옛 01_(9실험) | B에만 | `_github_legacy`로 보존 |
| 무결성(hash chain) | — | ✅ 검증 통과 (01_ 25행, 02_ OK) |

---

## 4. 무슨 작업을 했나 (순서)

1. 빈 `02_/readme`(네가 쓴 25KB 설계서) → `02_affective_grounding/MASTER_PLAN.md`로 이전, 빈 폴더 제거
2. `02_affective_grounding` 엔진 bootstrap (prefix AGR)
3. B에서 `.global`, `methodology`, `scripts`, `02_emotion_agent`, `gate_result.schema.json`, `.gitignore` 복사
4. B의 옛 `01_` → `01_au_regionformer_q2_github_legacy`로 보존
5. B의 `.git`(62커밋 history + remote) 이식 → 통합 커밋 → **GitHub 백업브랜치 push**
6. 이 환경 문서(`ENVIRONMENT_PLAN.md`) 추가
7. 테스트 찌꺼기(`99_bugfix_*`) 정리

---

## 5. ⚠️ 아직 안 끝난 것 + 주의

- **메인 브랜치 통합 미완**: 현재 우리 작업은 GitHub의 **백업 브랜치**에만 있음. `main`(원격)은 아직 v2.2 + 옛 프로젝트 상태. 메인에 합치는 건 네 "ㄱ" 대기 중.
- **엔진 코드 "훨씬 발전" 아님 (정정)**: A와 GitHub v2.2 **둘 다 6버그 수정된 동급**. A에 `_check_claim_linking` 게이트 1개 더 빡셀 뿐. → 메인은 v2.2 유지하기로 함.
- **state_machine.py 차이**: A가 +130줄 더 큼. 나중에 한 번 맞춰봐야 기능 후퇴 방지.
- 진짜 "발전"은 엔진이 아니라 **그 위 연구**(완료 실험 3개 + Agent 논문 로드맵).

---

## 6. 전체 디렉토리 트리

```
CLAUDE_RESEARCH_ENGINE/
├── .claude/            (commands, hooks)
├── .global/            (protocol, rules_template, templates)   ← B에서
├── docs/
├── engine/             ← 엔진 코드 본체
│   ├── core/      7개  (append_only_logger, state_machine, hashing,
│   │                    leakage_auditor, permission_policy,
│   │                    reproducibility_manifest, compute_budget)
│   ├── gates/     gate_c.py
│   ├── cli/       jy.py            (bootstrap/status/validate/verify-chain)
│   ├── agents/    7개  (method_planner, method_skeptic, novelty_critic,
│   │                    reviewer_simulator, failure_analyzer,
│   │                    repair_planner, insight_summarizer)  ← 폐루프 서브에이전트
│   ├── schemas/   10개 (experiment, hypothesis, claim, paper_tried,
│   │                    permission, task, state, reproducibility_manifest,
│   │                    gate_result + README)
│   └── tests/     test_tier2, test_v2_patches, validate_schemas, fixtures
├── methodology/        (directions.jsonl, logic_chains.jsonl, insights_global) ← B에서
├── scripts/            (init_project.sh, kill_switch_info.md)  ← B에서
└── projects/
    ├── 01_au_regionformer_q2/              ← 우리 Q2 (congruence/ceiling)
    │   ├── SUMMARY.md
    │   ├── experiments/  exp_001_yonsei_disagreement_phase_a
    │   │                  exp_002_phase_b_humankl_ablation
    │   │                  exp_003_ceiling_analysis
    │   ├── negative_results/  exp_001_congruence_head_KILLED
    │   ├── reproducibility_manifests/   (exp_001, exp_002 ×6 seed)
    │   └── state/   leaderboard(10) hypothesis_registry(6) paper_tried(8)
    │
    ├── 01_au_regionformer_q2_github_legacy/   ← B 옛 9실험 보존
    │   ├── experiments/  exp_001~009 (sad_path_fix, demographic_fairness,
    │   │                  knn_graph_smooth, consensus_aware ...)
    │   ├── research_log/  results/  src/
    │
    ├── 02_affective_grounding/             ← 새 Agent 논문
    │   ├── MASTER_PLAN.md        (네 25KB 설계서, §0~15)
    │   ├── ENVIRONMENT_PLAN.md   (시뮬 환경 §6~10)
    │   ├── experiments/  (비어있음 — exp_004~008 들어올 자리)
    │   └── state/   engine_state, permission_policy
    │
    └── 02_emotion_agent/                   ← B LoRA 본작업 (206MB)
        ├── experiments/  exp_000~034 (IEMOCAP/MELD/KEMOCON, LoRA, k-shot,
        │                  prompt ablation, attention viz ...)
        ├── baseline/  figures/  libs/(Emotion-LLaMA, M3NET)
        ├── research_log/  results/  setup/  src/agent/
        └── state/
```
