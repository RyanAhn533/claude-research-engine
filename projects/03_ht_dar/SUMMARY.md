# 03_ht_dar — Hierarchical Multimodal Fusion 연구 요약

> 최종 갱신: 2026-06-06 · 엔진 상태: BOOTSTRAP / iter 0 · GPU budget 50h
> HT-DAR(동적 앵커 라우팅) thesis는 **NO-GO(2026-06-05)**. 현재 SR-XMA(efficiency) 피벗 — **exp_005 E1 실행 중**.

## 🔴 진행 중: exp_005 SR-XMA E1 (생사 관문)

- 4모드(SR_FULL/FIXED/RAND/LEARN) × 3seed MOSEI, 2동시 큐 (`tools/run_srxma_e1.sh`).
- 생사: **learned-k(2) > random-k(2)?** CI 하한>0 이면 계속, 0 포함이면 STOP.
- 설계/가설: `experiments/exp_005_srxma_e1/design.md` · 집계: `aggregate_e1.py` · 함정: `tools/srxma_preflight.py`
- ⚠️ V1 효율은 B=1에서만 실재(batched=masked-full). 정확도 비교는 유효, 효율은 B=1 이론-FLOPs로만 방어.
- 모니터(`/home/ajy/HT-DAR/code/_e1_monitor.sh`)가 완료 시 자동 집계 → `logs_srxma_e1/_E1_RESULT.txt`.

---

## 한 줄 요약

> **공정한 비교(modality-dropout 학습)에서 동적/그래프 앵커 라우팅은 단순 고정앵커 baseline을
> 못 이긴다.** MOSEI clean·IEMOCAP·robustness 세 축 전부 within-noise. HT-DAR 컨퍼런스 thesis는 죽었다.
> 다음 베팅 = **SR-XMA**(sparse-routed cross-attention, efficiency Pareto) — 코드만 있고 검증 0%.

---

## 실험 현황

### ✅ 끝난 것 (재현성 OK)

| 실험 | 결과 | seed | 판정 | 엔진기록 |
|---|---|---|---|---|
| MOSEI clean ladder M0/M1 | 0.853 / 0.852 Acc2 | 5 | neutral (Δ=−0.04pp) | leaderboard |
| MOSEI ladder M2~M4 | 0.849~0.853 | 2/2/1 (불완전) | — | 미로깅 |
| IEMOCAP ladder M0~M4 + A1_parallel | 0.582~0.594 wF1 | **5 전부** | 전부 neutral (max Δ+0.84pp, p≥0.5) | leaderboard (9 rows) |
| MOSEI robustness/mdrop static/dar/hgar | clean 0.850/0.848/0.852 · 100%-drop static 0.647 최고 | 3(clean)+1(sweep) | **NO-GO** | 문서만 |

→ HT-DAR(dynamic/graph anchor routing)는 세 축 전부에서 fixed baseline 못 이김. 정직하고 잘 재현된 음성 결과.

### 🆕 SR-XMA V1 (피벗 후보) — 코드만 존재, 검증 0%

- 코어: `/home/ajy/HT-DAR/code/trains/singleTask/model/sr_xma.py` (148줄, self-contained, 4모드 full/fixed/random/learned + topk)
- ❌ 어디에도 import 안 됨(train 미연결) · ❌ FLOPs 측정 없음(edges_per_node만) · ❌ 교차검증 0회 실행
- 다음 관문 = **E1 (learned-k > random-k, k=2, 3-seed)** — 여기서 지면 이 방향도 STOP → 계획: `docs/plans/PLAN_SRXMA_experiments.md`

---

## 코드 위치 (엔진 밖)

- 실제 학습/결과 repo: **`/home/ajy/HT-DAR/code/`** (train_stripped.py, train_iemocap.py, logs_run/, logs_mdrop/, logs_robust/, pt/)
- 코드맵: `/home/ajy/HT-DAR/docs/HT_DAR_CODE_MAP.md`

## 디렉토리 (이 엔진 프로젝트)

```
03_ht_dar/
├── SUMMARY.md          ← 이 파일
├── docs/
│   ├── design/   SR-XMA 설계, MASTER_IMPLEMENTATION
│   ├── plans/    PLAN_SRXMA_experiments (E0~E7 사다리)
│   ├── findings/ NO_GO_FINDING, SESSION_REPORT
│   ├── survey/   REPO_SURVEY, NOVELTY_CHECK, GPT 자문
│   └── paper/    CONFERENCE_PAPER_PACKAGE_draft
├── tools/        gate_c_runner.py, log_results.py
├── artifacts/    CONFIG_FINGERPRINTS, exploratory_single_seed
├── state/ experiments/ negative_results/ reproducibility_manifests/ rules/
```

## 다음 결정 (JY)

1. **SR-XMA E1 준비·실행** — srxma→MOSEI 연결 + FLOPs훅 + 교차검증 → E0+E1 3-seed (생사 관문)
2. **접고 피벗** — NO_GO 옵션: CBBF 저널(`projects/04_cbbf/`) 집중 / 정직한 negative 워크샵
