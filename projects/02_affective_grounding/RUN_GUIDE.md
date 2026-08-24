# Affective Grounding — 실행 가이드 (명령어 모음)

> MASTER_PLAN.md §14 순서를 실제 실행 명령으로. 엔진 절차(design→가설등록→run→log→validate) 준수.
> ⚠️ 자동 chain으로 "존나 돌리려면" 각 exp의 run.py/analyze.py가 먼저 있어야 함. 없는 건 못 돌림.

---

## 0. 엔진 공통 명령 (어디서든)

```bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE

# 상태/무결성
python -m engine.cli.jy status   --project 01_au_regionformer_q2
python -m engine.cli.jy validate --project 01_au_regionformer_q2     # hash chain + schema
python -m engine.cli.jy validate --project 02_affective_grounding

# 새 실험 1개 돌리는 표준 루프 (코드 있을 때)
#  1) design.md 작성 (hypothesis 블록 포함)
#  2) 가설 사전등록  → state/hypothesis_registry.jsonl
#  3) python <exp>/analyze.py  또는  run.py
#  4) leaderboard + hypothesis outcome append
#  5) python -m engine.cli.jy validate
```

---

## 1. 실험별 상태 + 실행 명령

| exp | 내용 | 코드 | 실행 명령 / 필요한 것 |
|---|---|---|---|
| 001 | self-observer 비대칭 | ✅ done | (완료, 01_/experiments) |
| 002 | Human-KL ablation (6 run) | ✅ done | (완료) |
| 003 | partial ceiling | ✅ done | (완료) |
| **004** | V1-V4 taxonomy | ✅ **done** | `python projects/01_au_regionformer_q2/experiments/exp_004_visual_identifiability_taxonomy/analyze.py` |
| 005 | VLM prompt audit | ❌ 코드 없음 | VLM 필요 (로컬 Qwen2.5-VL 또는 API). 5프롬프트×3000img harness 작성 후 실행 |
| 006 | agent preference dataset 생성 | ❌ 코드 없음 | rule-based 생성기 작성 (exp_004 V1-V4 → chosen/rejected jsonl) |
| 007 | reward model / DPO | ❌ 코드 없음 | RM 학습 코드(권장: Option C reward model 먼저). GPU 필요 |
| 008 | physical AI action safety | ❌ 코드 없음 | Unity ML-Agents 또는 Habitat 3.0 환경 (ENVIRONMENT_PLAN.md) |

---

## 2. "바로 돌릴 수 있는" 순서 (코드 짜는 순서 = 실행 순서)

```
exp_005  ← 다음. fine-tuning 없이 VLM 평가만. 로컬 Qwen2.5-VL이면 GPU 추론만.
   ↓
exp_006  ← exp_004의 V1-V4 라벨로 rule-based preference jsonl 생성 (GPU 0)
   ↓
exp_007  ← exp_006 데이터로 reward model 학습 (GPU, sc와 공유)
   ↓
exp_008  ← 시뮬 환경 (별도 셋업 무거움, 마지막)
```

각 exp는 짜는 즉시 `run_affective_chain.sh`에 추가해서 expB처럼 순차 자동 실행 + 자동 로깅.

---

## 3. exp_004 핵심 결과 (다음 실험 설계 근거)

- V3/(V3+V4)=0.773 → observer rejection ≠ label invalidity (모델이 self-report 잘 따라감)
- 오답의 70%가 V2(명확한 얼굴 오답) → 모델 개선 여지는 "명확 샘플"에 있음
- 함의: exp_005에서 VLM이 **observer-reject 샘플에서 over-infer 하는지**가 핵심 측정 대상
