# ROADMAP — Emotion Agent Q1

> **4 주 타임라인.** 각 단계의 목표·이유·reference repo·산출물을 명시.

---

## Overview

```
Week 1  데이터/env sanity + baseline 재현
Week 2  Top-tier SOTA 1개 재현
Week 3  Agent prototype (Qwen + cultural prior)
Week 4  Full comparison + ablation → Q1 draft §4 재료
```

---

## Phase 0 — Setup (Day 0, **즉시 실행**)

**Why**: 실험 환경 먼저 안정화 안 하면 Week 1 첫 Day부터 deps hell.

### Tasks
| # | Action | Ref |
|---|--------|-----|
| 0.1 | `cre_q1` conda env 생성 (Python 3.10 + torch 2.6 + transformers + bitsandbytes) | `setup/SETUP.md` |
| 0.2 | Qwen2.5-7B / Qwen2.5-VL-7B 4-bit quant 로드 (GPU 7.7GB 내) sanity | `setup/SETUP.md` |
| 0.3 | Data path mapping (IEMOCAP/MELD/DEAP/MOSEI/KEMDy20) 확인 | `setup/SETUP.md` |
| 0.4 | `kill_switch` 경로 설정 | ROOT scripts/ |

### Gate
Qwen2.5-7B 4-bit로 1개 prompt inference 돌아가면 Phase 1 GO.

---

## Phase 1 — Week 1: Data + baseline sanity

**Why**: 어떤 agent든 **동일 protocol**에서 비교 안 되면 논문 reject. 4개 benchmark 전부 우리 env에서 로드 + SGMT-style baseline 수치를 확보해야 Week 2 SOTA 비교가 의미 있음.

### 1.1 IEMOCAP preprocessing 재활용
- **Ref**: `/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_data_preprocessing.ipynb` (BG 랩원 코드)
- **External ref**: [`feiyuchen7/M3NET`](https://github.com/feiyuchen7/M3NET) (CVPR 2023) — PyTorch preprocessing pattern
- 산출: `experiments/exp_001_iemocap_preproc/` — preprocessed pkl, 4-class emotion split

### 1.2 MELD load + schema
- **Ref**: [`declare-lab/MELD`](https://github.com/declare-lab/MELD) (official repo — schema, dev/test/train split)
- 산출: MELD utterance-level multimodal tensors (text, audio mel, face frame)

### 1.3 DEAP bio load (EEG 32ch + peripheral)
- **Ref**: [`willxxy/awesome-mmps`](https://github.com/willxxy/awesome-mmps) — physiological multimodal baseline list
- 기존 자산: `/mnt/hdd/BG/MER/DEAP_code/`
- 산출: DEAP 32ch × 32 subjects × 40 trials preprocessed

### 1.4 KEMDy20 / K-EmoCon (우리 고유)
- **Ref**: S-PACE pipeline (이미 작동)
- 재활용: SGMT 학습 시 사용한 전처리 pickle 재활용

### 1.5 Baseline reproduction (4 datasets × SGMT-style)
- **Ref**: JY's SGMT 논문 (Mathematics submit) — 동일 backbone
- 산출: `state/leaderboard.jsonl` 초기 4 entries

### Week 1 Gate
4 dataset 각각 baseline 수치 확보 → Week 2 GO.

---

## Phase 2 — Week 2: Top-tier SOTA 재현

**Why**: 우리 agent의 "+X%p"를 주장하려면 SOTA를 우리 환경에서 똑같이 돌려야 함. 논문 수치 인용만으로는 reviewer 공격 받음.

### 2.1 Primary SOTA: Emotion-LLaMA (NeurIPS 2024)
- **Ref**: [`ZebangCheng/Emotion-LLaMA`](https://github.com/ZebangCheng/Emotion-LLaMA)
- **Venue**: NeurIPS 2024 (top-tier)
- **Why 이것**: Multimodal emotion + instruction-tuned LLM — 우리 agent direction과 정면 비교 대상
- **Approach**: repo clone → MERR 데이터 활용 → IEMOCAP/MELD 위 실행
- **GPU 주의**: LLaMA-2 기반이라 7.7GB로는 타이트. 4-bit quant 활용
- 산출: SOTA 수치 재현 (±2%p tolerance)

### 2.2 Graph-based MER baseline: M3NET (CVPR 2023)
- **Ref**: [`feiyuchen7/M3NET`](https://github.com/feiyuchen7/M3NET)
- **Why**: Graph NN 기반 MERC의 전통 baseline. Qwen agent vs Graph 비교에 의미
- 산출: IEMOCAP/MELD weighted F1

### 2.3 Recent MLLM MERC: BeMERC (arXiv March 2025)
- **Ref**: [arXiv 2503.23990 — BeMERC](https://arxiv.org/html/2503.23990) (repo 공개 여부 확인 필요)
- **Why**: 가장 최근 MLLM-based MERC. 직접 비교
- 없으면: alternative reproduce

### 2.4 Bio+LLM baseline: EEG Emotion Copilot
- **Ref**: [Neural Networks 2025 논문](https://www.sciencedirect.com/science/article/abs/pii/S0893608025007282) (repo 확인 필요)
- **Why**: Bio+LLM direct competitor

### Week 2 Gate
SOTA 1-2개의 reported 수치를 ±2%p 내 재현 → Week 3 GO.

---

## Phase 3 — Week 3: Agent prototype

**Why**: 우리 novelty 3축 (bio-grounded, cultural prior, LLM reasoning)을 **돌아가는 시스템**으로 만들어야 ablation 가능.

### 3.1 Agent architecture 구현
```
Perception  →  LLM Reasoning (Qwen2.5-7B 4-bit)  →  Decision
(SGMT out)      ↑
                Cultural prior prompt
                (Jack 2012 + Yonsei consensus)
```

- **Ref**: [`QwenLM/Qwen-Agent`](https://github.com/QwenLM/Qwen-Agent) — agent framework
- **Ref**: [`yuntaoshou/Awesome-Emotion-Reasoning`](https://github.com/yuntaoshou/Awesome-Emotion-Reasoning) — reasoning pattern
- 산출: `src/agent/` 구현, IEMOCAP에서 end-to-end 1 sample 돌림

### 3.2 Cultural prior injection 메커니즘
- Jack 2012 핵심 finding 4-5개를 prompt template에 주입
- 298-person consensus score를 reliability hint로 주입
- **Ref**: Project 01의 `research_log/references/au_region_importance_2026-04-21.md`
- 산출: prompt templates + ablation 준비

### 3.3 Bio-grounded labeling integration
- JY 다른 서버의 bio labeling agent 결과 → 우리 pipeline에 pseudo-label로 통합
- **Ref**: JY가 다른 서버에서 내용 공유 필요 (**blocker**)
- 산출: Bio pseudo-label이 있는 training subset

### Week 3 Gate
Agent가 IEMOCAP 1 batch 처리 + SOTA 대비 ≥ 동등 수준 → Week 4 GO. 미달 시 scope 축소.

---

## Phase 4 — Week 4: Ablation + main table

**Why**: Q1 논문의 §4 results table을 만드는 단계. 여기 없으면 교수 미팅에서 설득 불가.

### 4.1 Ablation matrix
```
  Dataset        | Baseline | +bio-grounded | +cultural | Full agent
  IEMOCAP        |    ?     |       ?       |     ?     |     ?
  MELD           |    ?     |       ?       |     ?     |     ?
  KEMDy20        |    ?     |       ?       |     ?     |     ?
  DEAP (bio-only)|    ?     |       ?       |     ?     |     ?
```

### 4.2 Per-class F1 + statistical significance
- 3-seed mean ± std
- p-value vs baseline (bootstrap 1000)

### 4.3 Cultural prior contribution 유의성
- 이게 유의 안 하면 **novelty 1축 붕괴** → paper pivot 필요

### 4.4 Q1 §4 draft table 완성
- `results/phase0/` 아래 table/figure 저장
- `Q1_WORKING.md`의 §4 Results 채움

### Week 4 Gate
ICMI / ACII submit-ready 수치 + 2 figure + draft table → Moon 교수 미팅 가능.

---

## Reference Repo 전체 매핑

자세한 repo 리스트는 [`references/REPOS.md`](references/REPOS.md) 참조.

| 용도 | Repo |
|-----|------|
| Preprocessing / baseline | `declare-lab/MELD`, `feiyuchen7/M3NET` |
| Bio+MER resource | `willxxy/awesome-mmps` |
| Top-tier LLM+Emotion | `ZebangCheng/Emotion-LLaMA` |
| Landscape (curated) | `yuntaoshou/Awesome-Emotion-Reasoning` |
| MLLM landscape | `BradyFU/Awesome-Multimodal-Large-Language-Models` |
| Agent framework | `QwenLM/Qwen-Agent` |

---

## 종료 조건 (엔진 기준)

- **target_reached**: IEMOCAP F1 > SOTA + p<0.05 + cultural prior ablation 유의
- **plateau**: 20 iter 연속 개선 없음 → `state/plateau.md` + review_queue push
- **pivot**: Week 3 gate 미달 3회 → scope 축소 (4 dataset → 2)
- **kill_switch**: `scripts/kill_switch` 파일 존재 시 즉시 중단

---

## Decisions Log (업데이트 시 추가)

| 날짜 | 결정 | 이유 |
|------|-----|-----|
| 2026-04-22 | Qwen2.5-7B local (Claude API 대신) | GPU local + 비용 0 + reproducibility |
| 2026-04-22 | 4 benchmark (IEMOCAP/MELD/DEAP + KEMDy20) | TAFFC reviewer 기본 요구 |
| 2026-04-22 | Primary SOTA = Emotion-LLaMA | NeurIPS 2024, LLM+emotion direct |
| 2026-04-22 | Agent ID prefix = `EMA-D###` | Project 02 기본 |
| 2026-04-22 | 워킹 디렉토리 = `claude-research-engine/projects/02_emotion_agent/` | JY 지시 |
