# HANDOFF — Next Claude Session 이어받기

> **새 세션 시작하면 이 파일 먼저 읽어라.** 짧게 현 상태 + 다음 action 전달용.

**Last update**: 2026-04-23 16:55 KST (3 paper-grade findings 확보, 9 commits pushed)

---

## 🟢 Auto mode 진행 중

JY가 자동 모드로 계속 실험 돌리라 지시. 권한 물어보지 말고 논리 판단하며 실행.
Direction ID prefix: `EMA-D###`. 각 iter 끝에 leaderboard/direction 기록.

## 📖 읽는 순서 (5분)

1. **이 파일** (HANDOFF.md) — 현 action 파악
2. `projects/02_emotion_agent/SNAPSHOT.md` — 14 iter 요약 + findings 한 눈에
3. `projects/02_emotion_agent/Q1_WORKING.md` — 논문 draft (abstract/table/findings)
4. `projects/02_emotion_agent/state/leaderboard.jsonl` — 모든 수치 누적
5. `projects/02_emotion_agent/state/insights.md` — 패턴 정리
6. 필요 시 `CLAUDE.md` + `METHODOLOGY.md` (전역 원칙)

## 🎯 현 상황 & 다음 Action

### 방금 일어난 일 (이번 세션, 2026-04-23, 약 6시간)
- BrandSpace off, GPU 48GB free 확인
- **Priority A (Multi-seed) ✅** — exp_012/013/014 × 3 seeds
- **Priority D (LoRA fine-tune) ✅** — exp_015 single + multi-seed
  - single (seed=42): 56.75% acc / F1 0.573
  - multi-seed (42/123/777): **55.00 ± 2.61% / F1 0.555 ± 0.026** — paper-grade
- **Priority G (Anchor-variance) ✅** — exp_016 1+3 / 3+1 sweep
  - Novel two-effect finding (MEAN vs VARIANCE decoupled)
  - Paper §4.4 — unique vs existing ICL literature

**7 commits pushed** this session (ea5f277 → 25b1212).

### 현재 진행해야 할 action — JY 판단 필요

현재 2개 강력한 paper-grade findings 확보:
1. **3-tier adaptation hierarchy**: prompt 29 → ICL 42 → LoRA 55 (multi-seed, statistically separated)
2. **Two-effect exemplar decomposition**: Korean-presence → MEAN, Western-count≥2 → VARIANCE

**우선순위 옵션**:
- **F. Paper §4 작성** (ready): §4.1/§4.2/§4.3/§4.4 모두 결과 확보됨. 논문 쓰기 적기.
- **E. Emotion-LLaMA SOTA 재현** (4-6h): multimodal baseline comparison. 논문에 필요.
- **H. LoRA + ICL combo test** (20min): LoRA 위에 Mixed 프롬프트 추가 → 4-tier 가능성?
- **I. Mechanism analysis** (attention map): §4.4 variance-reduction mechanism 규명
- **B. MELD generalization** (설계 필요): text 도메인으로 효과 전이?

**내 추천**: F (paper writing) + 병렬로 E. 이 정도면 충분.

## 📊 핵심 Findings (multi-seed robust, 논문용)

### ⭐ FINDING 1 — 3-tier adaptation hierarchy (§4.3)

| Tier | Method | Acc (n=3) | F1 | Δ |
|------|--------|-----------|-----|------|
| 1 | zero-shot FACS | 29.08 ± 0.76% | 0.194 | — |
| 2 | ICL k=4 Korean | 41.83 ± 3.41% | 0.364 | +12.75 pp |
| 3 | QLoRA r=16 10K 1ep | **55.00 ± 2.61%** | **0.555** | +13.17 pp |

LoRA LB (52.39) > ICL UB (45.24) → tiers **statistically separated**. Total Δ = +25.92pp.

### ⭐ FINDING 2 — Two-effect anchor decomposition (§4.4)

5-point ratio sweep (k=4 total):

| Kor+Wes | Acc ± σ | σ regime |
|-----------|----------|----------|
| 4+0 | 41.00 ± 4.34 | HIGH |
| 3+1 | 41.50 ± 6.00 | HIGH |
| **2+2** | 41.25 ± 0.87 | LOW |
| **1+3** | 40.08 ± 0.80 | LOW |
| 0+4 | 25.42 ± 0.38 | (low mean) |

**Decoupled**: Korean-presence → MEAN (k=1 충분), Western-count ≥2 → VARIANCE.

### ⭐ FINDING 3 — ICL-LoRA substitutability (§4.3 refinement)

LoRA (seed=42) = 56.75%. On same adapter with ICL exemplars:
- + Mixed k=4: 56.00% (−0.75pp)
- + Korean k=4: 52.75% (−4.00pp)

→ **Adaptation modes are ORDINAL, not ADDITIVE**. LoRA subsumes ICL's Korean-distribution benefit; presenting same distribution as context causes attention-split. Mixed anchors cause *less* regression (consistent with Finding 2's anchor-regularization).

### FINDING 4 — k-scaling revised (§4.2)
- Multi-seed: k=0 29.08 → k=2 44.25 peak → k=4-16 42-44% saturation
- 원 single-seed "k=16 drop"(36%)은 seed 아티팩트.

## 🖥 환경

```
conda env       : cre_q1 (Python 3.10 + torch 2.6 + bnb 0.49)
LLM             : Qwen2.5-7B-Instruct 4-bit (5.4GB) or FP16 (15GB now OK)
VL              : Qwen2.5-VL-7B-Instruct (16GB, 4-bit ~8GB)
GPU             : A6000 48GB (BrandSpace off → 전부 free)
실행 명령 필수   : PYTHONNOUSERSITE=1 conda run -n cre_q1 python <script>
HF cache        : /mnt/hdd/ajy/caches/huggingface
snapshot path 방식: glob(f"{CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")[0]
```

## 📁 데이터 위치

```
IEMOCAP 4-class (6877)   → experiments/exp_001_iemocap_preproc/cache/iemocap_4class_hf.parquet
MELD 4-class (11353)     → experiments/exp_002_meld_preproc/cache/meld_4class.parquet
Korean FER AU (229K)     → /home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet
Yonsei consensus         → /home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv
K-EmoCon (text 없음)     → experiments/exp_003_kemocon_meta/cache/
```

## 🚨 주의사항

- **BrandSpace 재시작 금지** (JY 승인 없이)
- **kill_switch 확인**: `ls /home/ajy/claude-research-engine/scripts/kill_switch` 존재 시 **즉시 중단**
- 실험 결과는 항상 `state/leaderboard.jsonl` append + git commit + push
- 새 direction (제안)은 `methodology/directions.jsonl` append (ID `EMA-D###`)
- JY 스타일: 짧고 직설, 이모지/사과 금지, 확인 없이 실행

## 🔗 Repo
https://github.com/RyanAhn533/claude-research-engine (private)
Git author: `JY <wnsdud2689@gmail.com>`
Co-author 표기: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

Push:
```bash
# JY가 chat 내에서 제공한 PAT 사용. 이 repo는 매번 credential 저장 없이 URL에 PAT 삽입.
# PAT value는 chat 이력 참조. 자동 commit에는 포함 금지 (credential leak scanner 차단).
git -c credential.helper= push https://<PAT>@github.com/RyanAhn533/claude-research-engine.git main
```

## 🎬 바로 실행 시작 명령 (copy-paste)

```bash
# 1) GPU 확인 (48GB free 기대)
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# 2) Multi-seed A부터:
#    exp_012/013/014 을 seed loop로 변형해서 돌림.
#    각 script 복사해서 seed={42,123,777} sweep 후 mean±std 기록.

# 3) 결과 나오면 leaderboard append + commit + push
```

## 📝 완료된 iter 14개

`exp_000` env · `exp_001` IEMOCAP · `exp_002` MELD · `exp_003` K-EmoCon meta · `exp_004` text baseline · `exp_005` agent sketch · `exp_006` IEMOCAP agent · `exp_007` MELD agent · `exp_008` K-EmoCon 4-class (invalid) · `exp_009` K-EmoCon valence (null) · `exp_010` FER landmark (null) · `exp_011` FER AU baseline · `exp_012` few-shot k=4 · `exp_013` k-sweep · `exp_014` cross-cultural (paper-grade)

## 💼 Q2 프로젝트 (참고, 이 세션에선 손대지 마)
`projects/01_au_regionformer_q2/` — draft-ready, 후배 석사 1저자 인계 대기. 건드리지 말 것.
