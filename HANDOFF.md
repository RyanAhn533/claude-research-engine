# HANDOFF — Next Claude Session 이어받기

> **새 세션 시작하면 이 파일 먼저 읽어라.** 짧게 현 상태 + 다음 action 전달용.

**Last update**: 2026-04-23 06:50 KST

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

### 방금 일어난 일
- JY가 BrandSpace 종료 허가 → `pkill -f "serve.py --port 8090"` 실행 완료
- **GPU 48GB 전부 free (확인됨)**
- Auto mode 지속 — Multi-seed (A) + LoRA prep (D) 진행 중이던 상태

### 현재 진행해야 할 action

**Priority 1 (A — Multi-seed robustness, ~1h)**:
exp_012/013/014를 3-seed (42, 123, 777) 재측정. 현 findings에 p-value + std 붙여 reviewer 방어력.

Target files:
- `experiments/exp_012_fewshot_au/run.py` — seed 인자 추가
- `experiments/exp_013_kshot_sweep/run.py` — 동일
- `experiments/exp_014_cross_cultural_exemplar/run.py` — 동일

**Priority 2 (D — LoRA fine-tune, 2-4h, BrandSpace off 필요함 — 이미 off)**:
- Training data: Korean FER AU parquet + label → instruction format
- Base: Qwen2.5-7B-Instruct
- LoRA rank 16, alpha 32, 1 epoch on 10K Korean 4-class
- Output: fine-tuned adapter → 재평가 expected +10-15pp

## 📊 핵심 Findings (절대 잊지 말 것)

**3-levels of cultural grounding on Korean FER AU (exp_011→014)**:

| Level | Method | Δ acc |
|-------|--------|------|
| Abstract prompt | cultural text | **−7 pp** |
| Ekman FACS textbook prototype | idealized AU | **±0 pp (random)** |
| Actual Korean exemplars (k=2) | sampled from 237K | **+15 pp** |
| **Mixed Korean+Western** | 2+2 exemplars | **+18.75 pp (44.25 % best)** |

+ **Inverted-U scaling** (k=2 opt, k=16 drop). Paper §4 main finding.

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
