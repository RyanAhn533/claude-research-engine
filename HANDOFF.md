# HANDOFF — Next Claude Session 이어받기

> **새 세션 시작하면 이 파일 먼저 읽어라.** 짧게 현 상태 + 다음 action 전달용.

**Last update**: 2026-04-24 03:30 KST (overnight M1 queue 완료, cross-domain 결과 + §1/§2/§6 draft 완료. 16+ commits pushed.)

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

### 방금 일어난 일 (2026-04-23 저녁 + 2026-04-24 새벽 세션)

**2026-04-23 세션 (저녁)** ~ **2026-04-24 dawn 세션** (9 commits 추가):

Experiments added (exp_018~026 total 9 experiments):
- **exp_018 random-token anchor**: G1 σ=2.50 (중간), semantic content 역할 확인
- **exp_020 class-coverage**: **class redundancy가 σ 주 요인 (2.81pp)** — anchor-variance 주장 REVISED
- **exp_021 IEMOCAP T1+T2**: T1 47.83 / T2 48.92 → **ICL gain +1.09pp만** (대폭 감소)
- **exp_022 MELD T1+T2**: T1 55.50 / T2 55.17 → **−0.33pp (ICL 음수)**
- **exp_023 IEMOCAP k-sweep**: FLAT (46.67/46.17/47.00) — ICL 자체 무효
- **exp_024 IEMOCAP LoRA multi-seed**: **70.00 ± 3.70%** (+22pp vs T1)
- **exp_025 MELD LoRA multi-seed**: **61.75 ± 1.15%** (+6pp vs T1)
- **exp_019 attention entropy**: mechanism hypothesis REJECTED (D vs E entropy 거의 동일)
- exp_026 MELD k-sweep + exp_019b multi-layer attention: 진행 중 (04:00경 완료 예상)

**Paper narrative 대폭 revision**:
- "ICL universal 3-tier" → "**LoRA universal, ICL modality-gated**"
- "Western ≥2 threshold for σ" → "**3-axis decomposition** (class redundancy 2.81pp + anchor origin 0.66pp + semantic content 1.63pp)"
- Attention-level mechanism 기각 → "hidden-representation level" 가설로 이동

**Q1_WORKING v2 완료**: Abstract v2 + §1 intro + §2 related + §3 method + §4 findings (revised) + §5 discussion (확장) + §6 future work (5 subsections) + limitations. ~95% 작성됨.

**Figures 5개**: fig1 Korean FER 3-tier / fig2 anchor-decomp 5-config / fig3 k-sweep cross-domain / fig4 LoRA+ICL / fig5 cross-domain 3-tier (main).

**16+ commits pushed** 누적 (ea5f277 → e5c9b71).

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

## 📊 핵심 Findings (final revised, 2026-04-24 dawn)

### ⭐ FINDING 1 — Cross-domain 3-tier (LoRA universal, ICL modality-gated)

| Dataset | T1 zero-shot | T2 ICL k=4 | T3 LoRA | ΔT1→T3 |
|---------|-------------|-----------|---------|--------|
| Korean FER AU | 29.08 ± 0.76 | 41.83 ± 3.41 (+12.75) | **55.00 ± 2.61** | +25.92 |
| IEMOCAP text | 47.83 ± 2.50 | 48.92 ± 4.25 (+1.09) | **70.00 ± 3.70** | +22.17 |
| MELD text | 55.50 ± 3.27 | 55.17 ± 5.65 (−0.33) | **61.75 ± 1.15** | +6.25 |

→ LoRA 모든 domain robust. ICL은 Korean FER AU에서만 큰 gain.
→ k-sweep IEMOCAP에서 FLAT (46.67/46.17/47.00) → ICL 진짜 무효.
→ **New hypothesis: ICL은 "input modality가 LLM에게 novel"할 때만 효과** (AU intensity = novel, standard text = familiar).

### ⭐ FINDING 2 — Three-axis anchor variance decomposition (REVISED)

| Config | Acc ± σ | σ 기여 축 |
|---|---|---|
| E Kor4, 4-class | 41.00 ± **4.34** | baseline |
| H1 Kor4, 3-class | 40.67 ± **1.53** | class redundancy −2.81pp |
| G1 Kor2+Random2 | 41.08 ± **2.50** | semantic content (Ekman vs random) −1.63pp |
| D Kor2+Wes(Ekman)2 | 41.25 ± **0.87** | anchor origin (Mixed vs pure Kor) −0.66pp |
| C Wes4 Ekman | 25.42 ± 0.38 | (0 Korean → low mean) |

→ 기존 "Western ≥2 threshold" 주장 오해였음 — **class coverage가 주 요인**, origin+content는 부가.

### ⭐ FINDING 3 — ICL ⊂ LoRA (substitutability)
LoRA + Mixed k=4: 56.00 (−0.75), LoRA + Korean k=4: 52.75 (−4.00). Ordinal not additive.

### FINDING 4 — Attention entropy mechanism REJECTED (exp_019)
D_mixed (σ_out=0.87) entropy 0.645 vs E_kor4 (σ_out=4.34) entropy 0.639 — near identical at layer 14.
→ Variance mechanism은 hidden-representation level (attention 아님).

### FINDING 5 — Ekman FACS = random on Korean faces
25.42 ± 0.38% — seed-invariant strong negative result motivating anchor decomposition.

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
