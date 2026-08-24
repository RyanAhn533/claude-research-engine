# SR-XMA 구조 — 코드 파이프라인 walkthrough (2026-06-08)

> 실제 코드 기준, 데이터 흐름 순서로 파일 하나씩. MOSEI 기준 shape: d=50, heads=10, T≈50, B=16.
> 코드 위치: `/home/ajy/HT-DAR/code/` (engine 밖). V2 라우터 수정 반영본.

---

## 파이프라인 한눈

```
text/audio/video
  → [Conv1d proj + Transformer enc] ×3          (표준 인코더)
  → 각 modality를 3-level 계층 토큰화 → 9노드     (① HierTokens)
  → 9노드 요약 → 라우터가 노드쌍 affinity 계산    (② _route ★고유)
  → top-k 엣지만 골라 node-to-node cross-attn     (③ 루프 ★고유)
  → 노드→레벨평균→3 modality 벡터 → head           (④ pool)
```
**고유부 = ②③ (modality×level 9노드 그래프 + edge routing).** ①은 pyramid 차용, 인코더·cross-attn은 표준.

---

## ① `train_stripped.py` — 진입/설정 (배선, 구조 아님)

- `EXP_CFG[exp]` dict = 모델 config. SR-XMA = `SR_FULL/FIXED/RAND/LEARN`:
  `ht_attn_module='srxma'` + `srxma_route_mode={full|fixed|random|learned}`, `ht_num_levels=3`, `srxma_topk=2`.
- env: `MDROP`(modality dropout), `EARLY_STOP`(수렴 후 종료).
- `run._build_model` 패치 → `DLF_HTCMA_Stripped(args)` 인스턴스화.

## ② `DLF_HTCMA_stripped.py` — 인코더 + fusion dispatch

forward(text, audio, video):
```
text (B,T,768) → BertTextEncoder → (B,T,768)                       [27-32,90-91]
audio (B,T,74), video (B,T,35)
  → transpose → Conv1d proj(→d=50) → permute → (T,B,50)            [52-54,96-100]
  → modality별 TransformerEncoder ×3                                [64-66,102-103]
  → s_l, s_v, s_a  각 (T,B,50)
  → (mdrop>0이면 학습 중 랜덤 modality drop; E1은 0)                  [110-121]
  → self.ht_fusion(s_l,s_v,s_a,avail)   = SRXMAFusion 또는 HT-DAR    [69-92,131]
  → cat[last_h_l,v,a] (B,150) → head(Linear 150→50→1) → logit        [84-87,132-133]
return {output_logit, aux_loss(V2 routing loss), beta_w_l, dar_r}     [135-138]
```
**여기까지 표준 인코더.** 구조 핵심은 `ht_fusion` 한 줄. srxma 분기는 `__init__` [69-92]에서 swap.

## ③ `sr_xma.py` `SRXMAFusion` — 구조 본체

### Stage B — 계층 토큰화 `HierTokens` [36-53, 139-143]
각 modality s_m (T,B,50) → 3레벨:
```
L0 = SelfAttnBlock(s)               # (T,  B,50)  fine
L1 = SelfAttnBlock(Conv1d_s2(L0))   # (T/2,B,50)  medium   ← patch-merge stride2
L2 = SelfAttnBlock(Conv1d_s2(L1))   # (T/4,B,50)  coarse
```
3 modality × 3 level = **9 노드** `Z` (각 토큰 묶음, 길이 다름).
- `SelfAttnBlock` [22-33] = MHA + LN + MLP (표준 block).
- "Swin식"이라지만 실제론 윈도우/shift 없는 self-attn + Conv 다운샘플.

### 노드 요약 [145]
각 노드 시간평균 → `h` (B, 9, 50). 라우터 입력.

### Stage F — 라우터 `_route` [99-134] ★핵심 (+V2)
```
a[i,j] = (Wq·h_i)·(Wk·h_j)/√50 + edge_bias[i,j]   # (B,9,9)            [101-102]
self-edge -inf 마스크                                                   [104]
sel(어떤 엣지 계산):
  full   : 전 8 엣지                                                    [106-107]
  fixed  : 휴리스틱(인접level+같은level 타modality)  _build_fixed_mask   [87-97,108-109]
  random : 랜덤 top-2                                                   [110-115]
  learned: ★V2 — 학습=전 엣지 dense-soft / 추론=hard top-2              [116-124]
temp annealing(learned 학습만, high→temp_min)                           [126-130]
g = softmax(a/temp); 선택 안된 곳 0                                      [131-133]
```
→ `g` (B,9,9) = 노드 i가 j를 attend하는 게이트.
**V1 버그**: learned가 학습 중에도 hard top-k였음 → 라우터 grad≈0 → init 동결.
**V2 수정**: 학습=dense-soft(전 엣지 미분가능)로 라우터 학습되게 + 추론만 sparse.

### Stage H — 선택 엣지 cross-attention [148-156]
```
for i in 9:
  for j in 9:
    if g[i,j]>0:
      out = MHA(Q=Z[i], K=Z[j], V=Z[j])     # 노드 i가 노드 j 토큰 attend
      Z_new[i] += g[i,j] * out
Z_new = LayerNorm(Z_new)
```
- 단일 공유 `xattn` MHA [79]가 모든 엣지에 쓰임 (edge-type별 분리 없음).
- Python 9×9 이중 루프 = 느림 + batched서 union이 거의 full(효율 환상).

### Stage J — 출력 [157-161]
```
각 노드 시간평균 → (9,B,50)
→ reshape (3 modality, 3 level, B, 50)
→ level 평균 → (3,B,50)
→ last_h_l, last_h_v, last_h_a  각 (B,50)
```

### V2 추가 — routing 정규화 loss [162-175]
learned 학습 시: entropy(per-row peak↓) + balance(key 균등, anti-collapse) → `aux['aux_loss']`, `train_step++`.

## ④ `DLF.py` — 트레이너 [152-156]
```
stripped 감지(logits_l_hetero 없음):
  combined_loss = L1(logit, label) + aux_loss(V2 routing reg)   ← aux_loss 주입 추가
  → backward
```

---

## 실험이 보여준 것 (구조 평가)

| 모드 | MOSEI acc_2 (1seed) |
|---|---|
| FULL (72엣지, 상한) | 0.8465 |
| RANDOM (top-2) | 0.8434 |
| LEARNED-V2 (라우터 살림) | 0.8484 |
| LEARNED-frozen (V1 버그) | 0.8478 |

→ 전부 0.5pp 노이즈. **full조차 random 못 이김 = 어떤 엣지 고르든 정확도 무관(MOSEI text saturation).**
→ 구조의 고유부(②③ edge routing)가 이 데이터선 작동 안 함. 구조가 틀린 게 아니라 무대(데이터)가 틀림.

## novelty 채점
- ① 계층 그래프 토폴로지: owned (HFGCN/MSGFN/MHRG/HiGraMI)
- ② sparse routing: owned (Routing Transformer/BigBird)
- ④ cross-modal attn: 표준
- "modality×level 그래프 edge routing" 조합: △ 얇게 미점유(mechanism delta), but 실험이 그 슬라이스를 무력화.
- **결론: 약한 incremental novelty는 있으나 functional novelty(새 부분이 실제로 함)는 현재 0.**

관련: [[CODE_REVIEW_srxma_pipeline_20260608]](../findings/CODE_REVIEW_srxma_pipeline_20260607.md), NO_GO/gap 결과.

---

## 데이터별 전처리 + 스코어

### A. MOSEI (CMU-MOSEI, sentiment)
**전처리** (`config/config.json` datasetCommonParams.mosei):
- feature: `MOSEI/Processed/aligned_50.pkl` (MMSA 표준, **word-aligned 50 timestep**)
- dims `[768, 74, 35]` = **text BERT-base 768 / audio COVAREP 74 / vision Facet 35**
- text는 학습 시 `bert-base-uncased`로 재인코딩(`transformers='bert'`)
- 샘플 16,326(train) · label = sentiment 회귀값[-3,3] → **acc_2 = 부호 binary(pos/neg)**
- 모델측: Conv1d proj→d=50, heads=10, batch=16, lr=1e-4, grad_clip=0.6, patience=5

**스코어 — HT-DAR 사다리 (앵커 라우팅, 死, acc_2)**
| 변형 | acc_2 | n_seed | 판정 |
|---|---|---|---|
| M0 (meanpool) | 0.8528 ± 0.0036 | 5 | 기준 |
| M1 (hier L=3) | 0.8524 ± 0.0066 | 5 | neutral (Δ−0.04pp) |
| M2/M3/M4 (dar/evid) | 0.849~0.853 | 2/2/1 (불완전) | 미로깅 |

**스코어 — SR-XMA (efficiency 라우팅, 현재, acc_2, 1-seed, EARLY_STOP=8)**
| 모드 | acc_2 | 비고 |
|---|---|---|
| FULL (72엣지, 상한) | 0.8465 | 라우팅 없음(전부 계산) |
| RANDOM (top-2) | 0.8434 | null baseline |
| LEARNED-V2 (라우터 살림) | 0.8484 | dense-soft 학습 |
| LEARNED-frozen (V1 버그) | 0.8478 | 라우터 동결 |
→ **전부 0.5pp 노이즈. full조차 random 못 이김 = routing headroom 0 (text saturation).**

**스코어 — robustness/modality-dropout (HT-DAR NO-GO 근거, acc_2)**
| 모델 | clean(3seed) | 100%-drop(1seed) |
|---|---|---|
| static(고정) | 0.850 | **0.647 (최고)** |
| dar(동적) | 0.848 | 0.635 |
| hgar(그래프) | 0.852 | 0.636 |
→ 결손 학습 하에서 static≈dar≈hgar, 동적/그래프 이득 없음.

### B. IEMOCAP (MulT-style, 4-class emotion)
**전처리** (`train_iemocap.py`, `iemocap_data.pkl`):
- feature dims `[300, 74, 35]` = **text GloVe-300 / audio COVAREP 74 / vision Facet 35**
- **seq_len 20** (text (N,20,300)), 정렬됨
- label = 4-class **anger / happy / neutral / sad** (one-hot argmax)
- **class imbalance → weighted CrossEntropy** (class_counts 역가중)
- 모델측: d=40, heads=8 (MOSEI와 다름), 4-class head 패치
- metric = **weighted F1 (wF1)**

**스코어 — HT-DAR 사다리 (5-seed, wF1)**
| 변형 | wF1 | 판정 |
|---|---|---|
| M0 (baseline) | 0.5820 ± 0.0106 | 기준 |
| M1 (hier) | 0.5827 ± 0.0072 | neutral (Δ+0.07pp) |
| M2 (dar L1) | 0.5839 ± 0.0087 | neutral (+0.19pp) |
| M3 (hier+dar) | 0.5799 ± 0.0052 | neutral (−0.21pp) |
| M4 (full+evid) | 0.5897 ± 0.0184 | neutral (+0.78pp) |
| A1_parallel | 0.5904 ± 0.0185 | neutral (+0.84pp, p=0.57) |
→ 전부 within-noise. (IEMOCAP엔 **SR-XMA 미실행** — HT-DAR 사다리만 있음.)

**요약**: 두 데이터 모두 **변형 간 정확도 차이가 노이즈**. MOSEI는 SR-XMA full≈random으로 "라우팅 천장 0"까지 확인, IEMOCAP은 HT-DAR 사다리가 전부 tie. → 이 벤치마크들은 fusion 변형에 둔감(saturated).

---

## 부록 — 네 원래 아이디어 (JY 원설계, `DESIGN_SparseRoutedXAttn.md` 2026-06-05)

### 한 줄
> 각 modality를 **Swin식 계층 토큰**으로 만들고, (modality×level) 노드들 간 cross-attention을
> **전부(N²) 하지 않고 라우터가 중요한 쌍만 골라(sparse) 계산**해서 **연산량↓ + 성능 유지.**

### 승부 축 (네가 명시)
- **효율(FLOPs/latency/params)↓ + 정확도 유지** — 정확도 SOTA 안 노림(MOSEI 천장 인정).
- 메인 figure = **accuracy vs FLOPs Pareto**. ICASSP/mid-tier 적합.
- 부차 주장: 학습 sparse 라우팅 > random/heuristic.

### 원설계 아키텍처 (Stage A~D)
- **A 인코더**: BERT/feature → Conv1d → modality별 Transformer → s_m.
- **B 계층 토큰**: ★**진짜 Swin** (window-MSA + **shifted-window**-MSA + **PatchMerge**) 3단계 → 9노드.
- **C 라우팅(핵심)**: C-1 affinity `a_ij = (Wq z_i)·(Wk z_j)/√d + b_type(i,j)` (**edge-type bias**: intra/inter-modal·same/cross-level 구분) → C-2 query마다 top-k → C-3 선택 쌍만 토큰 cross-attn.
- **D 융합**: 노드 풀 → modality별 level 합 → concat → head.
- **§5 미분가능 top-k (★네가 처방해둔 부분)**: top-k는 비미분 → **① soft→sparse annealing(권장)** / ② Gumbel-ST / ③ sparsemax. 추론 시 hard top-k.
- **§5 손실**: `L = L_task + λ1·L_route(entropy/diversity, collapse 방지) + λ2·L_budget(쌍 수 ≈k)`.

### 원설계 vs 실제 구현 — 뭐가 잘렸나 (이게 핵심)

| 원설계(JY) | V1 구현 | V2 수정 |
|---|---|---|
| B: **진짜 Swin** (window+shifted-window+PatchMerge) | ❌ 윈도우/shift 없는 plain self-attn + Conv1d | 미수정 |
| C-1: **edge-type bias** (타입별 구분) | ⚠️ 그냥 flat `edge_bias` (N×N) | 미수정 |
| 노드 요약: **attentive pool** | ⚠️ mean pool | 미수정 |
| **§5 미분가능 top-k (anneal/Gumbel/sparsemax)** | 🔴 **통째로 무시 — hard top-k 학습/추론 동일** | ✅ **①anneal 복원**(학습 dense-soft→추론 hard) |
| **§5 L_route(entropy) + L_budget** | 🔴 **둘 다 없음** | ✅ entropy+balance 추가 (L_budget은 아직) |

### 정직한 결론
**네 원설계는 멀쩡했다. 라우터가 죽는 실패모드를 §5에서 이미 예견하고 처방(annealing/Gumbel/reg loss)까지 적어놨다.**
근데 **V1 구현(이전 세션)이 §5 전체를 스킵하고 hard top-k로 짜서 라우터가 init에 동결됐다** — 즉 버그는 "아이디어 결함"이 아니라 **"설계서 핵심 단계 미구현"**이었다.
V2가 §5 ①(anneal)+reg를 복원해서 라우터를 살렸고, 그 결과 **"라우터를 제대로 살려도 MOSEI에선 full조차 random을 못 이긴다"**(=데이터 천장)가 깨끗이 드러났다. 원설계 §9 실패기준("random과 차이 없으면 라우팅 무의미")에 정확히 걸린 것.
