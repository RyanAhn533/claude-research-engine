# 설계 문서 — Sparse-Routed Hierarchical Cross-Modal Attention
> 2026-06-05. JY 원설계 정본. 구현 전 명세. (이전 DAR=앵커선택은 폐기 — 이건 efficiency 라우팅)

작업명(가칭): **SR-XMA** (Sparse-Routed Cross-Modal Attention)
또는 논문 제목용: *"Routing the Cross-Attention: Efficient Hierarchical Multimodal Fusion via Learned Sparse Pair Selection"*

---

## 0. 한 줄
각 modality를 Swin식 계층 토큰으로 만들고, (modality×level) 노드들 간 cross-attention을
**전부(n²) 하지 않고, 라우터가 중요한 쌍만 골라(sparse) 계산**해서 연산량을 줄이면서 성능은 유지한다.

## 1. 목표 (무엇을, 왜)
- **문제**: 계층(level)×modality 노드가 N=M·L개. 이들 간 dense cross-attention = **O(N²) 쌍** × 토큰 attention.
  토큰까지 치면 매우 비쌈. (full all-pairs는 baseline이자 상한)
- **기여**: 라우터가 **query 노드마다 top-k 핵심 key 노드만 선택** → cross-attention을 **O(N·k), k≪N** 로 축소.
- **승부 축 = 효율(FLOPs/latency/params) ↓ + 정확도 유지** (정확도로 SOTA 안 노림 — 그건 MOSEI 천장).
  → "같은 성능을 훨씬 적은 연산으로" = Pareto 개선. ICASSP-friendly, 비-saturated 축.
- **부차 주장**: 학습된 sparse 라우팅이 random/heuristic 라우팅보다 우수 (어떤 쌍이 중요한지 학습).

## 2. Notation
- modality m ∈ {l(text), a(audio), v(video)}, |M|=3.
- level l ∈ {0(fine),1(med),2(coarse)}, |L|=3.
- 노드 = (m,l). 노드 수 N = M·L = 9.
- 노드 토큰 집합: Z_{m,l} ∈ R^{T_l × d}  (level l 의 토큰 길이 T_l = T/2^l).
- 노드 요약 벡터: z_{m,l} = Pool(Z_{m,l}) ∈ R^d  (attentive pool).

## 3. 아키텍처 (단계별)

### Stage A — 인코더 (재사용)
text=BERT, audio/video=feature → Conv1d proj → modality별 Transformer → s_m ∈ R^{T×d}.

### Stage B — 모달리티별 Swin 계층 토큰 (★ 진짜 Swin)
각 s_m 을 **윈도우 self-attention + patch merging** 으로 L단계:
```
Z_m^(0) = SwinBlock_0(s_m)                 # fine,  T 토큰
Z_m^(1) = SwinBlock_1(PatchMerge(Z_m^(0))) # med,   T/2
Z_m^(2) = SwinBlock_2(PatchMerge(Z_m^(1))) # coarse,T/4
```
- SwinBlock = window-MSA + shifted-window-MSA + MLP (윈도우 내 지역 self-attention).
- PatchMerge = 인접 2토큰 concat→Linear (T→T/2).
- 결과: 노드 9개 Z_{m,l}. **여기서 계층 attention이 구조적으로 일어남(네 원래 비전).**

### Stage C — ★ Routed Sparse Cross-Attention (핵심 기여)
모든 노드쌍 (i,j), i=(m,l), j=(m',l') 에 대해 dense 하게 안 하고:

**C-1. 라우터 affinity (싼 연산, pooled 벡터로):**
$$a_{ij} = \frac{(W_q\, z_i)^\top (W_k\, z_j)}{\sqrt{d}} + b_{\text{type}(i,j)}$$
- b_type = edge-type bias (intra-modal / inter-modal / same-level / cross-level 구분, 학습).
- 비용 O(N²·d) (토큰 아님, 노드요약끼리 → 매우 쌈).

**C-2. top-k 선택 (query 노드마다):**
$$S_i = \text{TopK}_j\big(a_{ij}\big),\quad |S_i| = k \ll N$$
자기 자신 포함(intra-modal self 보존). missing modality 는 후보에서 제외.

**C-3. 선택된 쌍만 토큰-레벨 cross-attention:**
$$\tilde{Z}_i = Z_i + \sum_{j \in S_i} g_{ij}\, \text{CrossAttn}(Q{=}Z_i,\,K{=}Z_j,\,V{=}Z_j)$$
- 게이팅 가중치 $g_{ij} = \text{softmax}_{j\in S_i}(a_{ij})$.
- **여기만 비쌈**(O(T_i·T_j·d)), 근데 노드당 k개만 → 전체 O(N·k·T²d).

### Stage D — 융합 → head
각 노드 풀링 z̃_{m,l} → modality별로 level 합치고(평균 or 학습가중) → modality 벡터 3개 → concat → head.
(※ Beta level-gate 는 네 설계 아니므로 기본 제거. 필요시 modality 신뢰도 gate 만 옵션.)

## 4. 복잡도 분석 (효율 주장의 근거 — 논문 핵심 표)
| 방식 | cross-attn 쌍 수 | 토큰 attn 비용 |
|---|---|---|
| Full (all-pairs) | N² = 81 | O(N²·T²·d) |
| **SR-XMA (ours)** | N·k (k=2~3) | **O(N·k·T²·d)** |
| 절감 | k/N ≈ 0.22~0.33 | **~3–5× FLOPs↓** |
- router overhead O(N²·d) 는 토큰 attn 대비 무시 가능.
- 논문 메인 figure = **accuracy vs FLOPs Pareto** (full 대비 성능 유지 + 연산 ↓).

## 5. 학습 (미분 가능 top-k 문제)
top-k 는 비미분 → 3택:
1. **soft→sparse annealing (권장)**: 학습 초기 soft(전체 softmax 가중), τ↓ 하며 점점 sparse. 추론 시 hard top-k.
2. Gumbel-softmax + straight-through (top-k 근사).
3. sparsemax/entmax (희소 softmax, 미분가능).

**손실:**
$$\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{route} + \lambda_2 \mathcal{L}_{budget}$$
- L_route: router entropy/diversity reg (collapse 방지, 한 쌍에만 몰리지 않게).
- L_budget: 선택 쌍 수를 k 근처로 유지(예산 제약) — 효율 보장.

## 6. 실험 (efficiency-accuracy)
**데이터**: MOSEI, IEMOCAP (표준, text+audio+video). CBBF(bio)와 안 겹침.
**비교군 (routing 방식 고정 backbone 위에서):**
| 변형 | 라우팅 | 목적 |
|---|---|---|
| Full | all-pairs (k=N) | 정확도 상한 + 연산 상한 |
| Random-k | 랜덤 k쌍 | "학습 라우팅이 의미있나" |
| Fixed-k | 휴리스틱(인접level+language-anchor) | 단순 규칙 대비 |
| **SR-XMA** | 학습 top-k | 제안 |
**측정**: Acc2/F1 + **FLOPs, params, latency.** 3-seed.
**핵심 결과**: SR-XMA 가 Full 정확도 ≈ 유지하면서 FLOPs 3–5×↓, random/fixed 보다 정확.
**Ablation**: k ∈ {1,2,3,N}, edge-type bias on/off, soft-only vs annealed, Swin window size.

## 7. Novelty 위치 (정직)
- 앵커 선택(KuDA/MODS) ❌ 아님 — 우리는 **cross-attention 쌍 선택(sparse)**.
- sparse attention(Routing Transformer/Reformer)·MoE-attention 존재하나, **계층(modality×level) multimodal cross-attention 에 적용 + efficiency framing** = 덜 점유된 니치.
- 정직 포지셔닝: "새 sparse-attn primitive"가 아니라 **"hierarchical multimodal fusion 을 learned pair-routing 으로 효율화"** = mechanism+efficiency 논문 (ICASSP/mid-tier 적합).

## 8. 구현 계획 (파일·클래스·순서)
파일: `/home/ajy/HT-DAR/code/trains/singleTask/model/` 에 신규 `sr_xma.py`.
1. `TemporalSwin(d, window, depths)` — modality별 Swin 계층 토큰 (Stage B). [visionmer TSE 참고]
2. `PairRouter(d, N, k)` — affinity + top-k 선택 + soft/anneal (Stage C-1,2).
3. `RoutedCrossAttn(d, heads)` — 선택 쌍 토큰 cross-attn (Stage C-3).
4. `SRXMAFusion` — B+C+D 조립, full/random/fixed/learned 플래그.
5. DLF_HTCMA_Stripped 에 `attn_module='srxma'` 추가 (인코더 재사용).
6. FLOPs 측정 훅(fvcore/thop) + 학습 스크립트.
**검증 순서(매 단계 수치 확인)**: Swin forward shape → router top-k 동작 → routed-xattn grad → full vs k FLOPs 차이 → 소규모 학습 수렴 → 3-seed.

## 9. 성공/실패 기준
- 성공: SR-XMA acc ≈ Full (within ~0.5pp) AND FLOPs ≥3×↓ AND > random/fixed. → ICASSP 논문.
- 실패: sparse 가 정확도 많이 깎거나 random 과 차이 없으면 → 라우팅 무의미, 재검토.
