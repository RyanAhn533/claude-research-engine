# exp_004 — HGAR: Hierarchical Graph Anchor Router (설계 문서)

status: design (구현 전)
date: 2026-06-03
목표: HT-DAR의 라우터를 "독립 softmax gating" → "modality×level 그래프 위 message-passing 라우팅"으로
격상. novelty 한 칸 올리기 + robustness 무대와 결합. 컨퍼런스(ICASSP/ACII) 코어.

---

## 0. 한 줄
앵커 라우팅을 **(modality, level) 9-노드 그래프** 위 GNN으로 계산한다. 노드는 cross-modal(같은 level)
+ cross-level(같은 modality fine↔coarse) 엣지로 메시지 패싱 → 각 노드가 "다른 modality와 다른
시간해상도까지 보고" 앵커 가중치를 정함. 기존엔 level별 독립 + modality별 독립이라 이 정보가 없었음.

## 1. 동기 (왜 그래프 + 계층)
- 현 라우터: 샘플마다 modality m에 점수 매겨 softmax. 점수는 modality 독립(holo만 약한 상호작용),
  level 독립(level마다 DynamicAnchorCMA 따로). → "그냥 gating" = novelty 약함.
- 가설: 어느 modality를 앵커로 쓸지는 (a) **다른 modality 상태**(예: text 깨지면 audio로)와
  (b) **시간 해상도 간 일관성**(coarse에서 신뢰 높은 modality가 fine에서도 앵커일 확률)에 의존.
  이 둘을 명시적 그래프 엣지로 모델링 = 구조적 기여.

## 2. 아키텍처

### 2.1 노드
- N = M × L = 3 modality × 3 level = **9 노드**.
- 노드 feature 초기값: `x_{m,l}^0 = TemporalPool(X_m^{(l)})` ∈ R^d  (level l, modality m의 시간평균; (B,d))
  - X_m^{(l)} = 기존 Hierarchical pyramid 출력 (fine/medium/coarse). 새로 안 만듦, 재사용.

### 2.2 엣지 (2 타입, 고정 구조)
- **E_modal** (cross-modal, 같은 level): (m,l)—(m',l), m≠m'.  "이 해상도에서 누가 앵커?"
- **E_level** (cross-level, 같은 modality, 인접): (m,l)—(m,l±1).  "fine↔coarse 신뢰 전파"
- 방향 무시(무향). 각 타입 인접행렬 A_modal, A_level (N×N), row-normalize.

### 2.3 Message passing (relational GCN, K=2층)
```
x^{k+1}_{m,l} = LN( x^k_{m,l}
                + ReLU( W_self x^k_{m,l}
                        + W_modal · mean_{m'≠m} x^k_{m',l}      # cross-modal 이웃
                        + W_level · mean_{l'~l} x^k_{m,l'} ) )  # cross-level 이웃
```
- W_self, W_modal, W_level ∈ R^{d×d} (타입별 weight = relational). LayerNorm + residual.
- 경량: K=2, d 그대로. 추가 파라미터 ≈ 3·d² ·K ≈ 작음.

### 2.4 라우팅 (Co-Belief을 GNN-refined 노드 위에서)
GNN 출력 x^K_{m,l} 로:
```
p_{m,l}   = softmax(mono_mlp(x^K_{m,l}))         # (B,Kc) per node, Kc=routing pseudo-class
c_mono    = max_c p_{m,l}                          # (per node)
p_bar_l   = mean_m p_{m,l}                         # level별 consensus
c_holo    = -KL(p_{m,l} ‖ p_bar_l)                # consensus 일치도
cobelief  = γ·c_mono + (1-γ)·c_holo               # (L,M,B)
cobelief  = (cobelief - mean_m)/(std_m+ε)         # cross-modal 정규화 (grad leverage, exp_002 fix)
r_{m,l}   = softmax_m( cobelief / τ )             # level별 modality 앵커 분포, τ=learnable
```
- exp_002에서 고친 vanishing-grad fix(정규화+temp) 그대로 계승.

### 2.5 Fusion (기존과 동일 인터페이스)
```
H_mix^l = Σ_m r_{m,l} · AnchorCMA_m^l(X_m^{(l)}, X_{-m}^{(l)})   # level별 mixture (기존 mixed 경로)
→ Cross-Level Context → Beta gate → heads
```
- AnchorCMA(modality m을 query로 cross-attention)는 기존 DynamicAnchorCMA의 attn 재사용.
- 즉 **바뀌는 건 r 계산뿐**. 나머지 파이프라인 그대로 → 위험 표면 최소.

## 3. 코드 변경 (정확히)
파일: `/home/ajy/HT-DAR/code/trains/singleTask/model/DLF_HTCMA.py`
1. 신규 `class HierGraphAnchorRouter(nn.Module)`:
   - `__init__(d, M=3, L=3, Kc=4, n_gnn=2)`: W_self/W_modal/W_level (ModuleList), mono_mlp(ModuleList per modality 공유 or per node), gamma, log_temp. 고정 인접 mask는 buffer.
   - `forward(pooled: (L,M,B,d)) -> r: (L,M,B)`: 위 2.3~2.4.
2. `HTCMABLF_Fusion`:
   - `attn_module='hgar'` 추가. pyramid+per-level AnchorCMA는 그대로, 라우팅만 HGAR로.
   - 기존 per-level pooled (`pooled_anchors`)을 (L,M,B,d)로 모아 router에 전달, 반환 r로 H_mix 계산.
   - `aux['hgar_r']` 로 라우팅 노출(시각화·robustness 분석용).
3. `DLF_HTCMA_Stripped` / `model_mm_v3_htdar`: `attn_module='hgar'` 패스스루 (이미 getattr로 받음).
4. train_stripped EXP_CFG에 `M5: dict(attn_module='hgar', num_levels=3, gate_type='beta', mixture_mode='mixed', output_mode='broadcast')`.

## 4. 검증 (스모크 → 실험)
1. **스모크**: dummy (L,M,B,d) → r shape (L,M,B), col-sum=1, mono_mlp/W_* grad>0 (라우터 학습 확인).
2. **MOSEI/IEMOCAP clean**: M1(static) vs M3(dar) vs M5(hgar) 3-seed. (clean은 tie 예상 — 정직 보고)
3. **★Robustness (메인)**: text-drop {0,50,100%} inference 마스킹 → degradation 곡선.
   가설: text 결손 시 static(text-anchor) 붕괴 > dar > **hgar 가장 robust** (그래프가 audio/video로
   메시지 패싱). robustness gap = 논문 메인 결과.
4. **K-EmoCon**: dar vs hgar CCC (지배 modality 없는 무대).
5. **Ablation**: hgar에서 (a) E_level 제거 (b) E_modal 제거 (c) GNN 1층 → 각 엣지/깊이 기여 분리.

## 5. 가설 (pre-register 예정, 구현 후)
- H_hgar_robust: text-drop 50%+ 에서 hgar acc − static acc ≥ +Δ (Δ TBD), p<0.05.
- H_hgar_clean: clean MOSEI에서 hgar ≈ static (within-noise, 정직).
- H_hgar_edges: E_level + E_modal 둘 다 제거 시 hgar → dar 수준으로 회귀 (엣지가 active ingredient).

## 6. 리스크 (정직)
- 9노드는 작다 → "structured routing"으로 프레이밍, "GNN 스케일 필요" 주장 금지.
- 파라미터↑ → overfit (특히 K-EmoCon). 경량 유지 + dropout.
- clean 정확도는 여전히 안 이긴다(text 지배 별개 문제). **robustness가 win 무대**임을 명확히.
- "그래프가 진짜 일하나" = ablation(E 제거)로 증명 안 하면 "장식 GNN" 공격.

## 7. 차별화 (CBBF 저널과)
- CBBF(KBS) = bio-anchor 고정 + Beta gate (생체 중심, DEAP/K-EmoCon).
- 본 컨퍼런스 = **앵커를 학습**(그래프-계층 라우터) + **결손 robustness**. 메커니즘·무대 다름.
