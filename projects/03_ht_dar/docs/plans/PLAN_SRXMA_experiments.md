# SR-XMA 실험 계획 — GPT 정리 + 채택/기각 프로토콜
> 2026-06-05. 컨퍼런스(ICASSP) 타겟. CBBF(bio)는 KBS 별도. 이건 MOSEI/IEMOCAP efficiency 논문.

---

## A. GPT 답변 정리 (핵심만)

**중심 claim (이거 벗어나면 망함):**
> 토큰 압축(SFT)도, bottleneck fusion(MBT)도, dominant-modality 선택(MODS/KuDA)도 아니다.
> **modality×level 노드 그래프에서 어떤 cross-attention edge를 실제로 계산할지 sparse routing 한다.**

**최종 구조명**: BSR-XMA (Budgeted Sparse-Routed Hierarchical Cross-Modal Attention).
단계적으로 쌓음 (V1→V4).

**각 논문에서 차용할 부품 (주의사항 포함):**
| 출처 | 차용 | 주의 (이렇게 쓰면 망함) |
|---|---|---|
| SFT | token compression | 메인 claim 으로 두면 X |
| MBT | bottleneck **safety hub** | fusion 중심이면 MBT랑 겹침 |
| HCT-DMG | modality reliability | primary modality 선택하면 DAR 회귀 |
| MODS | sample-wise modality score | selector 아니라 **router prior** 로만 |
| KuDA | knowledge-guided bias | optional ablation 만 |
| FuseMoE | **edge-type projection** | expert routing 으로 보이면 X |
| DynMM | dynamic budget k(x) | Pareto 강화용 |
| SparseCut | cross-level shortcut | MLLM과 도메인 구분 |
| Routing Transformer | learned sparse routing | token clustering 아니라 **node-edge** routing |
| DynamicViT | differentiable **edge** mask | token mask 아님 |
| TokenLearner | 더 나은 node pooling | router summary 개선 |
| BigBird/Longformer | 구조적 mandatory edge (안전망) | random은 train/baseline만 |

**필수 loss**: task + route entropy(작게) + **modality balance(text-collapse 방지)** + budget + **Full-attn KD(teacher→sparse student, 강추)**.

**3대 리스크 (생명선)**:
1. N=9라 pair 절감 작아 보임 → **실측 FLOPs/latency/memory** 반드시.
2. text 너무 강해 router가 text로 collapse → **modality balance loss + 분포 분석** 필수.
3. **learned 가 random-k 못 이기면 논문 사망** → routing 분석이 핵심 결과.

**반드시 인용**: SFT, MBT, HCT-DMG, MODS, Routing Transformer.

---

## B. 실험 사다리 — 컴포넌트별 채택/기각 (각 단계 3-seed + 교차검증)

각 단계: **이전 best + 새 컴포넌트 1개** → 측정 → 판정. 측정 = {Acc2/F1, **inference FLOPs/latency**, edges/node}.

| # | 실험 | 무엇을 보나 | 채택 조건 | 기각 시 |
|---|---|---|---|---|
| **E0** | Full (all-edge) | 정확도·연산 **상한 기준점** | (기준, 항상 보유) | — |
| **E1** | **Learned-k vs Random-k vs Fixed-k** (k=2) | ★핵심: 학습 라우팅이 의미있나 | **learned > random** (유의) AND learned ≈ Full acc | **여기서 지면 전체 STOP.** 라우팅 무의미 |
| **E2** | k sweep (k∈{1,2,3,N}) | accuracy–FLOPs **Pareto** | learned Pareto가 random Pareto 위 | k 무관하면 라우팅 약함 |
| **E3** | + modality reliability prior (MODS/HCT) | router bias로 +오르나 | learned 대비 Acc +유의 or 같은 acc 더 낮은 k | 기각, prior 안 씀 |
| **E4** | + bottleneck hub (MBT) | sparsity로 잃은 정보 복구하나 | sparse-acc gap 줄고 FLOPs 소폭↑만 | 기각 |
| **E5** | + Full-attn KD loss | teacher 증류로 sparse 정확도↑ | learned acc → Full에 더 근접 | 기각 |
| **E6** | + dynamic budget k(x) (DynMM) | sample별 k로 Pareto 개선 | 같은 평균 FLOPs서 acc↑ | 기각 |
| **E7** (옵션) | TokenLearner pool / edge-type proj | node summary·edge expert | +유의면 채택 | 기각 |

**메인 모델 = E1~E5 중 채택된 것들의 합** (GPT 권장: SR-XMA + modality prior + bottleneck 정도가 적당, dynamic budget은 여유되면).

**메인 figure** = E2의 accuracy vs FLOPs Pareto (learned/random/fixed/Full + MBT/SFT baseline).

---

## C. 판정 기준 (정량)
- **유의성**: 3-seed paired (gate_c_runner 재사용, bootstrap CI + Wilcoxon). within-noise면 "채택 안 함".
- **효율**: inference FLOPs(fvcore/thop) + GPU latency + memory **실측**. pair count만으론 불충분.
- **채택 = 둘 중 하나**: (a) 같은 FLOPs서 Acc 유의 상승, (b) 같은 Acc서 FLOPs 유의 하락.
- **전체 성공선**: learned 가 Full acc의 ~0.5pp 안 + FLOPs ≥3×↓ + **random-k 보다 유의 우위**.
- **전체 실패선**: E1에서 learned ≈ random → 라우팅 무의미 → 이 방향도 접고 보고.

---

## D. 교차검증 원칙 (매 코드마다 — 어김 금지)
모든 컴포넌트 추가 시 **학습 전** 다음 통과해야 함:
1. AST 컴파일
2. forward shape + 4모드 동작 (edges/node 정확)
3. router/신규파라미터 **grad 유한·nonzero** (NaN 검출 — 전에 잡은 그 버그류)
4. full vs sparse **edge수/FLOPs 차이 실측**
5. 소규모(1 fold/few epoch) 수렴 확인 → 그 다음에야 3-seed

(NaN·over-smoothing·잘못된 placement 같은 거 학습 전에 잡는다. 이미 V1 이 1~4 통과.)

---

## E. 진행 순서 (지금부터)
1. ✅ V1 코어 작성·교차검증 (full/random/fixed/learned, router 학습) — **완료**
2. [ ] MOSEI stripped 연결 (`attn_module='srxma'`, route_mode 플래그) + FLOPs 훅
3. [ ] **E0+E1** (Full / Random-k / Fixed-k / Learned, k=2, 3-seed) ← 생사 가르는 첫 관문
4. [ ] E1 통과 시 → E2 Pareto → E3~E5 순차 채택/기각
5. [ ] 채택된 것 합쳐 메인 모델 + IEMOCAP 재현 + routing 시각화

각 단계 끝에 leaderboard append + 채택/기각 기록(engine).
**E1이 1순위. 거기서 learned가 random 못 이기면 즉시 멈추고 재검토.**
