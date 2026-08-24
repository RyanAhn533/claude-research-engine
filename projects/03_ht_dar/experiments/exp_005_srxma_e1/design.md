# exp_005 — SR-XMA E1+E0 (생사 관문: learned-k > random-k ?)

> 2026-06-06. ICASSP efficiency 논문. MOSEI stripped, MDROP=0(clean), share=specific, d=50/heads=10.
> PLAN: `docs/plans/PLAN_SRXMA_experiments.md` E0/E1. 코드: `/home/ajy/HT-DAR/code/trains/singleTask/model/sr_xma.py`.

## 무엇을
(modality×level) 9노드 그래프에서 cross-attention edge를 4가지로 라우팅, MOSEI Acc2 비교:
- **SR_FULL** (k=8, all-edge): 정확도 상한 / E0 기준점
- **SR_FIXED** (k=2, 휴리스틱): 인접level + same-level cross-modality
- **SR_RAND** (k=2, 랜덤): ★null baseline — "학습 라우팅이 의미있나" 기준선
- **SR_LEARN** (k=2, 학습 top-k): 제안

각 4모드 × 3seed(1111/1112/1113). best-val(min val loss) epoch의 TEST acc_2 채택.

## hypothesis

```yaml
hypothesis_id: H_srxma_e1_learned_beats_random
primary: "MOSEI Acc2에서 SR_LEARN(학습 top-k=2) > SR_RAND(랜덤 k=2), 3-seed paired delta >= 0.5pp."
null: "SR_LEARN - SR_RAND <= 0 (학습 라우팅이 랜덤 대비 이득 없음)."
success_criterion:
  metric: mosei_acc2
  direction: higher_is_better
  delta_threshold: 0.005   # 0.5pp, pre-registered
secondary:
  - "SR_LEARN ≈ SR_FULL (정확도 상한의 ~0.5pp 안) → sparsity로 정확도 거의 안 잃음"
failure_implication: >
  learned ≈ random 이면 라우팅 무의미 → SR-XMA 방향 STOP, 03_ht_dar 컨퍼런스 접음.
  HT-DAR(동적 앵커)에 이어 sparse-edge 라우팅까지 random과 동급이면 이 라인 사망.
falsifiability_check: PASS   # CI가 0을 포함하면 명확히 기각
```

## 측정 주의 (preflight에서 확정된 함정)

`tools/srxma_preflight.py` 결과: V1은 per-sample top-k라 **batched(B=16)에선 union이 거의 full(92.5%)**, 효율은 **B=1 inference에서만 실재**(learned=full의 22% 이론 FLOPs).
→ **이 실험의 정확도 비교(learned vs random vs full)는 유효.** 효율 주장(메인 figure Pareto)은 B=1 이론-FLOPs로만 방어. batched wall-clock은 N²루프 때문에 오히려 느림 → 진짜 latency는 sparse-gather V2 필요(이 실험엔 안 막힘).

## 채택/기각
- E1 PASS = learned vs random: gate_c paired CI 하한 > 0 AND Δ>=0.5pp → SR-XMA 계속(E2 Pareto로)
- E1 NEUTRAL/FAIL = CI가 0 포함 → STOP, negative 기록
