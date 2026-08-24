# SR-XMA 코드 파이프라인 평가 — 한 줄씩 (2026-06-07)

> 목적: "구조는 맞는데 왜 작동 안 하나" 교차검증. 데이터→출력 순서로 3파일을 줄단위 평가.
> 태그: ✅정상 · ⚠️약함/설계미달 · 🔴버그/치명.

---

## ⭐ 결론 먼저 (실증 포함)

**learned 라우터는 학습이 안 된다. "learned ≈ random"은 발견이 아니라 구현 결함이다.**

학습된 체크포인트(`pt/STRIP_SR_LEARN_specific_s1111mosei.pth`) 실측:
- `edge_bias`: L2=0.041, absmax=0.010 — init=0에서 **거의 안 움직임**
- `Wq/Wk` std=0.0818 = **nn.Linear 기본 init std 그대로** → 사실상 미학습
- per-row 선호 std ~0.005, q·k affinity항(~scale 0.14×) 대비 **edge_bias는 ~100배 작아 무의미**

근본 원인: **hard top-k(미분 불가)** 를 학습 중에도 그대로 써서 라우터로 가는 gradient가 사실상 0.
→ 라우터가 init 난수 선택에 머묾 → learned 선택 = random 선택. E1 음성은 **false negative.**

---

## 1. `train_stripped.py` (엔트리·설정)

| 줄 | 코드 | 평가 |
|---|---|---|
| 12 | `os.environ['USE_GD']='0'` | ✅ stripped: graph distill 제거 |
| 24-31 | `EXP_CFG` dict | ✅ M0~M5 + SR_FULL/FIXED/RAND/LEARN. config-driven 깔끔 |
| 34-45 | `_patched(cfg)` 머지 | ✅ EXP_CFG→cfg, MDROP/EARLY_STOP env 주입 |
| 40 | `ht_modality_dropout=MDROP` | ✅ E1은 0(clean) |
| 41-42 | `use_ema=True, ema_decay=0.999` | ⚠️ EMA 켜짐 — 라우터 학습이 느린데 EMA가 더 뭉갤 수 있음(부차) |
| 57-63 | `_build_model` 패치로 stripped 주입 | ✅ |

**판정**: 엔트리·config 흐름은 정상. 문제 없음.

---

## 2. `DLF_HTCMA_stripped.py` (전처리·인코더·dispatch)

| 줄 | 코드 | 평가 |
|---|---|---|
| 52-54 | Conv1d proj (l/a/v → d=50) | ✅ DLF 동일 |
| 64-66 | specific TransformerEncoder ×3 (s_l,s_v,s_a) | ✅ 인코더 공유 OK |
| 69-92 | **srxma 분기 추가**(내가 넣음) | ✅ 시그니처 동일 swap, 인자 매핑 정상 |
| 89-100 | BERT→dropout→transpose→proj→(T,B,d) | ✅ |
| 110-121 | 학습 중 modality dropout | ✅ E1은 mdrop=0이라 비활성 |
| 131 | `self.ht_fusion(s_l,s_v,s_a,avail)` | ✅ SRXMAFusion 호출 정상 |
| 132-133 | `cat[3×d]→head→logit` | ✅ |
| 137 | `aux.get('beta_w_l')` (srxma엔 없음→None) | ✅ 안전 |

**판정**: 전처리·인코더·dispatch 정상. 문제 없음. **버그는 여기 없음.**

---

## 3. `sr_xma.py` (핵심 fusion — 여기가 문제)

### Stage B — HierTokens (계층 토큰) `36-53`
| 줄 | 코드 | 평가 |
|---|---|---|
| 41 | `SelfAttnBlock ×L` | ✅ 레벨별 self-attn |
| 42 | `Conv1d(d,d,3,2,1)` patch-merge | ✅ T→T/2 stride2. 동작 OK |
| 45-53 | levels 누적 | ✅ [fine,med,coarse] 9노드 생성 정상 |
| — | 레벨간 위치정보/PE 없음 | ⚠️ 약함(부차). Swin 의도엔 윈도우/shift 있어야 하나 단순 self-attn |

### Stage 노드요약 `128`
| 128 | `h = stack([z.mean(0)...])` (B,N,d) | ⚠️ mean-pool 요약. PLAN의 TokenLearner pool 대비 약하지만 치명 아님 |

### Stage F — `_route` (라우팅) `92-117` 🔴 **핵심 결함부**
| 줄 | 코드 | 평가 |
|---|---|---|
| 94-95 | `a = q·kᵀ·scale + edge_bias` | ✅ affinity 계산식 자체는 맞음 |
| 97 | self-edge `-inf` 마스크 | ✅ residual이 self 보존 |
| 109-112 | **learned: `a.topk(k).indices`로 hard 선택** | 🔴 **미분 불가.** 선택(어떤 엣지냐)으로 gradient 안 흐름 |
| 114-115 | `a_sel=mask_fill(~sel,-inf)` → `softmax` | 🔴 선택된 k개에만 grad. **안 뽑힌 엣지는 grad=0 → 영영 안 뽑힘**(탐색 불가) |
| — | `self.training` 분기 **없음** | 🔴 **주석(line15)은 "학습 시 masked-full"라는데 코드엔 train/eval 분기가 아예 없음.** 학습 때도 hard top-k → 설계의도와 코드 불일치 |
| — | entropy/balance/load-balancing loss 없음 | 🔴 PLAN "필수" 3종(route entropy, modality balance, full-attn KD) **전부 누락** |
| — | Gumbel/STE 없음 | 🔴 미분가능 선택 장치 부재 → 위 hard top-k 문제의 직접 원인 |

**→ 결과: 라우터 grad ≈ 0 → init 동결(실측 확인) → learned=random.**

### Stage H — cross-attn 루프 `131-139`
| 줄 | 코드 | 평가 |
|---|---|---|
| 133-138 | `for i: for j:` 81쌍 순차 `xattn` | ⚠️ 정확성 OK, but **Python N² 루프 = wall-clock full보다 느림** |
| 136 | `if (gij>0).any()` 배치 union | 🔴 **per-sample top-k가 배치서 union→거의 full(B=16서 92.5%)** → 효율 환상. B=1서만 실재(22%) |
| 137 | `xattn(Z[i],Z[j],Z[j])` 단일 공유 MHA | ⚠️ 모든 엣지타입 한 MHA 공유. PLAN의 edge-type projection(FuseMoE) 없음 |
| 138 | `Z_new[i] += g·out` residual | ✅ 집계식 정상 |

### Stage J — pool/출력 `141-147`
| 141-144 | node→(M,L,B,d)→level평균→3벡터 | ✅ 출력 형태 정상 |
| 145-146 | `aux['edges_per_node']` | ⚠️ per-sample 의도값(2)만 보고. 실제 계산 edge(union≈full)와 다름 → 오해 소지 |

---

## 종합 판정

| # | 문제 | 심각도 | 근거 |
|---|---|---|---|
| 1 | learned 라우터 학습 안 됨(init 동결) | 🔴 치명 | 체크포인트 실측 edge_bias≈0, Wq/Wk=init std |
| 2 | hard top-k, train/eval 분기 없음, 주석과 코드 불일치 | 🔴 #1의 원인 | `_route`에 self.training 없음 |
| 3 | entropy/balance/KD/Gumbel 전부 누락 | 🔴 | PLAN "필수" 미구현 |
| 4 | batched 효율 환상(B=1서만 실재) | ⚠️ | preflight 92.5% union |
| 5 | N² Python 루프(느림) | ⚠️ | 5 it/s, full보다 느림 |
| 6 | edge-type proj/PE/TokenLearner 부재 | ⚠️ | 모델링 약함(부차) |

**구조(아이디어)는 멀쩡하다. 구현이 미완성이라 핵심 주장("학습 라우팅이 의미있다")을 테스트조차 못 했다.**
JY 직감 정확: learned 경로는 사실상 작동 안 함.

## 다음 (V2 수정안)
1. **선택을 학습가능하게**: 학습 땐 full-soft routing(전 엣지 softmax 가중, 주석 의도대로) → 추론 땐 hard top-k. 또는 Gumbel-softmax/STE top-k.
2. **balance + entropy loss** 추가(text collapse·router 동결 방지).
3. 그 후 **E1 재실행** — 그때 learned vs random이 진짜 테스트됨.
4. 효율은 B=1 이론-FLOPs로 측정(또는 per-sample sparse gather 구현).

**현 E1 음성은 폐기(false negative).** 컨퍼런스각 판단은 V2 고친 뒤 E1로 미룬다.
