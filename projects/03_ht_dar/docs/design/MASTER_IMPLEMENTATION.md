# HT-DAR 컨퍼런스 통합 구현 문서 (MASTER)
> 2026-06-03. 흩어진 doc 통합. 이거 하나로 "무엇을·어떻게·왜" 다 본다.
> deep-dive: exp_002(라우터 버그), exp_003(K-EmoCon), exp_004(HGAR), CONFERENCE_PAPER_PACKAGE.

---

## 0. 목표 & thesis
- **타겟**: ICASSP 2027 (~9월 마감) / fallback ACII. mid-tier. (top-tier 불가 — novelty incremental)
- **thesis 한 줄**: *멀티모달 감정 융합은 앵커 modality를 고정한다(MSA=language, bio-centric=physiology).
  지배 modality가 없거나 결손되면 앵커는 sample마다 달라야 한다 → 앵커를 **학습**한다 (그래프-계층 라우터).*
- **win 무대**: clean 정확도는 tie(정직 보고). **modality 결손 robustness**에서 동적앵커 > 고정앵커 = 메인 결과.

---

## 1. 전체 아키텍처 (data flow)
```
                 modality m ∈ {language/video, audio, bio}   (데이터셋별)
                              │  per-modality encoder (재사용)
                              ▼
        [Step4] Hierarchical Temporal Pyramid  X_m → X_m^(fine/med/coarse)   T→T/2→T/4
                              │
        [Step6] per-level Anchor Cross-Attention  H_m^l = CMA(m, others)
                              │
        [Step5-7] ★ ANCHOR ROUTER  r_{m,l}  ← 여기가 기여 (3단 진화, §3)
                  H_mix^l = Σ_m r_{m,l} · H_m^l
                              │
        [Step8] Cross-Level Context (level 간 attention)
                              │
        [Step9] Beta-gated Level Fusion → z_fused
                              │
                          heads (sentiment / arousal·valence·quadrant)
```
- backbone(인코더~Step9)은 **모든 앵커 정책에서 동일** → 비교 공정. 바뀌는 건 라우터(Step5-7)뿐.

---

## 2. 컴포넌트 구현 상태 + 파일
코어 융합: `/home/ajy/HT-DAR/code/trains/singleTask/model/DLF_HTCMA.py`

| 컴포넌트 | 클래스 | 상태 |
|---|---|---|
| Hierarchical pyramid | `TemporalPatchMerge1D` / `ParallelTemporalLevels` | 기존 ✓ |
| per-level CMA | `IntraLevelCMA`(static) / `DynamicAnchorCMA`(dar) | 기존+수정 ✓ |
| **고정 라우터(DAR) 수정** | `DynamicAnchorCMA` | exp_002 수정 완료 ✓ |
| **★ 그래프-계층 라우터** | `HierGraphAnchorRouter` (신규) | 1차 구현, 스모크 대기 |
| Beta gate | `BetaGate`/`EvidentialBetaGate`/`SoftmaxGate`/`UniformGate` | 기존 ✓ |
| Fusion 통합 | `HTCMABLF_Fusion` (attn_module: static/dar/hgar) | hgar 분기 추가 ✓ |

데이터셋 어댑터:
- MOSEI/IEMOCAP: `train_stripped.py` (EXP_CFG M0~M5), `DLF_HTCMA_stripped.py` (+robustness용 `mask_text`)
- K-EmoCon: `/home/ajy/02_multimodal_emotion/03_visionmer_v3_binary/model_mm_v3_htdar.py` (V3 인코더+HT-DAR fusion swap), `train_binary_all.py --htdar --attn_module {static|dar|hgar}`

변형 매핑 (ablation 축):
| EXP | 라우터 | 의미 |
|---|---|---|
| M0 | none(meanpool) | 바닥 |
| M1 | static (고정 language/text anchor) | 고정앵커 baseline |
| M2/M3/M4 | DAR (수정된 동적) | 동적앵커 |
| **M5** | **HGAR (그래프-계층)** | ★ 제안 |

---

## 3. 라우터 진화 (기여의 핵심) — 무엇을 왜 어떻게
### 3a. 원본 = 그냥 gating (novelty 약)
modality별 독립 점수 → softmax. level별 독립. → "dynamic gating" 수준.

### 3b. exp_002 수정 (먼저 작동부터 시키기)
원본 DAR는 **구조적으로 죽어있었음**:
- Holo 항이 softmax에서 상쇄(dead) / c_mono 스펙불일치 / grad vanishing(1e-7, 균일동결) / per_modality=LayerNorm no-op.
- 수정: 모달별 −KL(p_m‖p̄), max-softmax c_mono, **cross-modal 정규화+learnable temp**, mixed-broadcast.
- 검증: 실 K-EmoCon mono_mlp grad **0→81.9** (라우터 작동).

### 3c. exp_004 = HGAR (novelty 올리기) ← 제안의 정체
- 노드 = (modality × level) 9개. 엣지 = cross-modal(같은 level) + cross-level(같은 modality 인접).
- relational GNN(2층) message passing → 노드가 "다른 modality + 다른 시간해상도"까지 보고 라우팅.
- 라우팅 = GNN-refined 노드 위 Co-Belief(수정본 그대로) → r_{m,l}.
- H_mix^l = Σ_m r_{m,l}·H_m^l. (수식·코드변경: exp_004 문서 §2-3)
- **단, §5 repo 서베이로 기존 graph-fusion 코드 있으면 그걸 기반으로 교체/검증** (from-scratch 지양).

---

## 4. 실험 설계 (어떻게 돌리나)
### E1. clean 정확도 (정직 baseline)
- MOSEI/IEMOCAP: M1(static) vs M3(dar) vs M5(hgar) ×3-5seed. 예상 tie(within-noise) → 정직 보고.
- 커맨드: `python train_stripped.py {M1|M3|M5} specific {seed} mosei`

### E2. ★ Robustness (메인 win)
- inference 때 `mask_text=True` (language 결손 시뮬, stripped forward에 구현됨).
- text-drop {0,50,100%} → acc_2 degradation 곡선. 가설: static 붕괴 > dar > **hgar 가장 robust**.
- standalone eval 스크립트 필요 (ckpt 로드 + 마스킹 test). [TODO]

### E3. K-EmoCon (지배 modality 없는 무대)
- `run_htdar_kemocon.sh` (static vs dar vs hgar, 6-fold, arousal/valence, CCC). crash 버그 수정됨.

### E4. Ablation (HGAR 진짜 일하나)
- E_level 제거 / E_modal 제거 / GNN 1층 → 엣지·깊이 기여 분리. 안 하면 "장식 GNN" 공격.

통계: gate_c_runner.py 재사용 (paired t + Wilcoxon + bootstrap CI + TOST).

---

## 5. 무엇을 fork하나 (서베이 확정) — from-scratch 금지 / 상세: REPO_SURVEY.md
| 컴포넌트 | fork 대상 | 라이선스 | 비고 |
|---|---|---|---|
| backbone | **DLF** (pwang322/DLF, AAAI'25) | MIT | 이미 보유. shared/specific 브랜치에 모듈 삽입. ALMT fallback |
| 그래프 레이어 | **COGMEN**(PyG RGCNConv/TransformerConv) | ⚠️GPL-3.0 | 카피레프트 → 막히면 **MM-DFN**(MIT) |
| 동적 라우팅 참조 | **KuDA** (MKMaS-GUET/KuDA, EMNLP'24) | MIT | per-sample 라우팅이 핵심인 유일 repo. gate 차용 |
| 결손 robustness | **CIDer**(프로토콜 5모드+OOD) + **MMIN**(6-condition baseline) | MIT | E2 평가 표준 — **직접 프로토콜 금지** |
| K-EmoCon/계층 | K-EmoCon supp(프로토콜) + **MHyEEG**(계층융합) | MIT/미상 | NPZ 어댑터는 우리가 작성 |

**핵심 verdict**: 단일 graph-routing repo 없음 → **COGMEN(그래프)+KuDA(라우팅) 결합 = HGAR의 ~80%**, 우리는
"graph-construction-over-routing glue" + bio NPZ 어댑터 + K-EmoCon 인코더만 작성.
**블로커(정직)**: 모든 fork가 gated pre-extracted feature .pkl(CMU-MultimodalSDK 등) 의존 →
**baseline 숫자 먼저 재현하고 나서** 우리 모듈 delta 측정해야 credible.

---

## 6. 구현 순서 (체크리스트)
1. [x] DAR 라우터 수정 (exp_002)
2. [x] K-EmoCon 통합 + crash fix
3. [x] HGAR 1차 구현 + 설계문서
4. [ ] repo 서베이 → fork 대상 확정 (서베이 진행 중)
5. [ ] HGAR 스모크 (forward/grad) — 기존 repo 기반이면 그쪽으로
6. [ ] E1 clean 3-5seed (MOSEI/IEMOCAP)
7. [ ] E2 robustness eval 스크립트 + 곡선 ← **win 여부 판가름**
8. [ ] E3 K-EmoCon static/dar/hgar
9. [ ] E4 ablation
10. [ ] 논문 작성 (CONFERENCE_PAPER_PACKAGE 골격)

## 7. 솔직한 확률
- 정식 컨퍼런스(ICASSP/ACII/workshop): ~55-65%. ICASSP/ACII full: ~35-45%.
- 최대 레버: E2 robustness gap이 크게 나오는가 + HGAR ablation이 엣지 기여 입증하는가.
- novelty가 천장 — 그래서 그래프-계층 라우터로 한 칸 올리는 것.
