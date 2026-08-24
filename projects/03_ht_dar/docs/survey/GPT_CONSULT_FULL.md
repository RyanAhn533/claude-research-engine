# GPT 상담용 — HT-DAR 전체 상황 (그대로 붙여넣기)

GPT에게: 멀티모달 감정인식 컨퍼런스 논문 상황 전체다. 끝의 질문들에 정직하게(양비론·립서비스 금지) 답해줘.

---

## 0. 나/목표
- 석사생. 진로 = 산업/대기업/창업(42dot 류), 연구사이언티스트 아님. → top-tier 집착 안 함, **좋은 컨퍼런스 1편**이 목표.
- 타겟: **ICASSP 2027 (제출 ~2026년 9월, 약 3개월 남음)**. fallback ACII / Q1 저널(TAFFC, Information Fusion).
- 시간: 3개월. GPU = RTX A6000 1장(공유).

## 1. 모델 (HT-DAR)
멀티모달 감정/감성 융합 모듈. 핵심 아이디어 = **앵커(cross-modal attention의 query) modality를 고정하지 말고 sample마다 학습.**
- 기존: MSA는 language를 앵커로 고정(DLF AAAI'25, ALMT), bio-centric은 physiology 고정.
- 우리: 라우터가 "이 샘플은 어느 modality를 앵커로?"를 결정.
- 구조: Hierarchical temporal pyramid(fine/med/coarse) + per-level cross-attention + **라우터** + Beta gate + head.
- 라우터 진화 3단:
  1. 원본 DAR: Co-Belief(per-modality confidence + cross-modal agreement) softmax. (코드상 버그로 죽어있던 것 수정함: holo 항 상쇄/grad vanishing/per_modality no-op → cross-modal 정규화+learnable temp+mixed-broadcast로 수정, 실데이터 grad 0→81 확인.)
  2. HGAR(제안): 라우팅을 **(modality×level) 9노드 그래프** 위 GNN(2층, cross-modal+cross-level 엣지)으로 계산. 검증: forward/grad 정상, r 학습됨.

## 2. 결과 (지금까지)
### MOSEI clean 정확도 (text 포함, 풀 모달리티)
```
M0 0.850 / M1(static 고정앵커) 0.861 / M3(dar 동적) 0.843 / M5(hgar 그래프) 0.842
DLF(SOTA 재현) 0.854 / DLF 논문 0.864
```
→ **전부 tie 밴드. 우리가 clean에서 baseline 못 이김** (괜찮지만 SOTA 아님).

### ★ MOSEI text-drop robustness (1-seed, 메인 결과)
| 모델 | clean(0%) | 25% | 50% | 75% | 100% drop | 붕괴폭 |
|---|---|---|---|---|---|---|
| static(고정 language-anchor) | 0.861 | 0.741 | 0.612 | 0.492 | **0.372** | −48.9pp |
| dar(동적) | 0.843 | 0.775 | 0.710 | 0.641 | 0.576 | −26.8pp |
| **hgar(그래프)** | 0.842 | 0.786 | 0.739 | 0.687 | **0.629** | **−21.4pp** |
→ text 빠질수록 고정앵커 붕괴(랜덤 이하), **그래프 라우터가 전 구간 최고, 단조.** 100% drop서 static 대비 +25.7pp.
→ **단 1-seed. 지금 3-seed 재현 중(거의 끝).**

## 3. 데이터셋 × 모달리티 현황
| 데이터셋 | 모달리티 | text | 상태 |
|---|---|---|---|
| MOSEI | text+audio+video | ✓지배 | 메인. clean tie + robustness 1-seed(3-seed 진행) |
| IEMOCAP | text+audio+video | ✓ | clean tie(wF1 0.58~0.59), robustness 미실행 |
| MOSI | text+audio+video | ✓ | config 문제로 보류 |
| K-EmoCon | video+audio+bio | ✗ | HT-DAR 통합만, 미학습 (별도 CBBF 저널 영역) |
| DEAP | bio only | ✗ | 이 머신에 없음 (CBBF 저널) |

(참고: K-EmoCon/DEAP의 bio-centric fusion은 **별도 KBS 저널 논문**으로 진행 중 — 이 컨퍼런스와 안 겹치게 분리.)

## 4. ★ novelty 검증 결과 (적대적 조사함 — 중요)
**헤드라인 주장들이 이미 점유됨:**
- "앵커를 sample마다 학습" → **KuDA(EMNLP'24), MODS(2025, Primary Modality Selector+cross-attention)**, RollingQ(2025). 거의 정중앙.
- "동적 라우팅 = 결손 robustness" → **Senti-iFusion(2025)**, **LNLN(NeurIPS'24, language=dominant 결손)**.
- Co-Belief(confidence+agreement) → MA-AFS, Conf-SMoE 점유.
- modality×level 계층 그래프 토폴로지 → HFGCN, MSGFN 점유.

**살아남은 유일한 슬라이스**: "**그래프가 앵커 가중치를 *계산*한다**"(graph 논문은 feature fuse만 하지 anchor weight emit 안 함; routing 논문은 graph 안 씀) + "**agreement를 살아남은 노드 위에서 구조적으로 재정규화**(결손에 graceful)".

**검증 결론**: 지금 프레이밍 그대로면 KuDA/Senti-iFusion으로 desk-reject. 살리려면 (A) 표준 결손 프로토콜에서 KuDA/Senti-iFusion/LNLN과 **head-to-head 승/무**, (B) **그래프 라우터 > MLP 라우터(같은 feature)** ablation(존재론적), (C) agreement-under-absence를 진짜 메커니즘으로.

## 5. 경쟁자 (결손 robustness 분야, 이미 강함)
LNLN(NeurIPS'24), MissModal(TACL), Proxy-Driven(ACL'25), CIDer(2025, RMFM task), CM-ARR(2024), TF-Mamba(2025), TMDC(2025), MMIN(ACL'21, 표준 baseline).
→ 무대(text 결손 robustness)는 ICASSP에 핫함(MEIJU 2025 Grand Challenge 존재) = 낼 만함 ✓, 근데 crowded.

## 6. venue 레벨
- ICASSP ≈ Q1 저널 IF~6급(Neurocomputing 정도). good, not elite. 석사엔 충분히 좋음. 대기업 취업엔 ICASSP면 됨(AAAI 불필요).
- 더 위: TAFFC, Information Fusion(IF18), TPAMI — ICASSP보다 명백히 위, 근데 우리 work론 무리/시간↑.

## 7. 솔직한 약점 + 확률
- novelty incremental + 분야 crowded + clean은 tie(못 이김).
- robustness 결과는 큼(1-seed) but 아직 우리 자체 baseline만 이김 — 표준 SOTA랑 안 붙어봄.
- 확률(내 추정): reframe + ablation B + head-to-head 다 하면 ICASSP ~50-65%. 그냥 내면 novelty reject.

---

## ★ GPT에게 묻는 것
1. novelty가 KuDA/MODS/Senti-iFusion에 거의 겹치는데, "graph-computes-anchor + agreement-under-absence"로 좁히면 **ICASSP 통과 가능한가, 아니면 이미 죽었나?** 솔직히.
2. **clean tie + robustness win** 구조가 ICASSP에 충분한가? (clean에서 SOTA 못 이기는 게 치명적인가?)
3. 우리가 **반드시 이겨야 할 baseline 1~2개**는? (MMIN/MissModal/LNLN/CIDer 중) 그리고 LNLN(NeurIPS) 같은 강자를 ICASSP 논문이 꼭 이겨야 하나, 아니면 표준 baseline만 이기면 되나?
4. **그래프 라우터(9노드)** 가 정당한가, MLP 라우터로 충분하고 GNN은 over-engineering인가? 정당화하려면 노드/엣지를 어떻게?
5. 평가를 **text-drop만** 할까, **모든 modality(audio/video도) drop**까지 표준 프로토콜로 해야 하나?
6. **3개월 석사 스코프**에서 위 중 **버릴 것 / 집중할 것** 하나씩?
7. **ICASSP vs TAFFC/Information Fusion 저널** — 우리 work과 진로(산업) 고려 시 어디를 노려야?
8. 우리가 못 본 **더 나은 프레이밍/무대**가 있나? (효율, calibration, cross-corpus, fairness 등)
