# NO-GO 판정 — HT-DAR robustness (2026-06-05, 자율 세션)

## 한 줄
**공정한 비교(modality-dropout 학습)에서 동적/그래프 앵커 라우팅은 단순 고정앵커 baseline을 못 이긴다. HT-DAR 컨퍼런스 thesis는 죽었다.**

## 실험 경로 (정직한 전말)
1. **1-seed robustness** (non-dropout 학습): static 0.86→0.37 붕괴, hgar 0.63 → +25pp. 대박처럼 보임.
2. **3-seed** (non-dropout): static 붕괴가 **seed 1111 운빨**로 드러남. 평균 gap +1~3pp, within-noise. hgar < dar (그래프 무의미).
3. **그래프 코드 결함 발견·수정**: HGAR v1이 GNN으로 feature를 over-smooth해서 라우팅 신호 뭉갬 (top-tier repo 교차검증으로 확인). → HGAR v2 (예측-위 availability-aware agreement, over-smoothing 제거)로 재작성 + 교차검증(NaN 버그까지 잡음).
4. **make-or-break: modality-dropout 학습** (표준 결손 프로토콜, MDROP=0.5), seed 1111:

| 모델 | clean | 25% | 50% | 75% | 100% drop |
|---|---|---|---|---|---|
| static(고정) | 0.856 | 0.797 | 0.754 | 0.705 | **0.647** |
| dar(동적) | 0.836 | 0.784 | 0.735 | 0.693 | 0.635 |
| hgar(그래프v2) | 0.859 | 0.797 | 0.745 | 0.697 | 0.636 |

→ **static이 100% drop에서 최고(0.647). dar −1.2pp, hgar −1.1pp. 그래프=dar (이득 0).**

## 결론
- static을 결손 학습시키면 그 자체로 robust → **동적/그래프 라우팅이 추가하는 가치 = 없음(오히려 약간 손해).**
- novelty 조사(KuDA/MODS/Senti-iFusion 점유) + 이 음성 결과 = **이 각(anchor routing for robustness)은 컨퍼런스 안 됨.**
- 1-seed 환상은 (a) 운빨 seed (b) baseline 불공정(결손 학습 안 함) 합작이었음. 공정 비교가 죽임.

## 무엇이 살아있나 / pivot 옵션
1. **정직한 negative/analysis 페이퍼**: "표준 결손 학습 하에서 dynamic/graph anchoring은 fixed anchor 대비 이득 없음" — 워크샵/숏페이퍼급. 낮은 임팩트, 근데 정직.
2. **CBBF 저널(K-EmoCon/DEAP)로 에너지 집중** — JY의 더 성숙한 work(KBS, 실제 결과 있음). 컨퍼런스 HT-DAR는 접고 저널 완성.
3. **완전히 다른 아이디어** — HT-DAR 라인 자체가 clean tie + robustness no-benefit이라 더 짤 axis가 안 보임.

## 권장 (자율 판단)
- HT-DAR 컨퍼런스는 **현 형태로 불가.** 헛돌리기 금지.
- 가장 합리적 = **옵션 2 (CBBF 저널 집중)** 또는 옵션 1(정직한 negative 워크샵).
- JY 결정 필요. 3-seed 확장은 안 함 (명확한 음성 + 자원 낭비).

## 자산 (재사용 가능)
- 검증된 코드: HGAR v2 + modality-dropout + robustness eval 파이프라인 (다른 데이터/아이디어에 재사용 가능).
- 교차검증 방법론: 1-seed→3-seed fluke 검출, 그래프 placement 진단, NaN 검출 — 다음 프로젝트에 그대로.
