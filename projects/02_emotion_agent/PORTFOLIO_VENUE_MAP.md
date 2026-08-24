# Portfolio → Conference Feasibility Map

> 2026-06-11. JY 전체 디렉토리(/home/ajy 01~06 카테고리) 스캔 후.
> 목표: **올해 마감 남은 상위 컨퍼런스 중, 될만한 것 중 가장 높은 곳**으로 진행.

---

## A. 출판 가능 자산 인벤토리 (전체)

| 자산 | 위치 | 핵심 결과 | 완성도 | 본질 |
|---|---|---|---|---|
| **modality-gated ICL** | engine/02_emotion_agent | ICL은 입력이 LLM에 낯설 때만 작동 + 표현압축 인과 | 95% draft | **ML/NLP conf** |
| AU-RegionFormer | 01/07, AU-RegionFormer/ | Mouth>Eyes 13%p, 연세대 298명 consensus, 413K | SGMT 제출(4/13), Q2 후배 인계 | **저널**(TAFFC/ESWA) |
| CBBF | 03/01 | causal bio→behavior, cccA=0.194 | 논문 작성중 | 저널/ACII |
| Jetson_thor DMS | 04/04 | 98% acc, 30fps, edge, 정부R&D, 8-byte | 상용화 직전 | 시스템/IEEE IV·ITSC |
| VisionMER V3/MoE | 02/03~05 | K-EmoCon CCC≈0=ceiling, bio 0.6% gate | 완료 | 저널/워크샵 |
| CARLA_agent | CARLA_agent/ | risk-adaptive driver-state uncertainty | 미완(Bench2Drive 보류) | 미성숙 |
| claude-research-engine | 06/04 | agentic research automation + RLHF | 활성 framework | 워크샵/position |
| brandspace/POPR | 05/01~02 | 저작권+특허, "AI는 증강만" 관찰 | IP 등록 | HCI/제품 |

→ **컨퍼런스 최상위 ceiling을 가진 건 modality-gated ICL 하나.** 나머지는 저널·응용·시스템 트랙.

## B. 컨퍼런스별 가능성 (modality-gated ICL 기준)

| Conf | Tier | 마감(올해) | 적합도 | as-is 확률 | +멀티모델 | 비고 |
|---|---|---|---|---|---|---|
| **AAAI 2027** | top general AI | full **7/27/2026** | 높음 | ~12% | **~30-40%** | 가장 가까운 top. 단일모델=desk-reject 위험 |
| **ICLR 2027** | top ML | ~**9월말** | 중(method 선호) | ~15% | ~30% | runway 길다. 메커니즘 angle이 ICLR엔 +. AAAI 백업 |
| ACL 2027 (ARR) | top NLP | ARR 여름~가을 사이클 | **최적**(ICL 본진) | — | main ~35% / Findings ~55% | 타임라인 늦지만 가장 자연스러운 집 |
| EMNLP 2026 | top NLP | ARR 5/25 **지남** | 최적 | — | — | 이번 사이클 종료 |
| NeurIPS 2026 | top ML | ~5월 **지남** | 중 | — | — | 종료 |
| CVPR 2027 | top vision | ~11월 | 낮음(NLP paper라) | — | — | ICL paper엔 부적합 |

**다른 자산의 컨퍼런스 ceiling (참고)**:
- AU-RegionFormer: ACM MM 2027(~4월) 가능하나 저널(TAFFC) 우위 — 컨퍼런스론 중상위.
- Jetson_thor: IEEE IV 2027 / ITSC — 좋은 응용 conf지만 "최상위 ML"은 아님. 98% 수치는 리뷰어가 의심(소규모 test 의혹).
- claude-research-engine: NeurIPS/ICLR **워크샵**, 또는 agentic-science position. main track은 엔진이 더 좋은 논문을 만든다는 실증 필요.

## C. 결론 — 진행 방향

> **될만한 것 중 최상위 = AAAI 2027 (7/27), vehicle = modality-gated ICL, 전제 = 멀티모델 검증.**
> 멀티모델 못 돌리면 → **ICLR 2027(9월)** 로 내려 runway 확보(같은 top tier, 시간 벌기).
> ACL ARR은 본진이라 병행 가능하나 타임라인이 가장 늦음.

**전환 레버(불변)**: Llama-3-8B + Qwen-14/32B에서 게이팅+압축 재현 = 단일모델 desk-reject 차단.
이게 AAAI를 "12% → 30-40%"로 만드는 단 하나의 실험. A6000 6주 내 가능.

**지금 진행(GPU 불필요)**: abstract/§1을 modality-gated ICL 헤드라인으로 재작성, 인과 figure 1장, §4 슬림화.
**GPU 게이트(JY GO 필요)**: 멀티모델 런.
