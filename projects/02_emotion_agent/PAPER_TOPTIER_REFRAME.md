# Top-tier Reframe — 02_emotion_agent → AAAI 2027

> 2026-06-11. 목적: 흩어진 자산 중 **최고 발견 하나로 좁혀** 상위 컨퍼런스 통과 확률 최대화.
> 판단 요지: 더하기(cross-project 병합)가 아니라 **빼기**(thesis 단일화)가 정답.

---

## 0. 한 줄 thesis (NEW)

> **In-context learning은 modality-gated다 — LLM에게 입력 형식이 *낯설 때만* 작동하고,
> 그 이득은 hidden representation이 *압축*되는지로 인과적으로 결정된다.**

감정/문화(Ekman이 한국 얼굴에서 random)는 **headline이 아니라 testbed**.
일반화되는 ICL 명제 + 메커니즘이 상위 venue가 사는 물건.

## 1. 왜 좁히나 (as-is 진단)

현재 abstract는 "LoRA만 믿을만하고 ICL은 fragile" — **deflationary**라 top main track엔 약하다.
실제 강한 건 그 반대 방향의 *positive* 발견:

- **모달리티 게이팅**: AU-intensity text(LLM에 낯섦) ICL +12.75pp vs 영어 대화(익숙함) ≤1pp, k-sweep flat.
- **인과 메커니즘 (이게 핵심 자산)**:
  - Korean FER: ICL이 within-class L2 거리 **−22%** 압축 → 출력 분산 5× 감소 → ICL 이득.
  - IEMOCAP: ICL이 within-class 거리 **+12.7% 확장** (tightness 0.601→0.571) → ICL 무효.
  - = "압축되면 ICL이 듣고, 안 되면 안 듣는다"는 **cross-domain causal-chain 확인**.
  - attention entropy는 5개 layer 전부 동일 → 메커니즘은 attention이 아니라 **표현 압축**이라고 배제 증거까지 있음.

이 한 쌍(게이팅 현상 + 표현 압축 메커니즘 + 도메인 reversal)이 top-tier 골격.

## 2. 차용 = 최고만 남기고 등급 매기기

| 발견 | 처리 | 이유 |
|---|---|---|
| **modality-gated ICL** | **CORE / headline** | 일반화됨, 감정 너머 적용 |
| **표현 압축 메커니즘 + IEMOCAP reversal** | **CORE / 메인 figure** | 인과 확인 = 리뷰어 신뢰 |
| attention-entropy 배제 (5 layer) | CORE 보조 | 메커니즘 위치 특정 |
| Ekman FACS = random on Korean | 강한 motivation | Jack 2012 LLM-era 확인, testbed 정당화 |
| LoRA universal (3 dataset) | 보조 결과 | "ICL≠만능, LoRA가 안정" 대비축 |
| 3-axis anchor variance decomp | **부록/demote** | 흥미롭지만 niche·thesis 분산 |
| ICL⊂LoRA 대체성 | 보조 | discussion 한 문단 |
| HT-DAR fixed-anchor negative | **이 페이퍼엔 넣지 않음** | 다른 thesis. 섞으면 흐려짐 |
| AURF ambiguity-ceiling | **이 페이퍼엔 넣지 않음** | 다른 thesis |

→ HT-DAR/AURF는 별도 트랙(워크샵/short/저널)으로 살리되, 이 main 페이퍼엔 미포함.

## 3. 상위 전환을 가르는 단 하나의 실험 (critical path)

**문제**: 단일 모델(Qwen2.5-7B 4-bit). AAAI/ACL급에서 empirical 명제의 **#1 desk-reject 사유**.
"Llama, 더 큰 Qwen에서도 되나?" 한 줄에 무너짐.

**전환 실험 (proposed, 미실행)**:
- modality-gating(zero-shot vs ICL k=4, 3 dataset)을 **2~3개 모델 패밀리**에서 재현:
  Llama-3-8B-Instruct, Qwen2.5-14B 또는 32B(4-bit), (가능하면) Mistral-7B.
- 표현 압축 메커니즘(within-class L2)도 같은 모델들에서 측정 → 인과 주장 일반화.
- A6000 48GB: 7~14B 4-bit 다수, 32B 4-bit ~20GB 가능. 6주 내 충분.

이거 하나가 확률을 mid→top으로 올린다. **JY 승인 시에만 launch.**

## 4. Venue + 통과 확률 (정직)

| Venue | 마감 | as-is | reframe+멀티모델 | 적합도 |
|---|---|---|---|---|
| **AAAI 2027** | abs 7/20, full **7/27** | ~10-15% | **~30-40%** | 가장 가까운 top. 단일모델이면 위험 |
| EMNLP 2026 | ARR 5/25 **지남** | — | — | 다음 ARR 사이클로 백업 |
| ACL 2027 / ARR | ~여름 사이클 | — | ~40-50% | ICL+메커니즘에 가장 자연스러운 집 |
| ICLR 2027 | ~9월말 | low | ~30% | runway 길다. AAAI 떨어지면 재타깃 |

**권장**: 1순위 **AAAI 2027 (7/27)** 스프린트 — 단, 멀티모델 실험 커밋 전제.
멀티모델 못 돌리면 AAAI 강행 대신 **ACL ARR 여름 사이클**로 (runway 확보, ICL paper 본진).

## 5. 지금 액션 (짧게)

1. abstract/§1을 **modality-gated ICL + 메커니즘** 헤드라인으로 재작성 (writing, 즉시 가능).
2. 메인 figure = 도메인별 (ICL gain × 표현 압축Δ) 산점도/이중축 — 인과 한 장.
3. **JY 결정**: 멀티모델 전환 실험 GO? (AAAI 1순위로 가려면 필수)
4. anchor-variance 3-axis는 부록으로 강등, §4를 2 core finding 중심으로 슬림화.

> 결정 필요한 단 하나: **멀티모델 실험을 돌릴지** (= AAAI 7/27 현실성의 분기점).
