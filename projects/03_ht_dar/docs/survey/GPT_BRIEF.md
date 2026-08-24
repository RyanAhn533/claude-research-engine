# GPT 자문용 브리프 — HT-DAR 컨퍼런스 (그대로 붙여넣기용)

GPT에게: 아래는 내 멀티모달 감정인식 컨퍼런스 논문 상황이다. 끝의 질문에 답해줘. 정직하게, 양비론 말고.

## 상황
- 나(석사)는 멀티모달 감정인식 융합 모듈 **HT-DAR**를 만들었다. 핵심 = **앵커 modality를 고정하지 말고 sample마다 학습**한다.
  - 기존 SOTA는 앵커를 고정: MSA(MOSEI/MOSI)는 **language**를 query 앵커로(DLF AAAI'25, ALMT), bio-centric은 **physiology**를 앵커로 박는다.
  - HT-DAR = Hierarchical temporal pyramid + Dynamic Anchor Routing(Co-Belief: mono-confidence + holo-agreement) + Beta gate.
- **결과(검증됨)**: MOSEI/IEMOCAP에서 HT-DAR는 baseline과 **통계적 tie (5-seed, within-noise)**. 이유 = 이 벤치들은 **text가 지배적**이라 라우터가 항상 text를 고름 → 동적 앵커가 결정할 게 없음. (절대성능은 SOTA band 0.85)
- 라우터가 원래 코드상 버그로 죽어있었는데(grad vanishing 등) 고쳐서 이제 실데이터에서 학습됨(grad 0→81.9). 그래도 clean 정확도는 여전히 tie.

## 내가 정한 방향
1. **win 무대 = modality 결손 robustness**: 고정 language-anchor는 text가 빠지면 붕괴, 동적 앵커는 audio/video로 갈아타 버틴다. → MOSEI/IEMOCAP(표준데이터)에서 text-drop degradation 곡선으로 "동적>고정" 입증. clean tie는 정직 보고.
2. **novelty 올리기 = 그래프-계층 라우터**: 라우터를 (modality×level) 그래프 GNN으로 (단순 gating→structured routing).
3. **fork 전략(from-scratch 금지)**: backbone=DLF, 라우팅참조=KuDA(EMNLP'24), 그래프레이어=COGMEN/MM-DFN, 결손프로토콜=CIDer+MMIN, K-EmoCon=MHyEEG.
4. 타겟 = ICASSP 2027(~9월) 또는 ACII. mid-tier. (별개로 bio-centric은 KBS 저널로 따로 감 — 겹치지 않게.)

## 솔직한 약점
- novelty가 incremental (dynamic modality weighting은 MoE/gated-fusion에 이미 있음).
- 그래프 노드 9개(3 modality×3 level)뿐이라 GNN 이득 modest.
- 결손 robustness 논문도 이미 많음(MMIN, CIDer, GCNet).
- 내 추정 합격확률: 정식 컨퍼런스 ~55-65%, ICASSP/ACII full ~35-45%.

## ★ GPT에게 묻는 것
1. "고정앵커 → 학습앵커 + 결손 robustness"가 ICASSP/ACII mid-tier 기여로 **충분히 차별적인가, 아니면 MMIN/CIDer 류와 너무 겹쳐서 reject 각인가?** 차별화하려면 뭘 더해야 하나?
2. **그래프-계층 라우터**가 9노드에서 의미 있나, 아니면 단순 attention 라우터로 충분하고 GNN은 over-engineering인가? 그래프가 정당화되려면 노드 구성을 어떻게 바꿔야 하나(예: modality×time, 채널 단위)?
3. clean tie + robustness win 조합에서, 리뷰어를 설득할 **단 하나의 결정적 figure/실험**은 뭐여야 하나?
4. 더 나은 무대/프레이밍이 있나? (예: 효율, calibration, cross-corpus generalization 등) robustness보다 나은 win 각이 있나?
5. 3개월 석사 스코프에서, 위 계획 중 **버릴 것 / 집중할 것** 하나씩 골라준다면?

## 참고 (원하면)
- 전체 구현계획: MASTER_IMPLEMENTATION.md / repo 상세: REPO_SURVEY.md / 라우터 버그수정: exp_002 / 그래프라우터 설계: exp_004.
