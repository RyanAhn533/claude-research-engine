# 통합 전략: CARLA + 감정/생체 → top-tier 검토 (토론 + GPT 프롬프트)

> 2026-05-28 · 질문: "지금까지 한 모든 연구(affective grounding / S-PACE 멀티모달 / emotion_agent LoRA / BioToken / CARLA_agent)를 엮어 top-tier 하나로 갈 수 있나?"
> ⚠️ 이 문서는 메모리의 "CARLA_agent ↔ 감정인식 혼용 금지(2026-05-11)" 경계를 JY가 의도적으로 뒤집는 결정에 따른 것.

---

## 0. 코드로 확인한 현실 (계획의 토대 — 추측 아님)

**CARLA_agent 실제 상태:**
- 외부 인지(카메라/라이다/depth/semseg)만 wired + 컷인 시나리오 + 사전학습 주행에이전트 6종 비교(TFv6/InterFuser/NEAT 등) 됨.
- **차내 캐빈캠 ❌ / ECG·PPG·GSR ❌ / 음성 센서 ❌ — 코드에 0개.** "cockpit/audio"는 KMER 자극영상용 PIL·ffmpeg 오버레이(deprecated)였지 in-sim 센서 아님.
- 운전자 상태 = `D_drowsy ∈ {0, 0.7}` 스칼라 mock 하나 → 브레이크 지연만. RL 코어 미착수(Stage 0-1).
- CARLA_agent 자체 규칙(Q1_WORKING)이 지금 "emotion 라벨 금지 + D를 reward에 넣지 마"라고 박혀있음 → 합치려면 프레임 재작성 필요.

**진짜 자산 (경로 정정: `/home/jy` 아님, 다 `/home/ajy/` 밑):**
| 자산 | 상태 |
|---|---|
| **PPB-Emo** 데이터셋 (운전자 face+EEG+운전행동+감정 V/A/D+7class) | 실데이터, 240 seg(작음) — 운전자 감정·생체·차량행동이 한 데 묶인 유일 |
| **VisionMER V3** (Bio/Vid/AST-Audio 인코더 + BioCond + gated fusion) | 학습됨, 재사용 가능 |
| **CBBF** causal 컨디셔너 (생리→행동 선행) | prototype |
| **emotion_agent** | LoRA 55%>ICL>prompt; **bio→텍스트→LLM 실패(16.7%)** → 생체는 학습 인코더로 |
| **affective_grounding** (이번 세션) | self/observer 불일치, VLM이 표정≠감정 미분리 + flat confidence |

---

## 1. 토론 (CLAUDE.md 프레임워크: 3 페르소나 + 본체 총괄)

페르소나: ① Visionary(통합추진) ② Pragmatist(실행회의) ③ Reviewer(탑티어 심사자)

### 라운드 1 — 초기 입장
- **① Visionary**: 유일 moat — PPB-Emo + 학습된 멀티모달 인코더 + 돌아가는 CARLA + affective-uncertainty 원리. "보정된 감정 불확실성→안전 행동"은 안 막힌 각도. embodied×affective×safety = top-tier 그릇. 크게.
- **② Pragmatist**: CARLA에 생체/캐빈/음성 0, RL 미착수, PPB-Emo 240 seg, 토대(affective_grounding)도 흔들림(단일평가자·선점). 5개 프로젝트 용접 = 6개월+ 배관, 통합 리스크 큼. 조각 따로 출시.
- **③ Reviewer**: top-tier는 "합쳤다"에 점수 안 줌. 이빨 있는 claim 하나 = *"감정 추정을 언제 불신할지 아는 agent가 모르는 agent보다 안전하게 운전한다."* 나머지(fusion·인코더·CARLA)는 engineering. 그 claim을 실주행 데이터로 못 보이면 reject.

### 라운드 2 — 교차 반박
- **②→①**: PPB-Emo 240 seg로 신뢰할 불확실성 추정기 못 만들면 moat 무의미. CARLA 생체 없음 = "멀티모달 시뮬"은 허구, 결국 스칼라 주입.
- **①→②**: in-sim 센서 불필요. PPB-Emo 오프라인 학습 추정기의 **보정된 출력을 CARLA에 주입** = 정당한 coupling. 작은 데이터 OK(claim이 정확도 아니라 불확실성 행동).
- **③→둘다**: 그 오프라인→CARLA 주입이 정확히 reviewer 공격 지점("합성 주입 의존"). 방어: (a) 주입 불확실성이 실신호품질과 상관함을 보이거나, (b) claim을 *"불확실성이 주어졌을 때 게이팅 행동이 더 안전"* 으로 좁혀라(end-to-end 센싱 주장 안 함).

### 라운드 3 — 수정 입장 + 합의
- 방어 가능한 top-tier claim **1개**: *"affective 운전자상태 추정에서 불확실성-게이팅 행동이 점추정 행동보다 안전하다."*
- 방법: (a) PPB-Emo로 추정기+**보정된 불확실성** 학습 → (b) CARLA closed-loop에서 agent가 추정치+불확실성 받아 개입 게이팅 → (c) driver-state shift 하 안전지표(collision/jerk/false-alarm) 비교.
- VisionMER/CBBF/emotion_agent = backbone 재사용/인용, **헤드라인 금지**. affective_grounding 정적이미지 = **motivation 섹션으로 격하**. scope 정직(uncertainty-gating 결과지 end-to-end 센싱 아님).

### 🔭 총괄 리뷰 (본체 판단)
**내 판단**: **가라. 단 "통합"을 팔면 죽고, "claim 하나에 모든 자산을 종속"시켜야 산다.** 자산은 진짜지만 그대로 쌓으면 프랑켄슈타인. Reviewer의 단일 claim 압축이 정답.

**합의**: uncertainty-gated driving safety = 유일 top-tier claim. PPB-Emo+CARLA coupling. scope 정직.

**미해결 쟁점**:
- PPB-Emo 240 seg로 calibration(불확실성) 신뢰 가능?
- coupling faithfulness 방어법 (합성주입 비판)?
- 게이팅을 RL로? rule-based로? (RL 미착수 리스크)

**권장 액션 3가지**:
1. **Critical-path 1발 먼저** — PPB-Emo 추정기+불확실성 → CARLA {점추정} vs {불확실성-게이팅} → 안전지표. 이기면 thesis 산다.
2. **RL 말고 rule-based uncertainty gate로** 싸게 검증(RL은 그다음). 미착수 RL에 바로 들어가지 마.
3. **claim 1개로 압축** + affective_grounding=motivation, 멀티모달 인코더=machinery 격하.

**블라인드 스팟** (제일 큰 사각):
- PPB-Emo가 public이면 선점 위험 — 확인 필수.
- CARLA driver-state 주입이 "사람 운전자 모델"과 안 맞으면 결과 무의미(메모리 원칙).
- **"감정"이 운전 안전에 정말 인과적인가?** 실제론 drowsiness/distraction만 중요하고 감정(분노/슬픔)은 약할 수 있음 → 감정축 약하면 thesis 붕괴.

---

## 2. GPT에 던질 프롬프트 (복붙용)

```
나는 AI/ML 연구자(석사)이고, 가진 자산들을 묶어 top-tier(NeurIPS/ICLR/CVPR급 또는
T-IV/IROS) 논문 하나를 만들 수 있는지 냉정하게 평가받고 싶다. 칭찬 말고 약점 위주로.

[가진 자산 — 코드/데이터 실제 확인함]
1. PPB-Emo 데이터셋: 운전자 face + EEG(5밴드 생리) + 운전행동(accel/brake/steer/speed)
   + 감정라벨(valence/arousal/dominance + 7class). 단 240 세그먼트(작음).
2. 학습된 멀티모달 감정인식 백본(VisionMER V3): bio/video/audio(AST) 인코더 +
   causal bio-behavioral 조건화(CBBF) + gated fusion.
3. 돌아가는 CARLA 0.9.15 headless rig: 외부인지(카메라/라이다/depth/semseg) + 컷인
   시나리오 + 사전학습 주행에이전트 6종 비교. 단 차내 생체/캐빈캠/음성 센서는 0,
   운전자상태는 스칼라 mock 하나, RL 에이전트 미착수.
4. 선행 결과(정적 얼굴): 한국 자기보고 감정에서 본인-외부관찰자 라벨 불일치가
   부정감정에서 2.5~3배 크고, VLM(Qwen2.5-VL)은 "보이는 표정"과 "느낀 감정"을
   구분 못하며 모호성에도 confidence가 평평(uncalibrated)하더라.
5. LLM 감정적응 실험: LoRA가 prompt/ICL보다 강하고, 생체→텍스트→LLM 경로는 실패(랜덤수준).

[내가 생각하는 thesis]
"Affective-uncertainty-aware driver assistance": 운전자 상태를 멀티모달로 추정하되
그 추정의 '보정된 불확실성'이 안전 행동을 지배한다(믿을 만할 때만 개입, 애매하면
유보/확인). claim 한 줄 = "감정 추정을 언제 불신해야 하는지 아는 agent가 모르는
agent보다 안전하게 운전한다."

[구조 초안]
- PPB-Emo로 추정기+불확실성 학습 → CARLA closed-loop에서 agent가 추정치+불확실성을
  받아 개입을 게이팅 → driver-state shift 하 안전지표(collision/jerk/false-alarm) 비교.
- 정적-얼굴 결과는 motivation, 멀티모달 인코더는 machinery로 격하, claim 1개 집중.

[너에게 묻는다 — 구체적으로]
1. 이 한 줄 claim이 top-tier에서 살아남나? 어느 venue가 가장 맞나(또는 안 되나)?
2. 가장 치명적인 약점 3개와 각각을 죽일/살릴 결정적 실험.
3. 특히: 오프라인 추정기→CARLA 합성주입 의존을 reviewer가 어떻게 공격하고,
   어떻게 방어하나? claim을 어디까지 좁혀야 정직하면서도 임팩트가 남나?
4. "감정"이 운전 안전에 정말 인과적인가, 아니면 drowsiness/distraction만 중요하고
   감정축은 약한가? 이게 thesis를 무너뜨리나? 관련 선행연구로 판단해달라.
5. PPB-Emo 240 seg로 '불확실성 보정' claim이 통계적으로 가능한가? 안 되면 대안은?
6. 이미 누가 했나(자율주행 affect-aware/driver-state uncertainty/calibrated
   intervention)? 선점 위험과 차별점.
7. 솔직히: 이걸 통합 top-tier로 가는 게 맞나, 아니면 (a) 정적 affective-grounding
   분석논문(ACII/ICMI) + (b) 별도 driving-safety 논문으로 쪼개는 게 기대값이 높나?

각 답에 근거(가능하면 논문/사례)를 달고, 마지막에 "내가 너라면 다음 2주에 뭘 할지"
3가지만.
```

---

## 3. GPT 2차 의견 대조 + 최종 판정 (2026-05-28)

**✅ GPT ↔ 우리 토론 강한 일치**: closed-loop에 운전자 없음(=핵심 급소) / PPB-Emo 240seg calibration 강주장 위험(bootstrap CI 먼저) / claim=uncertainty-gating(end-to-end 감정인식 ❌) / emotion 헤드라인 약함.

**🔧 GPT가 우릴 교정 (둘 다 GPT가 맞음)**:
1. affective_grounding을 "motivation으로 격하"가 아니라 **독립 논문(Track A)으로 빼라** (단독 80점 > 통합 한방 45점). 우리 총괄이 저평가했음.
2. 통합 한방은 **지금 보류**, **2-track 순차 + 나중 journal 통합**(85점)이 기대값 최고. Visionary의 "지금 통합"은 기각.

**➕ GPT 추가**: 감정-안전 인과성(우리 최대 블라인드스팟)을 **DPV-MFD 2024**(분노가 control/위험판단 손상; 56명 EEG/EDA/차량)로 부분 방어. 단 "감정 단독" 아니라 **"driver-state degradation의 한 축"**.

**❌ GPT가 못 본 것 (우리 SELF_ATTACK 모름)**: Track A는 이미 선점 — EmotionHallucer(2025), "What You Feel Is Not What They See"(2026) + 단일평가자 confound. → Track A 살리려면 (a)한국 방향성 비대칭 (b)self-report 라벨 VLM audit(선행엔 self-report 없음=유일 wedge) 전면 + (c)단일평가자 정직 선언/multi-rater 해결.

### 최종 결정: 2-TRACK
| Track | 내용 | venue | 캐비엇 |
|---|---|---|---|
| **A** | affective grounding 단독 (self/observer 불일치 + VLM miscalibration) | ACII/ICMI/AAAI-WS | 선점+단일평가자 → 한국비대칭+self-report-VLM-audit wedge, rater 질문 |
| **B** | driver-state **uncertainty-gating** intervention (CARLA) | T-IV/IROS급 | emotion=degradation 한 축, exogenous trace 정직선언(human-in-loop 아님), 차별점=affect/bio기반 uncertainty gating |
| 통합 | journal/T-IV 확장 | — | A·B 둘 다 나온 **다음에** |

### 다음 2주
1. **Track A outline** (결과 있음, 최速 산출) — rater 3명 가능 여부부터(claim 강도 결정)
2. **Track B claim 문장 고정** — "human-in-the-loop" 금지 → "offline-estimated exogenous driver-state uncertainty trace"
3. **PPB-Emo calibration bootstrap CI** — ECE/NLL/Brier 평균 말고 CI (Track B 생존 즉결)

### Track B 차별점 (기존 UQ-driving과 안 겹치게)
"UQ를 썼다"가 아니라 **운전자 상태/감정/생체 기반 uncertainty를 intervention gating에 넣었다**. (선택지 3 = emotion 완전 제거 → 기존 EnDfuser류와 너무 가까움, 비추.)
