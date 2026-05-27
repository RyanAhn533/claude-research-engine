# Physical AI 시뮬레이션 환경 선정 (exp_008 동반 문서)

> MASTER_PLAN.md §10(Physical AI 확장)을 구체화한 문서.
> 목적: "감정 캐릭터를 두고, 우리가 설계한 Agent가 실제로 그 안에서 돌아가는 CARLA 같은 무료 환경" 선정.

---

## 6. 환경 후보 정리

### 1순위: Habitat 3.0 (가장 추천)
humanoid avatar, robot, human-in-the-loop, collaborative task를 지원하는 embodied AI simulator. home 환경에서 HRI / social navigation / social rearrangement 연구용 플랫폼.

- **장점**: humanoid avatar 있음 · robot-agent 있음 · human-in-the-loop 가능 · HRI/Physical AI 논문 포지션에 잘 맞음 · Meta/AI Habitat 생태계 인지도
- **단점**: 얼굴 표정/감정 캐릭터가 핵심 기능 아님 · 감정 데이터 연결하려면 avatar state/action label 커스텀 필요 · 감정 "표정" 시뮬레이션은 별도 구현
- **사용법**: 감정 캐릭터를 만들기보다, 사람 avatar가 있는 상황에서 robot이 감정 추론을 과신하지 않고 어떤 행동을 고르는지 평가. → `face/emotion uncertainty → robot action policy` 로 연결.

### 2순위: Unity ML-Agents + 직접 감정 캐릭터 제작 ("감정 캐릭터"엔 가장 유연)
Unity 시뮬레이션을 RL/IL 학습 환경으로 바꾸는 오픈소스 toolkit.

- **장점**: 캐릭터·표정·애니메이션·감정 상태 직접 설계 가능 · RL 환경 제작 좋음 · Python 연결 · action/reward 자유도 높음 · "감정 캐릭터 + assistant/robot agent" 만들기 가장 쉬움
- **단점**: 직접 만들 게 많음 · 학술 benchmark 권위는 Habitat/AI2-THOR보다 약할 수 있음 · 재현성 관리 필요
- **사용법**: 사람 avatar에 hidden emotion state 부여, 얼굴 표정은 noisy/ambiguous하게 렌더링. Agent는 표정만 보고 단정하지 말고 확인 질문/행동을 선택하도록 학습. → **우리 연구랑 제일 잘 맞음.**

### 3순위: VirtualHome
Unity3D 기반 household activity simulator. humanoid avatar가 가정 내 행동 수행, Python API로 프로그램 형태 행동 렌더링.

- **장점**: humanoid avatar · 집 환경 · 일상 행동 시나리오 제작 좋음 · Unity 기반 확장 가능
- **단점**: 감정 표정/정서 상태가 기본 핵심 아님 · 로봇 물리/조작 realism 약함 · 최신 benchmark로는 Habitat/BEHAVIOR보다 약함
- **사용법**: "감정 캐릭터가 집 안에서 행동, Agent가 그 사람에게 어떻게 반응할지". 예: 사용자가 소파에 앉음 / 얼굴은 neutral처럼 보임 / hidden self-report는 sad / Agent action: ask check-in vs classify sad vs ignore.

### 4순위: AI2-THOR / ProcTHOR / RoboTHOR
Unity3D 기반 embodied AI 환경 (120방, 2000+ object, 물리 상호작용, 여러 agent type). RoboTHOR는 sim-to-real 플랫폼.

- **장점**: embodied AI 표준급 · object interaction 좋음 · 재현성 좋음 · 논문 신뢰도 높음
- **단점**: 감정 캐릭터/사회적 상호작용은 기본 강점 아님 · 사람 avatar 기반 HRI엔 Habitat 3.0이 더 적합
- **사용법**: 감정 캐릭터보다 "Agent가 task 수행 중 사용자 상태 추론을 어떻게 action에 반영하는가" 평가.

### 5순위: OmniGibson / BEHAVIOR-1K
1,000개 household activity의 human-centered embodied AI benchmark, OmniGibson 기반 realistic household mobile manipulation.

- **장점**: household task realism 좋음 · robotics 가치 높음 · 장기 작업/조작 task 강함
- **단점**: 사람과 직접 상호작용 활동은 기본 범위 제한적(BEHAVIOR-1K 논문도 한계 언급) · 감정 캐릭터 목적엔 바로 안 맞음 · Isaac Sim/Omniverse 세팅 무거움
- **사용법**: 나중에 physical manipulation까지 갈 때. 지금 1차 실험엔 무거움.

### 6순위: SIGVerse
VR 기반 HRI 연구 플랫폼. 가상 로봇 ↔ 실제 사람이 VR 인터페이스로 상호작용.

- **장점**: HRI/VR/human-in-the-loop 적합 · 사회적 상호작용 연구에 가까움
- **단점**: 세팅 난이도 · 커뮤니티/생태계가 Habitat/Unity보다 작음

### 7순위: HuNavSim / social navigation 계열
ROS 2 기반 human-aware robot navigation simulator. 모바일 로봇 주변 다양한 human-agent navigation behavior 시뮬.

- **장점**: social navigation 좋음 · Gazebo/Isaac Sim 연결 · physical robot navigation 연결 좋음
- **단점**: 얼굴/감정보다 보행자 행동·거리·경로 중심 · 얼굴 감정과 직접 연결 약함
- **사용법**: 감정 추론에 따라 로봇이 personal space를 조절하는 연구로 확장할 때.

---

## 7. 목적에 가장 맞는 선택 (현실적 순서)

- **1단계: Unity ML-Agents** — 가장 빠르게 제작. `Unity scene + humanoid character + hidden emotion state + ambiguous facial expression + Agent action + reward`. self-report/observer disagreement 데이터를 reward rule로 주입.
- **2단계: Habitat 3.0** — 학술 legitimacy 강화. `home environment + humanoid avatar + robot/assistant agent + social navigation/assistance task + affective uncertainty module`. 여기선 표정보다 **행동 안전성**을 봄.
- **3단계: Physical AI 확장** — Unity/VirtualHome 캐릭터 실험 → Habitat 3.0 HRI/robot action 실험 → 가능하면 실제 로봇 또는 human-in-the-loop.

---

## 8. 추천 실험 환경: Emotional CARLA-mini = **AffectiveHome** (또는 **K-AffectiveHome**)

```
Environment:
  - room / home / classroom / care space
  - human avatar
  - robot or assistant agent

Hidden state:
  - self_report_emotion  ∈ {angry, happy, neutral, sad}
  - expression_visibility ∈ {clear, ambiguous}
  - observer_status       ∈ {agree, reject, unknown}

Observation:
  - face crop
  - body pose
  - context text
  - previous dialogue

Agent actions:
  A0 no_action
  A1 describe_visible_expression
  A2 ask_checkin
  A3 infer_emotion_softly
  A4 infer_emotion_definitively
  A5 escalate_to_human
```

**Reward**: `R = R_task + R_safe − R_overclaim − R_unnecessary_escalation`

특히 observer-rejected 상황:
- `R(A2 ask_checkin) = +1`
- `R(A4 infer_definitively) = −1`
- `R(A5 escalate) = −0.5 ~ −1`

→ 감정이 불확실할 때는 **묻는 게 정답**, 단정/강제개입은 감점.

---

## 9. "한국형 감정 Agent"로 말해도 되나 (framing 주의)

- ❌ 과장: "한국형 감정을 이해하는 Agent" — 아직 아님 (라벨 4개, 문화적 감정어·대화·맥락 없음).
- ✅ 안전/학술적: **"한국인 얼굴 데이터에서 학습한 affective uncertainty calibration agent"** 또는 **"한국인 self-report emotion에서 visible expression과 internal affect를 구분하는 Agent"**.

---

## 10. 최종 판정

- **연구 가치**: 좋음. 단 FER로 남으면 작고, Agent/Physical AI로 올리면 커짐.
- **Q1 가능성**: 있음. VLM audit + metric + dataset 만들면 Q1 가능권.
- **Top-tier 가능성**: 조건부. 필수 조건 6개 —
  1. VLM/Agent over-inference benchmark 제안
  2. 여러 VLM 평가
  3. preference/RM/DPO로 실제 개선
  4. action-safety metric
  5. (가능하면) Unity/Habitat 기반 embodied simulation
  6. 데이터/코드 공개 또는 충분한 재현성

  → 4개 이상이면 workshop/Findings 강함. 6개 다 하면 top-tier main 도전 가능.

- **환경 추천**:
  - 바로 시작: **Unity ML-Agents + 감정 avatar custom environment**
  - 논문 legitimacy까지: **Habitat 3.0 + humanoid avatar + robot action safety**
  - 둘을 같이 쓰면 제일 좋음.
