# 지금 우리가 뭐 하는 중인가 (AAAI 2027 push)

> 길 잃었을 때 이 파일부터 봐. 마지막 갱신: 2026-06-11 (라이브 진행상황 §5).

---

## 1. 한 줄

너의 흩어진 프로젝트(01~06) 중 **상위 컨퍼런스 ceiling이 가장 높은 자산 하나**(`02_emotion_agent`의 ICL 발견)를 골라, **AAAI 2027 (마감 7/27)** 에 통과시키려고 밀고 있다.

## 2. 왜 이거냐 (포트폴리오 판단)
- AU-RegionFormer(연세대 298명) = **저널**(TAFFC) 트랙, 컨퍼런스 아님.
- CBBF/VisionMER = 저널/응용. CARLA_agent = baseline 빌려온 미완성(park). DMS = 응용 conf.
- → **컨퍼런스 최상위로 갈 수 있는 건 emotion_agent 하나.** 그래서 여기 화력 집중.
- 상세: [`PORTFOLIO_VENUE_MAP.md`](PORTFOLIO_VENUE_MAP.md)

## 3. Thesis (논문이 주장하는 것) — 쉽게
> **In-context learning(few-shot 예시 넣기)은 "modality-gated"다 — LLM에게 입력 형식이 *낯설 때만* 작동한다.**

- 너는 같은 4감정 과제를 **낯선 입력 1개**(얼굴 AU를 "AU6(cheek raiser)=75" 텍스트로) + **익숙한 입력 2개**(영어 대화 IEMOCAP/MELD)로 가지고 있다.
- 예측: 낯선 AU에선 ICL이 확 오르고(예시가 임시 fine-tune 역할), 익숙한 영어엔 flat.
- 메커니즘(이게 진짜 무기): ICL이 **hidden representation을 압축**(within-class 거리 ↓22%)하는 곳에서만 이득. IEMOCAP에선 오히려 확장(+12.7%) → ICL 무효. = 압축이 이득을 인과적으로 예측.
- 감정/문화는 headline 아니라 **testbed**. headline = "ICL은 언제 작동하는가"라는 일반 ML 명제.
- 재구성 상세: [`AAAI_DRAFT_v0.md`](AAAI_DRAFT_v0.md), [`PAPER_TOPTIER_REFRAME.md`](PAPER_TOPTIER_REFRAME.md)

## 4. 지금 돌리는 실험이 뭘 증명하나
**문제**: 기존 결과는 Qwen 한 모델뿐 → 상위 venue에서 "다른 모델에선?" 한 줄에 desk-reject.
**실험(exp_035)**: 게이팅 패턴(낯선↑ + 익숙flat)을 **4개 모델**에서 재현 → 단일모델 아티팩트 차단.
- 모델: Qwen2.5-7B(검증/재현) · Mistral-7B(다른 패밀리) · Phi-3.5(MS 패밀리) · Qwen2.5-14B(스케일)
- 각 모델 × {AU novel, IEMOCAP/MELD familiar} × 3 seed × {zero-shot, ICL k4}
- batched 추론으로 ~23배 가속(400샘플 34초). 4-bit 유지(published와 비교 가능).

## 5. 🟢 패널 결과 (완료 2026-06-11 17:09, 3 seed × N=400)

ICL gain (zero-shot → ICL k4, pp):

| 모델 | AU (novel) | IEMOCAP (fam) | MELD (fam) | novel/fam 비 |
|---|---|---|---|---|
| Qwen2.5-7B | **+10.1** | +1.0 | +0.7 | ~12× |
| Mistral-7B | **+15.2** | +2.1 | +0.0 | ~15× |
| Qwen2.5-14B | **+16.7** | +5.5 | +4.7 | ~3.3× |
| Phi-3.5 | (드롭) | — | — | transformers Phi3-longrope device 버그(4-bit/fp16 둘 다) |

**판정 (정직):**
- ✅ **핵심 성립**: novel 입력 ICL gain이 **3모델·2패밀리·2스케일 전부 +10~+16.7pp**. → headline의 단일모델 desk-reject 죽음.
- ⚠️ **wrinkle**: "familiar = flat"은 7B 둘은 깨끗(≤2pp)인데 **Qwen-14B는 familiar에서 +5pp 실제 gain**. 진단 결과 파싱 아티팩트 아님 — 14B zero-shot의 neutral 과예측을 ICL이 bias 교정. → "familiar 무조건 flat"은 못 씀.
- ✅ **방어 가능한 claim**: *낯선 입력 ICL gain이 익숙한 입력보다 항상 훨씬 큼(비 3.3~15×, 보편)*. 평균 novel +14.0pp vs familiar +2.3pp.
- 🔬 **추가 발견(논문 강화)**: familiar에서 ICL gain이 날 땐 그 정체 = label-bias 교정(새 정보 아님). novel gain은 그보다 큰 무언가(포맷/표현 학습). 이 구분이 메커니즘 §를 풍부하게 함.

원자료: `cache/gating_{qwen7b,mistral,qwen14b}.json`, 진단: `logs/diag_14b.log`

## 6. 다음 (패널 끝난 뒤)
1. 4×3 전체 표 종합 + 통계(paired delta) → AAAI_DRAFT §4.4에 박기
2. (선택) 메커니즘(표현 압축)도 멀티모델 확인
3. abstract/§1을 modality-gated ICL headline로 본문 재작성
4. leaderboard.jsonl 로깅(엔진 프로토콜)

## 7. 디렉토리 주소 (네가 요청한 것)

| 무엇 | 경로 |
|---|---|
| **AAAI 프로젝트 루트** | `/home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/` |
| 이 문서 | `…/02_emotion_agent/_WHAT_WE_ARE_DOING.md` |
| 멀티모델 실험 | `…/02_emotion_agent/experiments/exp_035_multimodel_gating/` |
| └ 실행 로그 | `…/exp_035_multimodel_gating/logs/{qwen7b,mistral,phi35,qwen14b}_batched.log` |
| └ 결과 JSON | `…/exp_035_multimodel_gating/cache/gating_<model>.json` |
| └ 순차 드라이버 | `…/exp_035_multimodel_gating/run_panel.sh` |
| 재구성/전략 문서 | `…/02_emotion_agent/{AAAI_DRAFT_v0,PORTFOLIO_VENUE_MAP,PAPER_TOPTIER_REFRAME}.md` |
| 기존 논문 드래프트 | `…/02_emotion_agent/Q1_WORKING.md` |
| CARLA(park됨) 재개노트 | `/mnt/hdd/carla/adas_framework/outputs/train_tfpp/RESUME_NOTE.md` |

## 8. 빠른 진행 확인 명령
```bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_035_multimodel_gating
grep -E ">>>|SUMMARY" logs/*_batched.log     # 모델별 gain 요약
tail -3 logs/panel.log                        # 패널 단계
```
