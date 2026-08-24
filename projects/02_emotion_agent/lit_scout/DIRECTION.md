# Lit-Scout 방향 정의 (에이전트가 매 iteration 읽는 기준)

> 목적: 우리 논문 방향에 맞는 AAAI/ACL/EMNLP/NeurIPS/ICLR 논문을 계속 읽고,
> **차용 가능한 방법론**을 뽑아 findings.jsonl에 누적. 특히 우리 약점을 메우는 도구 우선.

## 우리 논문 (한 줄)
**Modality-gated ICL**: in-context learning은 입력 *포맷*이 LLM에 낯설 때만(=모델이 그
포맷→과제 매핑을 안 배웠을 때) 크게 작동한다. Korean AU-intensity-text(novel) ICL +10~17pp
vs 영어 대화(familiar) ≤2pp. Qwen 계열 robust, Mistral은 confound.

## 우리의 약점 (= 스카우트가 메워야 할 것, 우선순위)
1. **🔴 robust한 positive 메커니즘 부재.** perplexity(음의상관 배제), 표현압축(null), format-swap(model-dependent) 다 실패. → "ICL gain을 예측하는 측정 가능한 양"이 필요. 후보 도구: task vectors / function vectors / induction heads / in-context vectors / label-space 분석.
2. **현상 일반화** — 지금 AU 하나·Qwen 위주. 2번째 modality·더 많은 패밀리로 일반화하는 프레이밍/방법.
3. **confound 분리** — format-novelty vs task-novelty를 깨끗이 가르는 실험설계 선례.
4. **프레이밍/포지셔닝** — "when does ICL help" 계열에서 우리 기여를 어떻게 차별화하나.

## 스카우트가 매번 할 것
1. findings.jsonl(이전 누적) 읽어서 **중복 회피 + 다음에 깊이 팔 주제** 정하기.
2. 아래 축으로 최신/고인용 논문 검색·정독:
   - "when does in-context learning work/fail" / ICL 효과 조건
   - ICL 메커니즘: task vector, function vector, induction head, in-context vector
   - ICL vs fine-tuning / label-space & format effect in ICL
   - LLM emotion/affect recognition, FACS/AU + LLM
3. 논문마다 추출: 출처 · 1줄 finding · **방법론** · **우리에 어떻게 graft** · applicability(high/med/low) · 우리 약점 중 무엇을 메우나.
4. findings.jsonl append + LITERATURE_LOG.md 갱신 + **top graft 제안 1-3개**를 ACTION_QUEUE.md에.

## graft 채점 기준
- high = 우리 약점(특히 #1 메커니즘)을 직접 메우고, 우리 데이터/코드로 1주 내 시도 가능.
- med = 프레이밍/관련연구 강화.
- low = 배경지식.
