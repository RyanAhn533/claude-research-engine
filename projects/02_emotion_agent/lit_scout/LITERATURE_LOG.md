# Lit-Scout 로그 (human-readable digest)

> 매 iteration 갱신. 구조화 기록은 findings.jsonl. 방향 = DIRECTION.md.

## iter 1 — 2026-06-11 (메커니즘 도구 사냥)
**대박 발견: TR/TL 분해가 우리 #1 약점(positive 메커니즘)을 메울 도구.**

### 🔴 high / mechanism (바로 차용)
- **TR vs TL 분해** (Pan&Gao 2023 ACL / Min 2022 EMNLP): gold/random label ICL → **TL = gold−random**. 예측: AU(novel) gain=대부분 TL(random label 주면 무너짐), text(familiar)=TR(random 괜찮음). **scalar가 gain 예측 + thesis 재정의**("ICL은 Task Learning 필요할 때만, novel 포맷이 강제"). → **exp_040으로 즉시 실행.**
- **Task Vectors** (Hendel 2023 EMNLP): mid-layer task-vector patch, recovery ratio R. AU가 더 깨끗한 task vector 형성 예측 → R vs gain 상관.
- **Function Vectors** (Todd 2023 ICLR'24): head별 AIE, FV-recovery scalar. **같은 top-AIE head가 Qwen-7B/14B엔 있고 Mistral엔 없으면 = Mistral confound를 메커니즘으로 설명.**
- **Label Words are Anchors** (Wang 2023 EMNLP best): info-flow S_pq(deep). 싼 backward pass 1번. AU에서 높을 것 예측.

### med / framing·generalization
- **Razeghi 2022**: pretraining term-frequency가 few-shot 좌우. → **format-schema 빈도**(token perplexity 아님, 그건 우리한테 실패)로 familiarity 재조작.
- **Wang 2024 ICLR'25 (OOD ICL)**: AU=pretrained hypothesis space 밖 → 진짜 매핑 강제. 이론 scaffold + inverted-U 실패모드 예측.
- **Wang 2024 ACL (pretraining TR/TL dynamics)**: pretraining mix가 TR/TL 균형 결정 → Qwen재현/Mistral confound 설명.

### low / 관련연구
- **TAG 2026 (arXiv 2602.18763)**: AU-grounded LLM FER. AU-as-text 최근접 prior → related work + baseline.

### 다음 iter 팔 것
- task-vector/function-vector 측정 코드 선례(repo) 찾기
- "format frequency" 추정 proxy 방법
- AAAI 2025/2026 ICL accepted 논문 직접 스캔(우리 포지셔닝 차별화)

### ACTION (우선순위)
1. **exp_040: TR/TL (gold vs random label) on AU vs IEMOCAP, 3 models** ← 지금 실행
2. exp_041: S_pq info-flow (label-anchor) — backward-pass saliency
3. (옵션) function-vector AIE — Mistral 설명용
