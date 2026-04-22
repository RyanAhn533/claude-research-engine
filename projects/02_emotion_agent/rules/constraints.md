# Constraints — Emotion Agent Q1

## 평가 프로토콜 (변경 금지)
- **IEMOCAP**: 5-fold CV on 5 sessions (session-independent) — MER 표준
- **MELD**: train/dev/test official split 사용
- **DEAP**: 32-subject LOSO (Leave-One-Subject-Out) — bio 표준
- **KEMDy20 / K-EmoCon**: SGMT 논문 split 재활용
- Primary metric pipeline: `Macro F1 per dataset`, 3-seed mean ± std
- Statistical test: bootstrap 1000 or paired t-test
- Agent reasoning: **same prompt template** for all runs (변경 시 iter 별로 명시)

## 데이터
- 원본 경로 변경 금지 (`setup/SETUP.md` 매핑 고정)
- Label 변경 금지 (agent pseudo-label은 별도 컬럼)
- **연세대 298 consensus data 사용**: 컨소시엄 합의 필요 (Moon 교수 경유)

## 하드웨어 / 환경
- **GPU VRAM free 12GB 이상 유지 (다른 사용자 보호)** → 우리 ≤ 7.7GB
- Qwen2.5-7B는 4-bit quant 강제 (FP16 15GB 금지)
- Batch size: inference는 1-4, training은 relative small
- `/` disk 사용 금지 (87% full). 모든 large output은 `/data` or `/mnt/hdd`로
- Base env 오염 금지 → `cre_q1` 전용 env

## 라이선스 / 윤리
- 연세대 raw response 외부 공개 금지
- Qwen2.5 Apache 2.0 — OK
- IEMOCAP/MELD: 학술 사용 OK, 재배포 금지
- POPR 특허 자료는 별도 — 본 프로젝트에 섞지 않음

## 자동화 제약 (semi-autonomous)
- **Baseline 교체 = JY 승인** (leaderboard 1위 갱신 시)
- **Thesis 변경 = JY 승인**
- **Moon 교수 미팅 자료 변경 = JY 승인**
- 외부 API 호출 (WebSearch 제외) = 승인 필요
- `scripts/kill_switch` 존재 시 즉시 중단
- `review_queue/` 2개 이상 누적 시 자동 루프 일시정지

## Iteration 내 금지
- 평가 프로토콜 수정 → 즉시 실격
- Metric 체리피킹 (fold 선택) → 즉시 실격
- 동일 method 재시도 (`paper_tried.jsonl` 체크)
- 최근 10 iter 내 동일 category 연속 3회 초과 금지
  - categories: `data / perception / reasoning / cultural_prior / bio_grounded / ensemble / ablation`
- 학습 데이터 leakage 의심 → `review_queue/`로 flag

## LLM 사용 제약
- Primary LLM: **Qwen2.5-7B-Instruct (4-bit)** — local, 재현 가능
- VL 필요 시: Qwen2.5-VL-7B (4-bit)
- Claude API 사용 시 **iter마다 명시** (비용 + 비재현)
- Prompt template 변경 = iter 별로 bump

## Direction ID
- Prefix: `EMA-D###`
- Global `methodology/directions.jsonl`에 append (project=`02_emotion_agent`)
