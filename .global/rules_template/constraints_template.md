# Constraints — TEMPLATE

각 프로젝트가 복사해서 채움.

## 평가 프로토콜 (변경 금지)
- Data split: ...
- Sample size: ...
- Primary metric pipeline: ...
- Report format: mean ± std + p-value

## 데이터
- 원본 경로 변경 금지
- Meta/label 변경 금지

## 하드웨어/환경
- GPU VRAM 여유 최소 Xgb
- Batch size ≤ ...
- 환경 오염 금지

## 라이선스/윤리
- ...

## 자동화 제약
- Baseline 교체 = JY 승인
- Thesis 변경 = JY 승인
- scripts/kill_switch 존재 시 즉시 중단

## Iteration 내 금지
- 평가 프로토콜 수정 → 즉시 실격
- Metric 체리피킹 → 즉시 실격
- 동일 method 재시도 (paper_tried 체크)
- 최근 10 iter 내 동일 category 4회 이상
- 학습 데이터 leakage 의심 → review_queue flag
