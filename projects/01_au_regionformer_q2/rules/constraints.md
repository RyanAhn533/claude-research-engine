# Constraints — 건드리면 실격

## 평가 프로토콜 (변경 금지)
- **Data split**: stratified, seed=42, 3-fold CV
- **Sample size**: 30K stratified subsample (per-class 7,500 balance)
- **Primary metric pipeline**: embedding → L2 norm → PCA(256) → Standardize → LogReg 3-fold
- **Class set**: 4 emotion (기쁨/분노/슬픔/중립) — 7 emotion 확장은 별도 연구
- **Report format**: mean ± std across folds + p-value vs current best (bootstrap 1000)

## 데이터
- 원본 이미지 경로 변경 금지
- meta CSV 파일(`meta_mobilevitv2_150.csv`, `meta_convnext_base.fb_in22k_ft_in1k.csv`) 재생성 금지
- 연세대 298명 검증 라벨 (`is_selected`) 변경 금지

## 하드웨어/환경
- GPU VRAM 12GB 이상 항상 free 유지 (다른 사용자 보호)
- Batch size 64 초과 금지 (OpenGraphAU inference)
- Base env (conda default) 오염 금지 — 신규 라이브러리는 별도 env
- / 디스크 90% 초과 금지 (대용량은 /data 또는 /mnt/hdd)

## 라이선스/윤리
- 연세대 298명 raw 응답 외부 공개 금지 (IP)
- Phase 5 KMER 시뮬레이터는 본 연구 범위 제외 (별도 프로젝트)
- POPR 특허 자료 `/data` 이관 금지 (feedback_ip_protection_storage 참조)

## 자동화 제약 (semi-autonomous)
- **Baseline 교체 = JY 승인 필수** (leaderboard 1위 갱신 시 자동 교체 금지)
- **Thesis 변경 = JY 승인 필수**
- **외부 API 호출**: 승인 없이 WebSearch 제외 어떤 external API도 금지
- **scripts/kill_switch 파일 존재 시 즉시 중단**
- **review_queue/** 디렉토리에 pending 항목 2개 이상 쌓이면 자동 루프 일시정지

## Iteration 내 금지
- 평가 프로토콜 수정 (train/test 비율 바꾸기 등) → 즉시 실격
- Metric 체리피킹 (cherry-pick fold 결과) → 즉시 실격
- 같은 method 재시도 (`paper_tried.jsonl` 체크)
- 최근 10 iteration 내 동일 카테고리 4회 이상 (regularization/architecture/data aug/training/inference/ensemble)
- 학습 데이터 leakage 의심 → review_queue로 flag
