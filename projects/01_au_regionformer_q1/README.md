# Project 01 — AU-RegionFormer Q1

**Thesis**: "한국인 감정 표현에서 AI가 포착한 region은 심리학이 예측한 region과 괴리가 있다. 298명의 사회적 합의는 어느 쪽에 가까운가?"

**Target**: IEEE TAFFC (IF 11, realistic 65-75%) / Nature Human Behaviour (dream 22-28%)

**Direction ID prefix**: `AUR-D###`

## Status (2026-04-22)

- **Phase**: 0 완료, Phase 1-4 대기
- **Iterations**: 9
- **Best metric**: 87.56% (exp_009 triplet fusion: Region + Landmark + kNN graph)
- **vs v1 학습 모델**: +7.86%p (linear probe no-training으로)

## Quick links

| 목적 | 경로 |
|-----|-----|
| 논문 drafting | `Q1_WORKING.md` |
| 전략 마스터 | `research_log/STRATEGY.md` |
| 현재 리더보드 | `state/leaderboard.jsonl` |
| 인사이트 | `state/insights.md` |
| 선행연구 | `research_log/references/` |
| Logic chains (RLHF) | `research_log/claude_evaluation/` |
| 실험별 기록 | `research_log/experiments/` |
| 실험 코드 | `src/analysis/phase0/` |
| 결과물 (figure/json/summary) | `results/phase0/` |

## 외부 링크

- **원본 코드 repo (가중치/학습 코드 원본)**: https://github.com/RyanAhn533/AU-RegionFormer (public)
- **큰 가중치 파일 (.pth)**: 원본 repo 또는 로컬 `/home/ajy/AU-RegionFormer/outputs/`
- **원본 이미지**: `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/`
- **Region embedding (.npy)**: `/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings/`

## 로드맵

| Phase | 기간 | 핵심 | 상태 |
|-------|------|------|-----|
| 0 진단 + AU 기초 | 3-4주 | Region probe, AU extraction | ✅ iter 1-4 |
| 1 한국형 AU 지도 | 4주 | Jack 재검증, FACS, region×clean | ✅ iter 3 |
| 2 사회적 합의 | 4주 | 연세대 consensus + demographic | ✅ iter 2, 5 |
| 3 Graph learning | 6주 | kNN prior, triplet fusion | ✅ iter 6, 7, 9 |
| 4 Cross-cultural | 6주 | AffectNet/KUFEC-II | ❌ 대기 |

## 다음 iter 후보

- Cross-cultural AU extraction on AffectNet
- Per-class F1 weighted probe (슬픔/분노 floor 개선)
- Triplet robustness on raw (mixed) subset
