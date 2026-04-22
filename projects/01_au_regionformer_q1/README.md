# Project 01 — AU-RegionFormer Q1

## Status (2026-04-22)

- **Q2 draft-ready** → 석사 후배 1저자로 인계 (ESWA/PRL target)
- **Q1 agent angle로 pivot 중** → JY 1저자, Moon 교수 교신

## Q2 정리 (후배 인계)

| 항목 | 값 |
|-----|-----|
| Best metric | **87.56%** linear probe (clean subset) |
| vs v1 학습모델 | **+7.86%p** |
| Per-class F1 | 전 class > 0.83 |
| Iterations | 1 ~ 11 (모두 `state/leaderboard.jsonl`에 기록) |
| 저자 구조 | 후배 1저자 · JY 2저자 · 연세대 공저 · Moon 교신 |
| Figure/Table 준비 | ✅ `results/phase0/` |

**Q2 논문 thesis 2축**:
1. Social-consensus-aware filtering (298명 컨소시엄 검증 데이터)
2. Multi-view orthogonal feature fusion (Region + Landmark + Graph, AU redundant 입증)

## Q1 (agent pivot, 앞으로)

**Thesis draft**: *"Culturally-aware multimodal emotion agent with bio-grounded labeling."*

### 3 축
1. Bio-grounded labeling agent (JY 다른 서버 진행 중)
2. Cultural priors (Jack 2012 × 한국인 데이터, Q2 finding 흡수)
3. LLM reasoning loop (perception → reasoning → action)

### Target venue
- 현실: ICMI / ACII / Sensors (IF 5-10)
- Stretch: IEEE TAFFC (IF 11)
- 연세대 공저 확보 시 NHB/PNAS 자격 ↑

## Next actions

| # | 내용 | 담당 |
|---|-----|-----|
| 1 | Moon 교수 sync: Q1 agent pivot + 연세대 공저 offer | JY |
| 2 | Bio-grounded labeling 실험 현황 통합 | JY (다른 서버) |
| 3 | Q1 agent outline draft | JY + Claude (Moon sync 후) |
| 4 | Q2 writing guide + 후배 인계 package | JY → 후배 |

## Quick links

| 목적 | 경로 |
|-----|-----|
| 논문 drafting | `Q1_WORKING.md` |
| 전략 마스터 | `research_log/STRATEGY.md` |
| 리더보드 | `state/leaderboard.jsonl` |
| 인사이트 | `state/insights.md` |
| 선행연구 | `research_log/references/` |
| Logic chains (RLHF) | `research_log/claude_evaluation/` |
| 실험별 기록 | `research_log/experiments/` |
| 실험 코드 | `src/analysis/phase0/` |
| 결과물 | `results/phase0/` |

## 외부 링크

- 코드 원본: https://github.com/RyanAhn533/AU-RegionFormer (public)
- 이미지 데이터: `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/`
- 가중치: `/home/ajy/AU-RegionFormer/outputs/` (원본 repo에 gitignore)

## Direction ID prefix
`AUR-D###` (이미 13개 축적)
