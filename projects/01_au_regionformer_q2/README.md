# Project 01 — AU-RegionFormer (Q2)

> **이 폴더는 뭐하는 곳**: Korean Multi-view FER 논문 (Q2). 후배 1저자, JY 2저자, 연세대 공저, Moon 교수 교신. ESWA/PRL target.
> **이 파일은 뭐**: 후배에게 넘길 때 10분 안에 파악 가능한 **지시 문서**.

---

## 1. 현재 상태 (완료된 것)

11 iteration 돌림. **Best 수치 확정**.

| 항목 | 값 |
|-----|-----|
| Best accuracy | **87.56%** (3-fold CV, linear probe, clean subset) |
| vs v1 학습 모델 | **+7.86%p** (v1 = 79.7%) |
| Per-class F1 | 기쁨 0.949 / 분노 0.840 / 슬픔 0.835 / 중립 0.869 (전부 >0.83) |
| Iterations | 11개 (leaderboard 15 entries) |

## 2. 논문 2축 Novelty

1. **Social-consensus-aware filtering**: 연세대 298명 검증 라벨로 label noise 제거 → +2.78%p
2. **Multi-view orthogonal fusion**: Region(CNN) + Landmark(geom) + kNN graph — AU는 redundant, Landmark는 orthogonal 명시적 입증

+ 보조 finding: **Multi-layer Jack 2012 counter-evidence** (Region/AU/Landmark 3 layer 모두 mouth 우위) → reviewer 공격 방어

---

## 3. 🎯 다음 할 일 (순서대로)

### Step 1 — Writing 시작 (1-2개월)
논문 각 section에 이미 데이터 있음. 아래 mapping대로 채우면 됨.

| 논문 section | 쓸 실험 결과 | 위치 |
|------------|-----------|-----|
| §3 Method | Triplet fusion (exp_009) | `experiments/exp_009_three_way/` |
| §4.1 Main result 87.56% | exp_009 + exp_011 | `results/phase0/linear_probe_acc.png` |
| §4.2 Per-class F1 | exp_011 | `experiments/exp_011_perclass_landmark_importance/` |
| §4.3 Landmark importance | exp_011 lip_corner_angle top | 같음 |
| §4.4 Ablation (Region/AU/Graph/Landmark) | exp_004, 006, 008, 009 | leaderboard entries |
| §4.5 Robustness (mixed vs clean) | exp_010 | `experiments/exp_010_triplet_mixed/` |
| §4.6 Social consensus ablation | exp_002, 003 | `experiments/exp_003_jack_clean_per_region/per_region_raw_vs_clean.png` |
| §4.7 Fairness | exp_005 demographic | `results/phase0/fairness_gap.png` |
| §5 Limitations | AU redundant (exp_004) + cultural specificity | `state/paper_tried.jsonl` |
| §2 Related work | 선행연구 조사 | `research_log/references/au_region_importance_2026-04-21.md` |

### Step 2 — 추가로 돌릴 여지 (선택)
남은 여유가 있으면:
- `experiments/exp_003_mlp_probe/design.md` — MLP probe ablation (§3.3 method justification 강화)
- Graph k sweep (k=10/30/100) — hyperparam 정당화
- 여의치 않으면 skip. 이미 논문 submit 가능 수준.

### Step 3 — 연세대 공저 협의
산업부 컨소시엄 데이터 사용 → 연세대 측 공저자 1명 필요. **Moon 교수 경유하여 offer**.

### Step 4 — Submit
- Target: **ESWA (IF 7.5)** or **PRL (IF 5.1)**
- 저자 순서: 후배 1저자 · JY 2저자 · 연세대 공저 · Moon 교수 교신

---

## 4. 파일 어디에 뭐 있나

```
01_au_regionformer_q2/
├── README.md                        ← 이 파일 (지시용)
├── rules/                           ← 실험 규칙 (변경 금지)
│   ├── target_metrics.md
│   └── constraints.md
├── state/                           ← 누적 기록
│   ├── leaderboard.jsonl            ← 15 entries, 모든 iter 수치
│   ├── paper_tried.jsonl            ← 시도한 방법 (중복 방지)
│   └── insights.md                  ← iter 패턴 정리
├── experiments/                     ← iter별 실행 코드 + 결과
│   ├── exp_001_sad_path_fix/ ... exp_011_perclass_landmark_importance/
│   └── (각 폴더: design.md, run.py, results.json, summary.md)
├── research_log/                    ← 세션 로그 + 선행연구 + Claude 평가
│   ├── STRATEGY.md                  ← 전략 마스터
│   ├── sessions/                    ← 논의 세션 기록
│   ├── experiments/                 ← 실험별 자세한 레포트 (8-section 포맷)
│   ├── references/                  ← 선행연구 (Jack 2012, FACS, KUFEC-II)
│   └── claude_evaluation/           ← Claude direction + logic chain
├── results/phase0/                  ← 논문 넣을 figure/table
│   ├── linear_probe_acc.png
│   ├── per_region_raw_vs_clean.png
│   ├── fairness_gap.png
│   ├── per_au_ranking.png
│   ├── au_group_comparison.png
│   ├── cm_pooled_*.png
│   └── *_summary.md, *_results.json
├── src/analysis/phase0/             ← 재실행용 python 코드
└── PROJECT_STATUS.md                ← 데이터 위치 + 실험 현황
```

---

## 5. 실험 재현 방법

### 필요 것
- Python 3.10+, torch 2.6+, sklearn, pandas, umap-learn
- GPU: inference only (linear probe은 CPU도 가능)

### 데이터 경로 (원본 repo에 있음)
- Region embedding: `/home/ajy/AU-RegionFormer/data/label_quality/au_embeddings/*.npy`
- Landmark features: `/home/ajy/AU-RegionFormer/data/label_quality/face_features.csv`
- AU intensity (OpenGraphAU): `/home/ajy/AU-RegionFormer/data/label_quality/au_features/opengraphau_41au_237k_v2.parquet`
- Yonsei consensus: `/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv`
- Original images: `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/`

### 주요 실험 재실행
```bash
cd /home/ajy/AU-RegionFormer
python experiments/exp_009_three_way/run.py   # Triplet best
python experiments/exp_011_perclass_landmark_importance/run.py
```

각 `experiments/exp_NNN/run.py`는 독립 실행. 결과는 같은 폴더에 저장.

---

## 6. 외부 링크

- 코드 원본 repo: https://github.com/RyanAhn533/AU-RegionFormer (public)
- 모델 가중치 (.pth): `/home/ajy/AU-RegionFormer/outputs/` (원본 repo는 gitignore)

---

## 7. 저자 정보

| 역할 | 담당 |
|-----|-----|
| 1저자 | 석사 후배 |
| 2저자 | JY (실험 설계 + 11 iteration 분석) |
| 공저 | 연세대 컨소시엄 측 (데이터 권한) |
| 교신 | Yeon-Kug Moon (Sejong University Heart Lab) |

---

## 8. 막히면

1. `research_log/STRATEGY.md` — 전체 전략 맥락
2. `research_log/sessions/` — 과거 논의 기록
3. `state/insights.md` — 패턴 정리
4. `state/paper_tried.jsonl` — 시도 방법 전체
5. JY에게 물어보기
