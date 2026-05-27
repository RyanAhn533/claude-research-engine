# exp_001_congruence_head — Structure A (Minimal Multi-task)

direction_id: AURP6-D010
date: 2026-05-23
stage: paper_ready
parent: Stage 6 full (F1=0.9256)
linked_claim: C1_per_image_alignment_recoverable

## Motivation

Stage 9 v_yonsei 재실행 결과 per-image C2 Pearson = 0.114 (옛 0.109 대비 +0.005, H1 falsified).
Region ablation 결과 8 region zero-mask 시 Δ acc ±0.0001 (H2 falsified).
→ **Per-image perceptual congruence 학습 신호가 모델에 직접 들어오지 않고 있다.**

본 실험은 **명시적 congruence head** 를 추가하여 per-image reject_rate 를 직접 학습 target 으로 둔다.

## Hypothesis (Pre-registered)

### H1 (Primary)
새 congruence head 추가 시:
- **emotion F1 = 0.9256 ± 0.005** (baseline 유지, regression 0.5pp 이상 발생하면 fail)
- **C2 per-image Pearson ≥ 0.15** (baseline 0.114 대비 +0.04 이상)
- delta_threshold: F1 -0.005, C2 +0.04

### H1 Null
- F1 baseline 대비 0.5pp 이상 떨어짐 또는
- C2 변화 < +0.02 (congruence head 학습이 alignment 신호로 전이 안 됨)

### Secondary
- λ_c (congruence loss weight) sweep [0.1, 0.3, 1.0] 중 C2 best 결정
- Cross-cultural AffectNet F1 — Stage 11 humankl_lam05 의 +3pp 효과 유지 (≥ 0.33) 또는 하락

### Falsifiability
3 seed × 3 λ = 9 run, paired delta CI 산출 후 결판.

## Method

### Architecture
```
[AURegionFormerV2, Stage 6 config 그대로]
       ↓ logits, global_feat (384-d)
       ├→ classification: 기존 head → 4-class logits
       └→ congruence_head (NEW): MLP(384 → 64 → 1) → reject_logit
                                                          ↓
                                                  BCE(reject_logit, mean_is_selected)
                                                  weight = λ_c
```

### Loss
```
L = L_CE + L_focal_combined + λ_c · BCE(σ(reject_logit), mr)
```
where mr = `mean_is_selected` (이미 dataset batch 에 있음, Stage 11 humankl 와 동일 채널)

### Training
- Base config: stage6_seed999.yaml
- Modifications:
  - `congruence_head_weight: 0.3` (initial sweep value)
  - `paths.train_csv`: master_train_v_yonsei_pathfix.csv
  - `paths.val_csv`: master_val_v_yonsei_pathfix.csv
  - 30 epochs, batch 384, focal loss, AdamW
- Seeds: [999, 123, 777] (3 seeds for paired CI)
- Hardware: RTX A6000 48GB, expected 2.5h/run × 3 = 7.5h
- λ_c sweep order: [0.3] first (단일 seed 999 validation), 결과 따라 [0.1, 1.0] 추가

### Evaluation
- val 50,805 (Yonsei-evaluated 45,274 subset 에서 C2 계산)
- Stage 9 v_yonsei.py 재사용 (옛 컬럼 변형 안 함)
- AffectNet cross-eval (cross_dataset_zeroshot.py)

## Reproducibility manifest

→ `reproducibility_manifests/exp_001.yaml`

## Self-attack (Phase 후 작성)

- method_skeptic: TBD
- novelty_critic: TBD
- reviewer_simulator: TBD
