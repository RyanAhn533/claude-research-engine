# exp_006 — PPB-Emo driver-state uncertainty probe (Track B survival gate)

## Context
- 2-track decision (STRATEGY_unified_carla_affect.md): Track B = driver-state uncertainty-gating intervention. Track A capped (single-rater, raters unavailable).
- Before building any CARLA gating, the load-bearing prerequisite: **is predictive uncertainty on PPB-Emo even informative?** If a driver-state estimator's uncertainty does NOT predict its own errors, then uncertainty-gating in CARLA is pointless and Track B's premise is dead.

## Question
Train a driver-emotion estimator on PPB-Emo tabular features (EEG aggregates + driving behaviour) with proper subject-wise CV; does its predictive uncertainty separate correct from wrong predictions, and does abstaining on high-uncertainty samples improve accuracy?

## Data
- `/home/ajy/04_driver_monitoring/02_DMS_PPBEMO_driving_behavior/PPB-EMO_codes/ppb-emo/features_all.csv` — 240 rows, 40 participants.
- Features: EEG aggregates (644) + driving-behaviour DBD (28) = 672 numeric. (Face/body/road are video paths — excluded in this probe.)
- Target: `category` 7-class (AD/DD/FD/HD/ND/SAD/SD). Secondary (reported, not gated): `valence` binarized at median.
- Median-impute NaN + standardize (fit on train folds only).

## Method
- **Subject-wise** `GroupKFold(n_splits=5)` on `participant` → out-of-fold (OOF) predictions for all 240 (no participant leakage).
- **Deep ensemble** of 5 small MLPs (1 hidden layer, dropout, weight decay) per fold → predictive distribution = mean of softmaxes; **uncertainty = predictive entropy**.
- Logistic-regression (L2) baseline for sanity.
- Repeat whole pipeline with 3 seeds {42,123,777} (ensemble init + fold shuffle) → 3 independent estimates (n_seeds=3).

## Metrics
- balanced accuracy (vs chance 1/7≈0.143 and majority baseline), per seed + mean.
- Calibration: ECE (15-bin), NLL (log_loss), multiclass Brier — with **bootstrap 95% CI** (B=2000) on pooled OOF.
- **Primary (decisive): error-vs-uncertainty AUROC** = roc_auc(y=misclassified, score=entropy). >0.5 means uncertainty predicts errors.
- Risk–coverage: selective balanced-accuracy at 80% / 50% coverage (keep most-confident) vs full.

## Pre-registered hypothesis (H_uncertainty_informative)
- **primary**: predictive uncertainty is informative about errors — `error_vs_uncertainty_auroc >= 0.60` AND its bootstrap 95% CI lower bound > 0.50.
- **null**: AUROC ≈ 0.5 (uncertainty uninformative; CI includes 0.5).
- **success_criterion**: `error_vs_uncertainty_auroc`, `higher_is_better`, `delta_threshold = 0.60`.
- **falsifiability**: PASS.
- **interpretation if SUPPORTED**: uncertainty-gating has a basis → build Track B CARLA gating + spin up dedicated project.
- **interpretation if REJECTED**: uncertainty is garbage on this small data → Track B premise weak; revisit (more data / different signal) or shift weight to Track A.

## Caveats (must report)
- 240 samples / 40 participants is small; EEG features are pre-aggregated stats (not raw). 672 features ≫ 240 samples → overfit risk, hence heavy regularization + subject-wise CV + bootstrap CI.
- A pass here is necessary-not-sufficient for Track B (CARLA closed-loop gating is the real test).

## Logging
- results.json; leaderboard row (method_id="ppbemo_uncertainty_probe", stage prototype, n_seeds=3); hypothesis outcome.
