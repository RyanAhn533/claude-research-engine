# exp_007 — DROZY drowsiness uncertainty probe (Track B survival gate, pivot)

## Context
- exp_006 killed PPB-Emo emotion as Track B substrate (estimator at chance, uncertainty AUROC ≈ 0.5 even on easier binary targets).
- Pivot per STRATEGY_unified_carla_affect.md + GPT note: drowsiness/distraction is more predictable AND more safety-causal than emotion. DROZY is the canonical drowsiness EEG dataset.
- Same gate as exp_006 — but substrate change to drowsiness. If THIS uncertainty isn't informative either, Track B is dead regardless of substrate; if it is, Track B's CARLA gating has a real foundation.

## Data
- `/home/ajy/04_driver_monitoring/03_DMS_sleep_classification/251219/data/sessions/*.npz` — 36 sessions, 14 subjects (each subject ~2-3 sessions).
- Each session npz: `X` (600 windows × 512 samples × 5 EEG channels), `y` session-level 3-class drowsiness (0=alert / 1=moderate / 2=sleepy, derived from KSS), `kss`, `session`.
- Total ≈ 21,600 windows. Labels per-window inherited from session.

## Why this matters
- Existing pre-trained `drozy_eeg_tcnformer_v1` reports session_acc 0.571 (3-class) — but its split is SESSION-level, leaking subject identity (same subject in train+val). This experiment uses **subject-wise GroupKFold** → strictly fair generalization.

## Method
- **Subject-wise GroupKFold(n_splits=5)** on subject ID. No subject in both sides.
- Small 1D-CNN ensemble (5 members, dropout 0.5, weight decay) per fold → per-window softmax; ensemble mean → predictive entropy = uncertainty.
- 3 seeds {42,123,777} → 3 independent estimates (n_seeds=3).
- Window-level out-of-fold predictions for all ~21.6k windows; also session-level via mean prob.

## Metrics
- **Window-level**: balanced acc (vs chance 1/3, majority baseline), ECE, NLL, multiclass Brier; bootstrap 95% CI (B=2000).
- **Session-level**: balanced acc (mean-prob vote per session, ~36 sessions).
- **Primary gate**: window-level `error-vs-uncertainty AUROC` (roc_auc(misclassified, entropy)) ≥ 0.60 AND bootstrap CI lower > 0.50.
- Risk-coverage: window-level selective bal_acc at coverage 80%, 50%.

## Pre-registered hypothesis (H_drowsy_uncertainty_informative)
- **primary**: predictive uncertainty on DROZY drowsiness (subject-wise) is informative — `error_vs_uncertainty_auroc ≥ 0.60` AND bootstrap CI_lo > 0.50.
- **null**: AUROC ≈ 0.5 (uninformative).
- **success_criterion**: `error_vs_uncertainty_auroc`, `higher_is_better`, `delta_threshold = 0.60`.
- **falsifiability**: PASS.
- **interpretation**: SUPPORTED → Track B substrate confirmed (drowsiness), build CARLA gating next. REJECTED → no usable uncertainty on either substrate → Track B fundamentally not viable on these data; deprioritize.

## Caveats
- Window-level samples are not iid (windows from same session correlated) — bootstrap CI is over-optimistic; report session-level too.
- 14 subjects is small; 5-fold = ~3 subjects per val fold; subject-level variance large.

## Logging
- results.json; leaderboard row (method_id="drozy_uncertainty_probe"); hypothesis outcome.
