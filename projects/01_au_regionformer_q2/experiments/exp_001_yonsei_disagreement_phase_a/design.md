# exp_001 — Yonsei Disagreement Phase A (Self-Observer Affective Congruence)

## Context
- Dataset: AI Hub Korean FER (236K train+val images, ~1,082 subjects)
- Each image has: (1) self-report label by the photographed subject [4-class: happy/angry/sad/neutral], (2) external rater(s) from Yonsei 298-person consensus who answered "does this face look like {label}?" (binary).
- Master CSV: `/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired/csvs/master_{train,val}_v_yonsei.csv`
- 225,596 / 253,996 = 88.8% have ≥1 Yonsei evaluation.

## Question (RQ1)
Does external-observer rejection rate depend on the **emotion category** the subject self-reported, after controlling for subject identity?

## Stage
prototype — single-shot statistical analysis. Bootstrap iterations (1000) substitute for ML seed replicates.

## Method
1. Restrict to `yon_evaluated == 1` rows. 225,596 images.
2. Multi-rater reliability bound: subset N≥2 raters (~4,385 imgs) → unanimity %, used as a ceiling.
3. Marginal rates per emotion (mean, χ²).
4. Mixed-effect logistic GLM:
   `reject ~ C(emotion, ref='neutral')` with **cluster-robust SE on subject_hash** (GEE-style; statsmodels GLM(cov_type='cluster')) on a balanced 12K/emotion sample (48K total).
5. Within-subject paired Wilcoxon: for 948 subjects covering all 4 emotions, compare mean-reject-rate (negative={angry,sad}) vs (positive={happy,neutral}).
6. Bootstrap 95% CI of (neg − pos) within-subject delta, B=1000.

## Pre-registered hypothesis
- **primary**: rejection rate for negative emotions (angry, sad) is higher than for positive/neutral, even within the same subject. Quantitatively: odds ratio (angry vs neutral) > 1.5.
- **null**: OR ≈ 1, within-subject Δ(neg − pos) = 0.
- **success_criterion**:
  - metric: `or_angry_vs_neutral`
  - direction: `higher_is_better` (i.e. evidence of asymmetry; we *want* the asymmetry to be real because that motivates the paper)
  - delta_threshold: 0.5 (OR must exceed 1.5; OR-1 = 0.5)
- **failure_implication**: if asymmetry is small or absent, RQ1 collapses and the self-observer congruence framing is not viable. The narrative would shift back to "label noise" framing.
- **falsifiability**: PASS — predicts specific OR magnitude, falsifiable by data.

## Confounders / Limits
- Mostly N=1 rater per image (98% of evaluated set). Multi-rater unanimity ceiling (98.9%) is from a small subset (~4K).
- Yonsei raters were not blind to the self-reported label (they answered "does it look {label}?" — query-validation, not free labeling). Can inflate agreement.
- Subject heterogeneity controlled by cluster-robust SE + within-subject Wilcoxon.

## Out of scope (this exp)
- Scene effect, AU intensity × reject, model prediction agreement → exp_002+.
- Multi-task model — Phase B (separate experiment).
