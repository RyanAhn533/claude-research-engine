# exp_002 — Phase B: Human-KL ablation on master_v_yonsei

## Context
- Follows exp_001 (statistical asymmetry of self-observer congruence: OR_negative ≈ 2.7, p≈1e-117).
- The narrative requires that this disagreement signal is **useful for the model**, not just descriptively interesting.
- AU-RegionFormer's `trainer.py` already implements a Human-KL distillation loss (Stage 11). Phase 6's exploration of λ ∈ {0.001..0.010} on the buggy old `master.csv` showed essentially no effect (±0.02pp). This experiment retests with: (1) the correct emotion-aware `master_v_yonsei.csv`, (2) a much larger λ (0.1 — 10×–100× prior).

## Question (RQ2)
Does adding a Human-KL distillation term (target = soft distribution derived from external rater agreement) **improve val_f1_macro** over a strong baseline of the same architecture trained on the same data?

## Stage
prototype. 3 seeds per condition (Gate C prototype minimum).

## Method
1. Same architecture (`stage6_full`): AU-RegionFormer v2 with Stage A self-attn + DualBeta + Beta-anchored aux + heavy regularization (mixup, EMA, distill_alpha, drop_path, label_smoothing).
2. Same data: `master_train_v_yonsei.csv` (203,191) + `master_val_v_yonsei.csv` (50,805). Subject-wise split.
3. Two conditions:
   - **baseline** (KL OFF): `human_kl_weight = 0.0`
   - **treatment** (KL ON): `human_kl_weight = 0.1`
4. 3 seeds each: 42, 123, 777. Total 6 runs, ~2.5h each.
5. Per-seed: val_f1_macro_best. Aggregate: paired delta (treatment − baseline) per seed → mean, bootstrap 95% CI, Cohen's d_paired, Wilcoxon signed-rank.

## Bug-fix prerequisite (already applied)
`src/data/dataset_v2.py` previously treated `yon_reject_rate` as `mean_is_selected` directly. Fixed to `mean_is_selected = 1 − yon_reject_rate`. Without this fix, Stage 11 KL distillation would push the model in the opposite direction.

## Pre-registered hypothesis
- **primary**: treatment improves val_f1_macro by at least 0.3pp over baseline (paired delta CI lower bound > 0.003).
- **null**: no improvement; paired delta ≈ 0 within ±0.3pp.
- **success_criterion**: `paired_delta_val_f1_macro`, `higher_is_better`, `delta_threshold = 0.003`.
- **failure_implication**: even with correct labels and 100× λ, Human-KL distillation gives no model-level benefit. Story shifts from "useful supervision signal" to "interesting descriptive finding only" — would need a different model integration (multi-task head, calibration, etc.) to salvage the narrative.
- **falsifiability**: PASS.

## Gate C (prototype)
- n_seeds = 3 per condition.
- Required: mean, std per condition + paired delta with 95% bootstrap CI + Cohen's d_paired + Wilcoxon p.
- Verdict: improvement_candidate iff (CI_lower > 0 AND |delta| ≥ 0.003 AND Wilcoxon p < 0.05).

## Risks
- Single λ (0.1) chosen. If too aggressive, model overfits to congruence signal at cost of emotion classification; if too small, no signal. We're betting that 10–30× larger λ than prior is the right band given the loss-scale.
- Sequential GPU access (CARLA + sc user co-located). OOM risk low (~12GB needed vs 42GB free).
- Same architecture as Phase 6 baseline; if architecture has already saturated on this data (~F1 0.926 ceiling), even a working signal won't show up. Mitigation: report effect on per-class F1 (esp. angry/sad), not just macro.

## Logging plan
- Each run, after training completes, `log_one_run.py` appends:
  - `leaderboard.jsonl` row (one per seed × condition = 6 rows total).
  - `paper_tried.jsonl` row.
  - Per-run reproducibility manifest (yaml).
- After all 6 runs: aggregate analysis (paired stats) → final hypothesis outcome row supersedes the pre-reg row.
