---
attack_persona: method_skeptic
exp_id: exp_001_congruence_head
direction_id: AURP6-D010
date: 2026-05-23
overall_concern_level: high
---

# Top attack vectors (ranked by leverage)

1. **Wrong baseline. Stage 6 full (F1=0.9256, C2=0.114) was trained on the OLD collapse-bug master.csv.** Your new run uses `master_*_v_yonsei_pathfix.csv` with corrected `yon_*` columns and matches only 88.8% of rows (28k rows now fillna). Comparing new-run F1 against old-data F1 is apples-to-oranges: any delta could be the data fix, not the congruence head. **Change my mind**: rerun Stage 6 baseline on the *identical* `_v_yonsei_pathfix.csv` with the same fillna policy, same 3 seeds {999,123,777}, paired. Without this baseline, H1 is unfalsifiable.

2. **Missing the real competitor: Stage 11 humankl_lam05 already used `mean_is_selected` as a signal** (per STATE notes and BUGS_FOUND ref). If humankl reweighting already shifted C2 or AffectNet F1, then the explicit congruence head is *not* a novel contribution — it's the same signal in a different wrapper. **Change my mind**: include Stage 11 humankl_lam05 as a *second* baseline (also rerun on v_yonsei data) and report (Stage6, Stage11, exp_001) triple.

3. **λ_c=0.3 is unjustified — chosen by no validation.** Design says "sweep [0.1, 0.3, 1.0] after seed 999 on 0.3". This is sequential, not joint. If 0.3/seed999 looks fine you'll skip 0.1/1.0. That's selecting λ by 1 seed × 1 value = N=1 decision driving the headline. **Change my mind**: run all 3 λ × 3 seeds = 9 runs *before* declaring a primary λ, or pre-register that the *median* λ across seeds is the reported config.

4. **n_seeds=3 is below the v2 paper_ready bar of ≥7.** Paired CI with n=3 has effective df=2; Wilcoxon at n=3 cannot reach p<0.05 (min two-sided p = 0.25). H1's "C2 ≥ 0.15 vs 0.114" cannot be tested with adequate power. **Change my mind**: bump to 7 seeds for the chosen λ before any LOGGING; the 3-seed sweep is exploratory only.

5. **C2 per-image Pearson is computed on the 45,274 Yonsei-evaluated *val* subset — but the congruence head is trained on `mean_is_selected` for the 88.8% of *train* rows that have Yonsei coverage.** Val rows without Yonsei evaluation are silently dropped from C2 but kept in F1. So the F1 baseline and the C2 baseline live on different row sets. Plus the 28,315 fillna(0) train rows inject "rejected=0" bias into the head — it learns "no evaluation == perfect congruence", not actual congruence. **Change my mind**: (a) restrict L_congruence to rows where `yon_evaluated==1` (masked BCE), (b) report F1 on the same 45,274 val subset that C2 uses, paired.

6. **Hyperparameters inherited from Stage 6 are not tuned for multi-task.** base_lr=0.0008, batch=384, bf16, cudnn.benchmark=true, tf32=true, torch_deterministic=false. Adding a new head with a new loss term changes the gradient norm landscape; reusing the single-task lr is the *weakest reasonable* multi-task setup. Also `torch_deterministic=false` + benchmark=true means seed differences include kernel-selection noise, inflating between-seed variance and hiding real effects.

7. **Nothing reported about the congruence head's own accuracy.** If the head reaches BCE plateau early (say ROC-AUC on mr ≥ 0.8) but C2 doesn't move, the head learned the target without coupling to the backbone — i.e., a gradient bypass. If BCE never converges, λ_c=0.3 is too small. Either way, head-internal diagnostics decide interpretation, and they're not in the design. **Change my mind**: add `reject_logit` ROC-AUC and calibration to results.json.

8. **No train_loss vs val_loss gap reporting.** Multi-task with an auxiliary head can either regularize (good) or overfit on the auxiliary signal (bad). Without train/val curves for both heads, you cannot tell "C2 went up because backbone learned alignment" from "C2 went up because val happens to have the same fillna pattern as train".

# The one experiment that would settle it

Run a **3×3 factorial** in one shot, paired across {seeds 999, 123, 777}:
- factor A: model ∈ {Stage 6 backbone, Stage 11 humankl_lam05, exp_001 congruence head}
- factor B: λ_c ∈ {0.0 sham-head, 0.3, 1.0} (sham-head = same params, weight 0 — controls for extra capacity)
- all on `master_*_v_yonsei_pathfix.csv` with `yon_evaluated==1` mask for C2 and BCE.

Report paired ΔF1 and ΔC2 with 95% CI + Cohen's d. If exp_001 with λ_c>0 beats *both* Stage 11 humankl *and* sham-head on C2 without F1 regression, the claim survives. Anything less, it doesn't.

# Configurations the author should have tried but didn't

- **Masked BCE** (`yon_evaluated==1` only) — current design contaminates the head with 28k synthetic "0"s, biasing toward predicting low reject_rate for unevaluated images.
- **Detached congruence head** (gradient stop on global_feat) — separates "does the head predict mr?" from "does the gradient help the backbone?" If detached head reaches same C2, the multi-task framing is decorative.
- **MSE on `yon_reject_rate` (continuous) instead of BCE on majority** — `mean_is_selected` is a rate in [0,1] with multi-rater info; BCE on it as a Bernoulli probability is a mismatched likelihood for n_raters>1.
- **Subject-stratified val** — subject-wise split is claimed but not verified for the new v_yonsei csvs after the path fix; one re-check of subject_hash overlap = 0 should be in the manifest.
- **Joint vs sequential λ sweep** — current plan is sequential (run 0.3 first, then decide). Pre-register joint or commit to median-λ reporting.
- **Stage 6 rerun on v_yonsei** — without this the baseline number 0.9256/0.114 is from a different data construction.
- **n_seeds=7 power check** — with paired CI and expected ΔC2=0.04, σ_C2 from Stage 9 (0.114 → 0.109 across reruns ≈ 0.003), you actually only need ~3 seeds for *that* effect — but you don't know σ for the new data + new head. Pilot σ first.

# Bottom line

The H1 falsification criterion ("F1 ≥ 0.9251 AND C2 ≥ 0.15") is structurally untestable until (a) the baseline is rerun on the same data, (b) the head is masked to evaluated rows, and (c) n_seeds reaches the paper_ready bar. As written, a positive result is consistent with "data fix lifted C2, head did nothing". Advisory: hold LOGGING until items 1, 2, 5, and 4 above are addressed.
