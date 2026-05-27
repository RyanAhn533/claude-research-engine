# exp_003 — Is val F1 ≈ 0.926 the annotation-agreement ceiling?

## Context
- exp_001: self-observer congruence asymmetry is real (OR_negative ≈ 2.7, p≈1e-117). Negative emotions get rejected by external observers ~3× more.
- exp_002: using that signal via Human-KL distillation (λ=0.1) gives **no model-level benefit** (paired Δ=+0.013pp, CI includes 0, verdict=neutral). Per-class check (incl. angry/sad) = noise.
- Open question before chasing more model tricks: **is the model already at the ceiling set by annotation (dis)agreement?** If yes, no integration trick can move macro F1, and the paper's contribution is descriptive (discovery of the asymmetry + the ceiling), not a model improvement.

## Question (RQ3)
Conditional on the external rater's verdict, how does the model behave?
- A = samples the single external rater **agreed** with the self-label (yon_reject_rate = 0, evaluated).
- B = samples the external rater **rejected** (yon_reject_rate ≈ 1).
- C = samples with **no** external evaluation (yon_n_evals = 0).

If 0.926 reflects an annotation-agreement ceiling, an appearance-based model trained on self-labels should be near-perfect on A and collapse on B (the face does not show the self-emotion, so the model — reading the face — predicts the rater's view, scoring "wrong" against the self-label).

## Data facts (verified)
- val = 50,805 samples. `yon_n_evals`: 0 → 5,531 (10.9%); 1 → 44,430 (87.5%); 2 → 832; 3 → 12.
- **Single external rater for 98% of evaluated samples** → cannot compute true inter-rater agreement; reject_rate is mostly binary (0 or 1).
- mean(1 − yon_reject_rate) over all val = 0.913 (= naive single-rater agreement = naive ceiling estimate).
- Group sizes (evaluated): A (reject=0) ≈ 40,852; B (reject≈1) ≈ 4,422; C (unevaluated) = 5,531.
- Model val_acc ≈ 0.9258, best_f1 ≈ 0.9252.

## Method (no GPU — reuses saved predictions)
- Each of 6 runs already wrote `paper_artifacts/predictions.csv` (path, true_label, pred_label, correct, prob_*).
- Join predictions ⟕ `master_val_v_yonsei.csv` on `path` → attach yon_reject_rate, yon_n_evals.
- Compute model accuracy in A / B / C, overall + per class (angry/happy/neutral/sad).
- Primary model = 3 baseline seeds (KL OFF); report mean ± across seeds. Treatment reported as secondary.

## Pre-registered hypothesis (H_ceiling)
- **primary**: model accuracy on rejected samples (B) is far below agreed samples (A): `acc(A) − acc(B) ≥ 0.30` (30pp), with `acc(A) ≥ 0.95`.
- **interpretation if SUPPORTED**: the model already extracts essentially all appearance-consistent signal; the residual error lives where humans themselves reject the self-label. 0.926 ≈ ceiling. Paper = descriptive (asymmetry + ceiling); stop chasing model tricks on macro F1.
- **interpretation if REJECTED** (acc(B) high, e.g. acc(A) − acc(B) < 0.15): the model is **not** annotation-ceiling-limited → real headroom on rejected samples → the congruence signal is exploitable by a better integration than KL (multi-task head / reweighting). Reopens the model angle.
- **null**: no meaningful A vs B gap.
- **success_criterion**: `acc_gap_A_minus_B`, `higher_is_better` (for the ceiling claim), `delta_threshold = 0.30`.
- **falsifiability**: PASS — a high acc(B) directly refutes the ceiling claim.

## Gate
- Descriptive analysis over fixed saved predictions; no training stochasticity beyond the 3 already-run seeds.
- Report A/B/C accuracy per seed → mean, and the across-seed range to show stability.
- Caveat (must appear in results + any paper text): single-rater labels for 98% of evaluated samples; "ceiling" = single-external-rater agreement, not multi-rater Bayes ceiling.

## Risks / threats to validity
- Single rater → reject_rate is noisy per sample; mitigated by large B (n≈4,400) so the aggregate acc(B) is well estimated.
- Path-join mismatch (encoding / absolute paths). Must verify join coverage ≥ 99% before trusting numbers.
- C (unevaluated) is not random — could be a distinct subpopulation; report separately, do not fold into A.

## Logging plan
- `results.json` (A/B/C acc overall+per-class, gap, verdict).
- leaderboard.jsonl row (method_id = "v2_ceiling_probe", stage prototype).
- hypothesis_registry.jsonl outcome row superseding the pre-reg.
