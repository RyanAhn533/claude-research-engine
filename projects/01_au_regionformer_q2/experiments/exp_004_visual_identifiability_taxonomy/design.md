# exp_004 — Visual Identifiability Taxonomy (V1–V4)

## Context
- Foundation of the affective-grounding paper (see `projects/02_affective_grounding/MASTER_PLAN.md` §1, RQ4 prep).
- exp_003 showed model accuracy = 94.3% on observer-agree vs 77.3% on observer-reject (partial ceiling). This experiment **decomposes** those errors into a 4-quadrant taxonomy so we can say *which kind* of error dominates.

## Question (RQ4-prep)
Split every evaluated val sample by (observer verdict) × (model correctness):

| group | observer | model | meaning |
|-------|----------|-------|---------|
| V1 | agree  | correct | visually identifiable self-report |
| V2 | agree  | wrong   | pure model error |
| V3 | reject | correct | observer missed it but model recovered self-report |
| V4 | reject | wrong   | visually underdetermined self-report |

Key reads:
- **V2 large** → model is the weak point.
- **V4 large** → label not visible in the face (data limit).
- **V3 large** → observer rejection ≠ label invalidity (model sides with self-report where a single observer did not).

## Method (no GPU — reuses saved predictions)
- For each of 3 baseline seeds: join `paper_artifacts/predictions.csv` ⟕ `yon_reject_rate` (basename join, as exp_003).
- A=agree (reject_rate=0, evaluated), B=reject (reject_rate≥0.5); C=unevaluated reported separately, not in taxonomy.
- Compute, overall + per class (angry/happy/neutral/sad), averaged across seeds:
  - counts and P(Vk) = N(Vk)/N
  - ErrorShare_reject = N(V4) / (N(V2)+N(V4))
  - ModelCorrectGivenReject = P(ŝ=s | reject) = V3/(V3+V4)
  - ModelWrongGivenAgree = P(ŝ≠s | agree) = V2/(V1+V2)

## Pre-registered hypothesis (H_taxonomy_v3_substantial)
- **primary**: On observer-reject samples, the model still recovers the self-report label in a substantial fraction — `ModelCorrectGivenReject = V3/(V3+V4) ≥ 0.50` (averaged over 3 seeds). This operationalizes "observer rejection ≠ self-report invalidity": even where a single external rater rejected the label, the model agrees with self-report more often than not.
- **null**: model recovers < 50% on reject samples (V4 dominates), i.e. rejection mostly marks truly invisible self-report.
- **success_criterion**: `model_correct_given_reject`, `higher_is_better`, `delta_threshold = 0.50` (threshold-as-level).
- **falsifiability**: PASS — a value < 0.50 refutes it.
- consistency check: must agree with exp_003 (acc on B ≈ 0.773 ⇒ expect V3/(V3+V4) ≈ 0.773).

## Logging
- results.json (V1–V4 counts/shares overall + per class, the three derived metrics).
- leaderboard.jsonl row (method_id = "v2_id_taxonomy", stage prototype).
- hypothesis_registry outcome row.
