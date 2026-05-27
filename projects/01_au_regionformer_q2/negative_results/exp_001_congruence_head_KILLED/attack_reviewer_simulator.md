---
attack_persona: reviewer_simulator
exp_id: exp_001_congruence_head
claim_id: C1_per_image_alignment_recoverable
venue: TAFFC (realistic) / NHB-PNAS (stretch — rejected outright)
estimated_review_score: 3.5
summary_one_line: "Reject. Engineering tweak masquerading as a thesis; the authors' own ablation refutes the interpretability claim."
---

# Strong points (≤3)
- Pre-registered hypothesis with explicit delta thresholds (F1 -0.005, C2 +0.04) and a null definition — rare in FER literature, mildly increases credibility.
- The 8-region zero-mask ablation is genuinely interesting and the authors deserve credit for reporting a null (Δ acc ±0.0001) rather than burying it.
- Per-image perceptual alignment metric (C2 Pearson against `mean_is_selected`) is a reasonable proxy and the authors define it transparently.

# Weak points (by severity)

## High
- **The headline ablation contradicts the central interpretability framing.** Authors report that zero-masking all 8 region patches changes accuracy by ±0.0001 at F1 = 0.9256. This means the "region patches" are decorative — the backbone alone carries the signal. Adding a congruence head on a global 384-d feature does not rescue the regional interpretability claim; it abandons it. The paper either needs to be reframed as "global-feature multitask learning for perceptual congruence" (and drop the RegionFormer interpretability narrative entirely) or the authors must show the head's gradient flows through the region tokens. *Address by: ablating congruence head with regions zeroed — if C2 still improves, the head is a global-feature trick, not a region story.*
- **C2 target of +0.04 Pearson (0.114 → 0.15) is not justified as meaningful.** What is the noise floor? Bootstrap CI of the baseline 0.114? Inter-annotator Pearson upper bound on `mean_is_selected`? Without these, +0.04 is an arbitrary engineering target, and reviewers will assume it was chosen because preliminary runs hit ~0.15. *Address by: report annotator-annotator Pearson ceiling and 95% bootstrap CI of baseline 0.114 BEFORE running the experiment.*
- **n=3 seeds for paired CI on a 0.5pp F1 effect is statistically inadequate.** Stage 6 baseline std is unreported in design.md. With 3 seeds and F1 std plausibly ~0.003-0.005, the paired CI on a 0.005 regression threshold is essentially uninformative. NeurIPS/TAFFC reviewers in 2026 expect n≥5, ideally n≥7 for paper_ready claims. The engine's own METHODOLOGY mandates this. *Address by: 7 seeds × 3 λ, or drop the F1-preservation claim and reframe as C2-only.*
- **"Self-report ≠ self-validation" thesis is not in this design.md.** If this is supposed to be the Q1 thesis (per the prompt), there is no operationalization here. The design tests "can we add a head to lift per-image C2 by +0.04". That is a method paper section, not a thesis. The thesis as stated belongs to social psychology; the experiment is an architectural ablation. *Address by: write the falsifiable form of the thesis in claim_registry, separate from the engineering H1.*

## Medium
- **Structure A (single multi-task head) does not warrant a paper.** This is one section (§3.4 or §4.x) in a larger study. Reviewers will ask why this is not folded into the Q2 paper. With Stage 11 humankl already delivering +3pp cross-cultural transfer, adding a congruence head is incremental engineering, not a contribution worth a venue slot.
- **Stage 11 humankl already exploits `mean_is_selected` via KL on attention.** Structure A uses BCE on a global head. What is the conceptual delta? The design does not compare against Stage 11 as the baseline — it compares against Stage 6 (no humankl). The relevant ablation is `humankl vs congruence_head vs both`, not `vanilla vs congruence_head`. Without this, the +3pp transfer effect is double-counted. *Address by: include Stage 11 humankl_lam05 as the second baseline column in every table.*
- **F1 = 0.9256 on the Korean 4-emotion subset is not a strong external baseline.** With class imbalance and Yonsei consensus filtering, this number is not comparable to public FER benchmarks. Korean FER ceiling at ~93% means the F1-preservation constraint has almost no headroom to be informative.
- **λ_c sweep [0.1, 0.3, 1.0] with selection on val C2 then reporting val C2 is a textbook selection-bias setup.** Need held-out test or nested CV. *Address by: λ_c selection on 80% train-subset, evaluation on held-out 20% per fold.*

## Low
- AffectNet "cross-cultural" framing is shaky — AffectNet is web-scraped, not annotated as Western. Use a properly Western-annotated set (RAF-DB, FER+) or drop the cross-cultural framing.
- "Stage 6 config 그대로" — what config? Reviewers cannot reproduce without manifest content quoted in design.
- Reproducibility manifest is referenced but not present in design — confirm it exists before submission.

# Most likely sentence in the review that hurts

"The authors' own ablation demonstrates that zero-masking the eight region patches changes accuracy by 0.0001 — this is a confession, not a contribution, and adding a single MLP head on the global feature does not transform a refuted interpretability claim into a publishable one; it merely obscures it."

# Recommended pre-submission additions

1. Run the *crossed* ablation: `{regions_on, regions_zeroed} × {congruence_head_on, off}` — 4 cells, 3 seeds each. This is the only experiment that distinguishes "regional alignment" from "global feature multitask".
2. Add Stage 11 humankl_lam05 as a baseline column in every table, not just as a secondary comparison.
3. Compute annotator-annotator Pearson ceiling on `mean_is_selected` and report 95% bootstrap CI of the 0.114 baseline. Justify +0.04 as meaningful relative to this ceiling, or revise threshold.
4. Increase seeds to ≥7 for paper_ready stage (METHODOLOGY §Gate C). 3 seeds will be rejected at desk-check.
5. Reframe contribution: either (a) drop the RegionFormer interpretability narrative and call this a global-feature perceptual-alignment regularizer, or (b) show the head's gradient routes through region tokens with grad-flow visualization.
6. If pitching to NHB/PNAS: the thesis "self-report ≠ self-validation" must be operationalized as a psychological claim with human subject preregistration, not an architecture ablation. Otherwise, target ICMI/ACII/Sensors honestly.
