---
attack_persona: novelty_critic
exp_id: exp_001_congruence_head
claim_id: C1_per_image_alignment_recoverable
severity: high
---

# Citations to address

- **Peterson, Battleday, Griffiths & Russakovsky 2019**, *Human Uncertainty Makes Classification More Robust* (ICCV; CIFAR-10H) — trains directly on per-image human soft labels (the same `mean_is_selected`-style signal we use) and shows robustness/generalization gains; our BCE on `mean_is_selected` is a binarized special case of this distribution-matching objective.
- **Hinton, Vinyals & Dean 2015**, *Distilling the Knowledge in a Neural Network* — establishes auxiliary soft-target supervision as a generic recipe; our λ_c · BCE(σ(W·g), mr) is structurally a soft-target auxiliary loss in their framework, with mr playing the role of the teacher distribution.
- **She, Hu et al. 2021 (DMUE, CVPR)**, *Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for FER* — adds auxiliary branches that mine per-sample label distribution / ambiguity from annotator disagreement on the *same task* (FER); directly overlaps "per-image annotator-ambiguity signal as auxiliary head on FER."
- **Liu et al. 2023 (MTAC, IEEE TMM)**, *Uncertain FER via Multi-task Assisted Correction* — multi-task auxiliary heads + per-sample confidence estimation specifically to handle annotator-ambiguous FER samples; same problem, same MTL recipe.
- **Fornaciari et al. 2021 (NAACL)**, *Beyond Black & White: Leveraging Annotator Disagreement via Soft-Label Multi-Task Learning* — explicit auxiliary task predicting annotator soft-label distribution alongside the main classifier; identical architectural pattern (shared backbone → main head + soft-label head), different domain (NLP).
- **Liu et al. 2023 (LA-Net) / Zhang et al. 2023 (ReSup)** — predict per-image label-noise/reliability as auxiliary signal in FER; close to "predict reject_rate as auxiliary scalar."
- **Internal: Stage 11 humankl_lam05** — our own prior baseline already consumes `mean_is_selected` via KL distillation and obtained AffectNet +3pp. The "novel" head re-uses the *same input channel* with a different loss (BCE vs KL on the same target).

# Where our novelty is weakest

The core idea — *predict per-image annotator agreement/soft-distribution as an auxiliary head over a shared backbone* — is published (Peterson 2019; DMUE 2021; MTAC 2023; Fornaciari 2021), and we ourselves already exploit the identical `mean_is_selected` signal via KL in Stage 11. Swapping KL→BCE on a scalar `mr` is a loss-function variant of an existing baseline, not a new contribution. The framing "per-image perceptual congruence prediction" is a relabeling of "predict human (dis)agreement," which reviewers will recognize immediately.

# Required additions to defend

- **baseline_to_add**: Stage 11 humankl_lam05 (KL on full distribution) *and* a Peterson-style direct soft-label CE — both must appear in the same table. Expected C2 Pearson: 0.10–0.18 (overlapping our target).
- **baseline_to_add**: DMUE-style auxiliary branch on our backbone (or cite as out-of-scope with justification). Expected emotion F1: ±0.5pp of Stage 6.
- **ablation_to_run**: BCE-on-scalar vs KL-on-distribution vs soft-CE on the *same* `mr` channel, 3 seeds — to show the *choice* of loss (not the auxiliary-head idea) is what moves C2. Reason: reviewers will ask "is this just Hinton/Peterson with binarization?"
- **ablation_to_run**: congruence-head with `mr` shuffled across batch (label-permutation control). Reason: rules out that the head is learning generic regularization rather than per-image congruence; without this, reviewers reject the per-image claim.
- **reframing required**: the contribution must be repositioned as either (a) a *region-conditioned* congruence prediction (using AU-region tokens, not just `g_feat`) which is genuinely new vs DMUE/Peterson, or (b) an empirical study of *which* annotator-disagreement loss best transfers to per-image C2 under AU-region architectures. As written, "novel multi-task contribution" will not survive R2.
