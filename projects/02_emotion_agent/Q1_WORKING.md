# Q1 Paper Working Draft — Emotion Agent

## Status (2026-04-23, end of Week 3)
- Phase 0 setup ✓
- Week 1 preprocessing ✓ (IEMOCAP/MELD/K-EmoCon)
- Week 2 agent baseline eval ✓ (IEMOCAP/MELD/K-EmoCon/FER)
- **Week 3 few-shot scaling + multi-seed + LoRA ✓ — paper-grade findings**
- Week 4 writing + SOTA baseline + cross-domain: planned

## Title (draft)

*"From Prompt to Parameter: A Tiered Study of Cultural Grounding in
Facial Emotion Agents"*

alt: *"Target-Culture Exemplars, Diverse Anchors, and LoRA: Decomposing the
Contributions of Cultural Grounding in LLM Emotion Classifiers"*

## Abstract draft (multi-seed + LoRA)

We present a three-tier study of cultural grounding in LLM-based facial emotion
classifiers, using Korean FER action-unit data (229K images, 4-class) with
Qwen2.5-7B-Instruct. Each tier is evaluated under multi-seed (n=3) protocol
with statistical reporting.

Three findings:

**(1) Adaptation hierarchy is cumulative across tiers.** A zero-shot FACS prompt
reaches 29.08 ± 0.76 %; adding k=4 Korean exemplars raises this to
41.83 ± 3.41 % (+12.75 pp); applying QLoRA (r=16, 10K training samples,
1 epoch) reaches 55.00 ± 2.61 % (+13.17 pp). Tiers are statistically
separated under 1σ.

**(2) The role of exemplars decomposes into two independent effects.** Across a
five-point Korean-to-Western exemplar ratio sweep (k=4 total), we observe:
(i) the **mean** depends on the presence of ≥1 target-culture (Korean)
exemplar (single-Korean setup reaches 40 %, no Korean stays at 25 %);
(ii) the **variance** collapses from σ≈4-6 pp to σ<1 pp once at least two
Western (non-semantic, idealized-prototype) anchors are included.
Mean and variance are driven by different exemplar types.

**(3) ICL and LoRA are substitutes, not complements.** Adding Korean exemplars
on top of a Korean-trained LoRA adapter hurts accuracy by 0.75-4.00 pp —
adaptation modes overlap in information, and presenting the same
distribution twice causes attention-split regression. Mixed (anchored)
exemplars hurt *less* than pure target-culture ones, consistent with the
anchor-regularization story in (2).

Idealized Ekman FACS prototypes — the textbook contribution of Western
FACS literature — perform at random (25.42 ± 0.38 %) on Korean faces,
regardless of seed. This negative result motivates finding (2)'s
decomposition.

These results challenge the common practice of injecting cultural metadata
via system prompts and provide a concrete recipe for culturally-situated
LLM agents: use at least one target-culture exemplar for mean accuracy,
at least two diverse anchors for variance stability, and prefer LoRA over
ICL when training data is available (do not stack both).

## Headline Results Table (multi-seed robust)

### Main: 3-tier hierarchy on Korean FER AU 4-class (n=3 seeds, N=400 each)

| Tier | Method | Acc | Macro-F1 | Δ vs prev |
|------|--------|-----|----------|-----------|
| 1 | Zero-shot FACS prompt | 29.08 ± 0.76 | 0.194 ± 0.011 | — |
| 2 | ICL k=4 Korean exemplars | 41.83 ± 3.41 | 0.364 ± 0.040 | **+12.75** |
| **3** | **QLoRA r=16, 10K, 1 ep** | **55.00 ± 2.61** | **0.555 ± 0.026** | **+13.17** |

Total Δ (tier 1 → tier 3) = **+25.92 pp**.

### Anchor-ratio ablation (k=4 total, n=3 seeds)

| k_Kor + k_Wes | Acc ± σ | σ regime |
|---------------|---------|----------|
| 4 + 0 (pure Korean) | 41.00 ± 4.34 | HIGH |
| 3 + 1 | 41.50 ± 6.00 | HIGH |
| **2 + 2 (Mixed)** | **41.25 ± 0.87** | LOW |
| **1 + 3 (Wes-heavy)** | **40.08 ± 0.80** | LOW |
| 0 + 4 (Ekman only) | 25.42 ± 0.38 | (low mean) |

### ICL on LoRA substitutability (seed=42, N=400)

| Config | Acc | Δ vs LoRA-only |
|--------|-----|---------------|
| LoRA only | 56.75 | — |
| LoRA + Mixed k=4 (2+2) | 56.00 | −0.75 |
| LoRA + Korean k=4 | 52.75 | **−4.00** |

### k-scaling (revised — saturation, not inverted-U)

| k | Acc ± σ |
|---|---------|
| 0 | 29.08 ± 0.76 |
| 2 | 44.25 ± 2.82 |
| 4 | 42.75 ± 5.02 |
| 8 | 43.50 ± 1.52 |
| 16 | 42.33 ± 1.84 |

## Key figures (TBD — plot generation)

1. **Figure 1**: 3-tier hierarchy bar chart with error bars (prompt vs ICL vs LoRA).
2. **Figure 2**: Anchor-ratio variance curve (k_Wes on x, acc±σ, showing threshold at k_Wes=2).
3. **Figure 3**: k-scaling with saturation + note on prior single-seed artifact.
4. **Figure 4**: LoRA+ICL stacking plot, showing substitutability regression.

## §3 Method outline

### 3.1 Agent architecture
Base: Qwen2.5-7B-Instruct, 4-bit NF4 (5.4 GB VRAM inference).
Prompt format: system (task + FACS prototypes + optional exemplars) + user (AU intensities).
Output parsing: regex on `LABEL:` and `REASON:`.

### 3.2 Cultural grounding variants (tier 2 exemplar design)
(A) zero-shot FACS baseline
(B) Korean real exemplars (from OpenGraphAU 229K Yonsei-clean Korean FER)
(C) Western Ekman FACS-textbook prototypes (idealized AU patterns from FACS manual)
(D) Mixed = k_Kor Korean + k_Wes Western, varying (k_Kor, k_Wes) ratios

### 3.3 LoRA fine-tune (tier 3)
QLoRA r=16, α=32, dropout=0.05, target = all linear layers (q,k,v,o,gate,up,down).
Training: 10,000 balanced samples from Korean FER AU, 1 epoch,
effective batch=16, lr=2e-4 cosine, paged_adamw_8bit. VRAM during train ≈ 12 GB.
Target: LABEL only (REASON masked). Fresh adapter per seed.

### 3.4 Eval protocol
Stratified 400-sample test set per seed, seeds {42, 123, 777}. Both exemplar
sampling and test sampling reseeded per run. Report mean ± std across 3 seeds.

## §4 Findings

1. **3-tier adaptation cumulates.** Prompt, ICL, LoRA each add +12–13 pp
robust gain over the previous tier. Tiers are statistically separated.
2. **Ekman FACS prototypes = random** on Korean data (25.42 ± 0.38 %),
despite being the textbook canonical representation.
3. **Target-culture presence → mean**: 1 Korean exemplar is sufficient;
Korean count within [1,4] does not materially change mean accuracy.
4. **Diverse-anchor count → variance**: threshold at k_Wes ≥ 2;
std drops 4-6 pp → < 1 pp. Effect is on exemplar-sampling variance,
not on mean.
5. **LoRA subsumes ICL** for same target-distribution; stacking causes
small regression (−0.75 to −4 pp). Mixed exemplars hurt less than pure.
6. **k-saturation**, not inverted-U. Single-seed "k=16 drop" was a seed artifact.

## §5 Discussion

### 5.1 Why do Western anchors reduce variance?
Hypothesis: idealized Ekman prototypes lie on a distinct manifold from real Korean
face distributions. When mixed in context, they act as *structural regularizers*:
they give the model a stable frame for "what the label classes look like" independent
of the specific Korean subsample. Pure Korean exemplars vary widely (sampling noise in
AU intensity), producing high variance across seeds. Adding non-semantic Ekman
anchors pins down the label structure without competing with the Korean signal.

This predicts: (i) random-token or noise-pattern anchors should also reduce
variance if they provide structural class-coverage; (ii) the effect should
disappear if the anchors overlap with the Korean distribution. (Future work.)

### 5.2 Why is adaptation ordinal?
LoRA trained on 10K Korean samples has internalized the Korean distribution.
Adding the same distribution as ICL context introduces redundancy; the model's
attention splits between context and internal representation, with marginal
cost. This is consistent with the finding that Mixed (which partially lies
*outside* the Korean distribution via Ekman anchors) hurts less than pure Korean.
→ ICL helps when LoRA has *not* seen the distribution; once it has, more ICL is noise.

### 5.3 Implications for cultural LLM agents
- Abstract cultural prompts ("this person is Korean") are ineffective.
- FACS textbook prototypes are inadequate for non-Western faces.
- Best low-cost intervention: 1-2 target-culture exemplars + 2 anchors.
- If training data available: LoRA > ICL; stacking is counterproductive.

## Limitations

- Single base model (Qwen2.5-7B); no ablation over model family/scale.
- AU-intensity input, not raw face image — Qwen-VL image path untested.
- Only one target culture (Korean); cross-cultural extension needed.
- 1-epoch QLoRA; longer training or larger rank may compound further.
- Mechanism for anchor-variance effect (§5.1) is hypothesis, not verified
  (attention-map analysis = future).

## Remaining tasks (choose next)

1. **Paper writing fleshout** — §1 intro, §2 related work, plots, polish.
2. **SOTA baseline comparison** — Emotion-LLaMA on Korean FER AU (need
   conversion) or direct comparison on a shared benchmark (e.g., RAF-DB).
3. **Mechanism verification for §5.1** — attention-map analysis showing
   where model attends in Mixed vs pure Korean exemplar cases.
4. **Cross-culture replication** — same protocol on e.g. Japanese/Chinese FER.
5. **MELD text extension** — does "anchor regularization" generalize to text
   emotion classification (MELD conversation + English/Korean exemplars)?

Session 2026-04-23 delivered findings 1-6 (§4). §5 discussion has two
open hypotheses (mechanism for variance effect, ordinality of adaptation).
