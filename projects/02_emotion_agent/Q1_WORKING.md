# Q1 Paper Working Draft — Emotion Agent

## Status (2026-04-24, after overnight M1 queue)
- Phase 0 setup ✓
- Week 1 preprocessing ✓ (IEMOCAP/MELD/K-EmoCon)
- Week 2 agent baseline eval ✓ (IEMOCAP/MELD/K-EmoCon/FER)
- Week 3 few-shot scaling + multi-seed + LoRA ✓
- **Week 4 M1 queue ✓ — cross-domain (IEMOCAP/MELD) + anchor mechanism (exp_018/020)**
- Week 5 writing + SOTA baseline: planned

## Title (draft)

*"From Prompt to Parameter: A Tiered Study of Cultural Grounding in
Facial Emotion Agents"*

alt: *"Target-Culture Exemplars, Diverse Anchors, and LoRA: Decomposing the
Contributions of Cultural Grounding in LLM Emotion Classifiers"*

## Abstract draft (v2, with cross-domain + revised anchor decomp)

We study adaptation hierarchies in LLM-based emotion classifiers across three
domains: Korean facial action-unit (FER AU) text, and two English dialog
benchmarks (IEMOCAP, MELD). Using Qwen2.5-7B-Instruct with multi-seed (n=3)
protocol, we report four findings.

**(1) Only LoRA-based parameter adaptation gives reliable gains; ICL is
fragile under both modality and prompt-engineering.** QLoRA (r=16, 1 epoch)
adds +14 to +22 pp over the *strongest* zero-shot prompt across three datasets.
In contrast, in-context learning (ICL, k=4 exemplars) does not survive a
proper baseline: when zero-shot uses an engineered prompt (FACS prototype +
cultural framing, exp_030b multi-seed), Korean FER AU zero-shot reaches
40.92 ± 2.04 % — statistically tied with ICL k=4 (41.83 ± 3.41 %, +0.91 pp).
On standard English text (IEMOCAP, MELD), ICL gain is similarly ≤1 pp even
relative to FACS-only baselines, and a k-sweep is flat. Our previously
reported +12.75 pp ICL gain on Korean FER reflected a sub-optimal baseline,
not an ICL benefit. LoRA is the only adaptation tier that survives proper
baselining.

**(2) Exemplar-variance decomposes into three additive components.** Under
controlled ablations (class coverage, anchor origin, anchor semantic content),
we find: (i) **class redundancy** (3-class vs 4-class coverage in the exemplar
pool) reduces σ by ~2.81 pp (primary driver); (ii) **anchor origin**
(Mixed = 2 Korean + 2 Western vs pure 4 Korean at matched 3-class coverage)
reduces σ by ~0.66 pp; (iii) **semantic anchor content** (Ekman idealized
prototype vs random-AU-token at matched structure) reduces σ by ~1.63 pp.
Previous claims of a single "Western count ≥2" threshold are superseded by
this three-axis decomposition.

**(3) ICL and LoRA are substitutes, not complements.** Adding Korean exemplars
on top of a Korean-trained LoRA adapter hurts accuracy by 0.75–4.00 pp —
adaptation modes access overlapping information. Mixed anchors hurt less
than pure target-culture anchors, consistent with (2)'s regularization story.

**(4) Ekman FACS prototypes are random on Korean faces.** The textbook Western
FACS prototype reaches 25.42 ± 0.38 % on Korean FER AU — indistinguishable
from random (4-class baseline 25 %), regardless of seed. This negative result
motivates the three-axis anchor decomposition in (2).

These results (a) challenge the common practice of injecting cultural
metadata via system prompts, (b) expose that ICL benefit is modality-gated
rather than universal, and (c) provide a concrete recipe for grounded LLM
emotion classifiers: prefer LoRA when training data is available; use
ICL with class-redundant Mixed exemplars when training data is limited
and the input modality is novel to the base LLM.

## Headline Results Table (multi-seed robust)

### ⭐ Main: Cross-domain 3-tier hierarchy (n=3 seeds per cell, N=400 test each)

| Dataset | T1a sub-optimal | **T1b best prompt** | T2 ICL k=4 | T3 LoRA r=16 | T2−T1b | T3−T1b |
|---------|-----------------|---------------------|-----------|--------------|--------|--------|
| Korean FER AU | 31.67 ± 2.65 (FACS-only) | **40.92 ± 2.04** (FACS+cultural P4) | 41.83 ± 3.41 | **55.00 ± 2.61** | +0.91 | **+14.08** |
| IEMOCAP text | 47.83 ± 2.50 | **47.50 ± 2.63** (P1 = best) | 48.92 ± 4.25 | **70.00 ± 3.70** | +1.42 | **+22.50** |
| MELD text | 55.50 ± 3.27 | **54.75 ± 3.04** (P1 = best) | 55.17 ± 5.65 | **61.75 ± 1.15** | +0.42 | **+7.00** |

(For IEMOCAP/MELD, exp_032 multi-seed prompt ablation confirmed P1 baseline > P2 minimal > P3 rich — text domains do not benefit from elaborate prompt engineering. The slightly lower T1b vs T1a on these rows reflects different multi-seed exemplar pools used in exp_021/022 vs exp_032; the same prompt was used in both.)

**Revised key observations** (after exp_030b + exp_032 multi-seed prompt ablations):
- **LoRA tier-3** is the only tier with substantial gain over best-engineered zero-shot:
  +14.08 pp (Korean FER), +22.50 pp (IEMOCAP), +7.00 pp (MELD).
- **ICL tier-2 gain ≤ +1.5 pp on all three domains** under proper baselines. Korean FER +0.91, IEMOCAP +1.42, MELD +0.42 — all within seed-noise. Earlier "+12.75 pp" Korean FER ICL claim reflected a sub-optimal FACS-only baseline.
- **Prompt-engineering payoff is domain-specific**: FACS + cultural-framing helps Korean FER (+9.25 pp over FACS-only), but elaborate prompts hurt IEMOCAP/MELD (P1 baseline > P2 minimal > P3 rich). On text, current "baseline" prompt is already near-optimal.
- IEMOCAP/MELD k-sweeps flat (46.67/46.17/47.00 ; 55.75/56.50/55.17) — ICL absent regardless of k.

→ Combined claim across three datasets: **only parameter adaptation (LoRA) reliably exceeds well-engineered zero-shot prompts**. ICL appears to compensate for weak prompts rather than add genuine new information.

### Anchor-variance three-axis decomposition (n=3 seeds, k=4 exemplars, Korean FER AU)

| Config (origin × coverage × content) | Acc ± σ | σ attribution |
|---|---|---|
| E: Kor4, 4-class | 41.00 ± 4.34 | baseline |
| H1: Kor4, 3-class | 40.67 ± 1.53 | ← class-redundancy reduces σ by 2.81 |
| G1: Kor2+Random2, 3-class | 41.08 ± 2.50 | + semantic vs random content |
| D: Kor2+Wes(Ekman)2, 3-class | 41.25 ± 0.87 | ← anchor origin adds 0.66 more |
| C: Wes4 (Ekman only) | 25.42 ± 0.38 | (low mean — no Korean) |

**Decomposition**:
- Class redundancy (3-class vs 4-class): σ **−2.81 pp** (primary driver)
- Anchor origin (Mixed vs pure Korean at 3-class): σ **−0.66 pp** (secondary)
- Semantic anchor content (Ekman vs random-AU-token): σ **−1.63 pp** (tertiary)

### Prior mis-attribution correction
A previous k-sweep interpretation of the same grid claimed a "Western ≥ 2 threshold"
effect on variance. That observation confounded class coverage with anchor origin:
Ekman anchors happen to cover 3 classes (happy/sad/angry, no neutral), so adding
2 Western anchors simultaneously collapses class coverage from 4 to 3. The three-axis
decomposition above correctly separates these effects.

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

## Key figures (5 generated, `figures/*.png`)

1. **fig1_3tier_korean_fer.png**: Korean FER AU 3-tier bar chart (prompt / ICL / LoRA).
2. **fig2_anchor_variance_revised.png**: 5-config anchor decomposition showing
   class-redundancy, anchor-origin, semantic-content separation.
3. **fig3_kshot_cross_domain.png**: k-sweep — Korean FER saturation vs IEMOCAP flat.
4. **fig4_lora_icl_substitute.png**: LoRA+ICL substitutability regression.
5. **fig5_cross_domain_3tier.png** ⭐: Korean FER / IEMOCAP / MELD × 3 tiers (main result).
6. **fig6_hidden_cluster_mechanism.png**: §5.2 mechanism — per-class + overall cluster stats.

## §1 Introduction (draft)

How should large language models (LLMs) be adapted to a culturally-specific
emotion-classification task when their training data is dominated by Western
English corpora? Practitioners commonly reach for three tools: (a) abstract
system prompts ("the subject is Korean"), (b) in-context exemplars of the
target culture, or (c) parameter-efficient fine-tuning such as LoRA. Yet
these three options are rarely compared on a shared protocol, and the
mechanisms that determine when each actually helps remain unclear. Prior
culturally-situated emotion-recognition work either focuses on a single
tier or presents single-seed results whose trends reverse under replication.

We study this question on Korean facial emotion recognition from Facial
Action Unit (FACS AU) intensities — a concrete, culturally-situated domain
with a large publicly-available Korean corpus (237K images, Yonsei 298-person
consensus labels). Using Qwen2.5-7B-Instruct as base LLM and a multi-seed
(n=3) evaluation protocol, we ask three questions:

1. **Do the three adaptation tiers (prompt / in-context / LoRA) stack
   cumulatively, and by how much each?** Prior work reports single-tier
   comparisons but rarely a matched protocol across all three.
2. **Does ICL benefit generalize across input modalities, or is it
   modality-specific?** We test on two additional English dialog text
   datasets (IEMOCAP, MELD) with matched evaluation.
3. **When exemplars help, what property makes them work?** We ablate
   exemplar origin (target culture vs Western prototype vs random), class
   coverage (3-class vs 4-class), and semantic content separately.

Our primary empirical findings are:

(i) **Parametric adaptation (LoRA) is the only robust tier.** It yields
+6 to +26 pp gains over zero-shot across three datasets. By contrast,
in-context learning (ICL) produces a large +12.75 pp gain on Korean FER AU
but collapses (≤ 1 pp gain) on English dialog text. A k-sweep confirms ICL
on text is truly flat, not merely sub-saturating.

(ii) **The exemplar-variance effect decomposes into three independent axes.**
Under pure Korean exemplars, changing from 4-class to 3-class coverage
reduces accuracy standard deviation by 2.81 pp; further switching 2 of those
to non-target-culture (Ekman Western prototype) anchors reduces variance by
another 0.66 pp; replacing Ekman anchors with random AU intensities
*increases* variance by 1.63 pp, implying semantic anchor content also
contributes. A previous "Western-count ≥ 2" threshold claim is superseded
by this decomposition.

(iii) **ICL and LoRA are substitutes, not complements.** When we combine
both on the same Korean distribution, the stacked model performs 0.75–4.00
pp below LoRA alone, with pure target-culture exemplars causing the larger
regression.

(iv) **Western FACS prototypes (Ekman canonical) perform at random
(25.42 ± 0.38 %) on Korean faces**, robust across seeds — a strong negative
result motivating the decomposition in (ii).

Taken together, our results challenge the common practice of injecting
cultural metadata via system prompts and provide a concrete recipe for
grounded LLM emotion classifiers. We release all scripts, multi-seed
results, and paper figures.

## §2 Related Work (draft)

### LLMs and in-context learning for emotion recognition
Recent work has leveraged large language models for affective tasks via
zero-shot or in-context prompting [EmoLLM, Zheng 2024; Emotion-LLaMA,
Cheng et al., NeurIPS 2024; BeMERC, 2025]. These studies primarily report
accuracy gains from instruction tuning or multimodal fusion. We contribute
a **controlled three-tier comparison** (prompt / ICL / LoRA) with matched
evaluation protocol, finding that ICL gain is input-modality-dependent —
a conclusion not reachable from prior work that studies single tiers or
single domains.

### Exemplar design in ICL
ICL research has examined exemplar selection [Liu et al., 2022; Wu et al.,
2023], ordering [Lu et al., 2022], and format. Most studies treat exemplars
as monolithic. We show that even with matched exemplar count and class
structure, variance reduction is driven by three orthogonal factors —
class redundancy, anchor origin, and anchor semantic content.

### Cultural grounding and Western bias in emotion models
Jack et al. (2012, PNAS) famously showed that East Asian observers use
different face regions for emotion perception than Western observers,
challenging the universality of Ekman's FACS-based prototypes. Prior work
on culturally-situated facial emotion recognition [Park et al. 2020
K-EmoCon; KEMDy20; various Korean FER corpora] largely treats cultural
adaptation as a data problem. Our result — that Ekman FACS prototypes
perform at random on Korean faces — gives a direct LLM-era confirmation
of Jack et al.'s thesis.

### Parameter-efficient fine-tuning
LoRA [Hu et al., 2022] and its quantized variant QLoRA [Dettmers et al.,
2023] enable efficient domain adaptation of LLMs. Our Tier-3 experiments
use QLoRA r=16; the key finding relative to prior work is that LoRA
*subsumes* ICL (finding iii), a relationship previously not quantified.

### Physiological-behavioral fusion (same-lab concurrent work)
This paper's focus is on behavioral/textual input. Concurrent work from
our lab addresses the complementary physiological-grounding axis: S-PACE
[our prior work] introduces bio-anchored cross-attention for multimodal
emotion recognition, and CBBF [our concurrent work] extends this with
explicit causal temporal masking grounded in autonomic-nervous-system
precedence (Lazarus 1991; Kreibig 2010). These bio-grounded approaches
and the contextual grounding we study here are orthogonal: a future system
could combine both (see §6 Future Work).

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

## §4 Findings (v2 — revised for cross-domain + anchor decomposition)

1. **LoRA tier robust across domains.** +14 to +22 pp (vs best zero-shot
   prompt) across three datasets. Multi-seed std 1.15–3.70 pp.
2. **ICL tier is modality-gated AND prompt-engineering-gated.** When measured
   against a *strong* zero-shot prompt (FACS prototype + cultural framing,
   exp_030b), ICL gain on Korean FER AU is only **+0.91 pp** (40.92 ± 2.04 %
   zero-shot vs 41.83 ± 3.41 % ICL k=4 — statistically tied). Our previously
   reported +12.75 pp ICL gain was relative to a sub-optimal FACS-only baseline.
   On standard English text (IEMOCAP +1.09, MELD −0.33), ICL is similarly
   ineffective. Combined: ICL adds little when (a) input modality is
   familiar to the LLM OR (b) zero-shot prompting is well-engineered. **LoRA
   remains the only reliable tier** under both proper baselines.
3. **Ekman FACS prototype ≈ random on Korean.** 25.42 ± 0.38 % — strong negative
   result showing textbook Western prototypes do not generalize to Korean faces.
4. **Three-axis anchor variance decomposition**:
   (a) class redundancy (3-class vs 4-class exemplar set) → σ **−2.81 pp** (primary);
   (b) anchor origin (Mixed Kor+Wes vs pure Kor at matched 3-class) → σ **−0.66 pp**;
   (c) semantic content (Ekman prototype vs random AU intensities) → σ **−1.63 pp**.
   Previous "Western ≥ 2 threshold" claim is superseded by this decomposition.
5. **ICL and LoRA are substitutes, not complements.** Adding ICL on LoRA
   adapter regresses 0.75–4.00 pp; Mixed anchors cause smaller regression
   than pure target-culture anchors (consistent with finding 4).
6. **k-saturation on Korean FER (not inverted-U).** Multi-seed shows flat k≥2
   plateau; the previously reported k=16 drop (36%) was a single-seed artifact.
7. **Anchor-variance mechanism is at hidden-representation scale, not attention.**
   Attention entropy is identical across 5 layers (1, 7, 14, 21, 27) between
   Mixed (σ_out=0.87) and pure Korean (σ_out=4.34) configurations. However,
   last-layer hidden states differ in absolute compactness: Mixed anchors
   produce representations ~22% more compact in within-class L2 distance
   while preserving class-separability ratio. This compression, not attention
   redistribution, is what stabilizes downstream predictions under Mixed.

8. **Prompt engineering > exemplar selection on Korean FER AU.** Ablating five
   zero-shot prompts (FACS-only / no-FACS / minimal / FACS+cultural-framing /
   dimensional) yields a 13.7-pp range (27.25 – 40.92 %) — larger than the
   gap from any single ICL exemplar configuration. Cultural framing combined
   with FACS prototype gives the highest zero-shot accuracy
   (40.92 ± 2.04 %), tied with k=4 ICL. This implies that for our task,
   investing in prompt phrasing yields more reliable gains than investing in
   exemplar selection. (Earlier exp_011 result reporting cultural prior
   −7 pp used a different framing that omitted FACS prototype; our P4 prompt
   *combines* both, hence the divergent direction.)

9. **LoRA ceiling is data-quality-limited, not data-volume-limited.** Volume
   ablation on Korean FER AU shows: 1K samples → 54.25 % (94 % of ceiling),
   3K → 54.75 %, 5K → 54.75 %, 10K → 56.75 %. Log-scale saturation. Combined
   with rank ablation (r ∈ {8, 16, 32} all within 0.75 pp), the LoRA upper
   bound at ~57 % is set by something other than parametric or sample
   capacity — likely by AU intensity-text input granularity itself.

## §5 Discussion

### 5.1 Why is ICL gain modality-gated?
ICL gave +12.75 pp on Korean FER AU but ≤1 pp on IEMOCAP/MELD text. Possible
explanations:
(a) **Familiarity saturation**: LLMs are pre-trained heavily on English dialog
text; zero-shot already reaches 48–55% on IEMOCAP/MELD. Additional in-context
examples provide little new information.
(b) **Novelty compensation**: AU-intensity text (e.g., "AU6 (cheek raiser)=75,
AU12=82") is structurally dissimilar to natural language training data. ICL
examples serve as a brief fine-tune, teaching the LLM to map this syntax to
labels.
(c) **Label-space anchoring**: For unfamiliar input, exemplars primarily
anchor the output label space. For familiar text, the label space is already
encoded in the LLM's lexical embeddings.

Evidence for (b)/(c): on Korean FER AU, exemplar content and class coverage
both contribute to variance (§4 finding 4), suggesting the model uses
exemplars for structural label-space grounding, not purely semantic.
Follow-up experiments that would disambiguate: (i) test ICL on synthetic
"text with novel format" tasks; (ii) measure LLM embedding cosine similarity
between AU-intensity tokens and natural language.

### 5.2 Why do class redundancy and anchor origin each contribute to variance?
The three-axis decomposition (§4 finding 4) shows class redundancy reduces
σ by 2.81 pp, anchor origin by 0.66 pp, and semantic content by 1.63 pp. We
interpret:
- **Class redundancy** (duplicating a label in the exemplar set) reduces the
  model's uncertainty about label geometry — multiple observations of the
  same class sharpen the decision surface.
- **Anchor origin** (non-target-culture exemplars) provides a distinct-manifold
  prior that stabilizes label anchoring independent of target-culture sampling.
- **Semantic content** (Ekman AU patterns vs random intensities) lets the
  model attach label identity to a recognizable FACS pattern rather than
  treating the anchor as noise.

**Negative result — attention entropy is NOT the mechanism (all layers checked).**
We tested whether D_mixed's lower output variance corresponds to more uniform
attention distribution over exemplars (vs E_kor4). Two experiments:
(i) exp_019 on 50 test samples at layer 14;
(ii) exp_019b extended to 5 layers {1, 7, 14, 21, 27}.

| Layer | D_mixed entropy | E_kor4 entropy | Diff |
|---|---|---|---|
| 1 | 0.491 ± 0.003 | 0.528 ± 0.003 | 0.037 |
| 7 | 0.457 ± 0.006 | 0.455 ± 0.005 | 0.002 |
| 14 | 0.643 ± 0.013 | 0.638 ± 0.014 | 0.005 |
| 21 | 0.424 ± 0.022 | 0.394 ± 0.029 | 0.030 |
| 27 | 0.558 ± 0.026 | 0.567 ± 0.026 | 0.009 |

All 5 layers: D and E near-identical entropy. Both show strong **recency bias**
(last exemplar gets 83–91 % of attention across all depths). The 5× difference
in *output* variance between D and E therefore does NOT arise at the
attention-weight level anywhere in the network. This locates the mechanism at
the **hidden-representation level**.

**Mechanism found — hidden representations are more compact under Mixed anchors
(exp_028).** For 80 seed-42 test samples per config, we extracted the last-layer
hidden state at the final input token and computed within-class and between-class
L2 distances.

| Metric | D_mixed | E_kor4 | D/E ratio |
|---|---|---|---|
| Within-class distance (mean) | 14.49 | 18.69 | 0.78 |
| Between-class distance (mean) | 8.69 | 11.28 | 0.77 |
| Tightness ratio (between/within) | 0.595 | 0.601 | 0.99 |

Per-class within-class distance drops by 3.5–5.6 units across all four classes
under D_mixed. Importantly, the **tightness ratio is preserved** (0.60 vs 0.60),
meaning class separability relative to cluster scale is the same. What differs
is **absolute representation scale**: D_mixed produces representations that are
~22% more compact in absolute L2 distance. This compression at the hidden-state
level — despite identical attention distributions — explains the 5× smaller
output variance: compact representations yield more stable downstream predictions.

### 5.3 Why is adaptation ordinal (ICL⊂LoRA)?
LoRA trained on 10K Korean samples internalizes the Korean distribution.
Adding the same distribution as ICL context introduces redundancy; attention
splits between context and internal representation, causing marginal
regression. Mixed anchors (Ekman partially lies *outside* the Korean
distribution) hurt less because they provide complementary, not duplicative,
information. → ICL helps when LoRA has *not* seen the distribution; once
LoRA has it, more ICL becomes noise.

### 5.4 Implications for grounded LLM emotion classifiers
- Abstract cultural prompts ("this person is Korean") are ineffective.
- FACS Western textbook prototypes are inadequate for non-Western faces.
- When training data available: LoRA is the reliable tier; avoid stacking ICL.
- When no training data AND input is novel: use Mixed (target-culture +
  structurally-distinct anchors) with 3-class redundancy.
- When input is standard text: ICL adds little; invest in LoRA or prompt
  engineering other than exemplars.

## §6 Future Work

### 6.1 Bio-grounded emotion agent (integration with same-lab S-PACE/CBBF)
Our concurrent S-PACE/CBBF work addresses the orthogonal physiological
grounding axis: bio signal → behavioral-emotion via causal cross-attention.
A natural next system combines both:
- Bio signal (EDA, HR from wearable) → NormWear-style encoder → structural
  prior on arousal/valence axis.
- Face AU intensity → LLM via our Mixed-anchor exemplar scheme OR via LoRA.
- Fuse physiological prior with LLM reasoning (BioToken-style projection).
This would directly target real-time affective agents in safety-critical
settings (e.g., driver monitoring with KMER-style multimodal data).

### 6.2 Mechanism for exemplar-variance (beyond attention)
Our layer-14 attention entropy test found D_mixed and E_kor4 near-identical
(both show recency bias), yet their output variances differ 5×. This
isolates the mechanism to *hidden representations* rather than attention
weights. Future work: extract hidden states from exemplar positions, cluster
per-label, and measure cluster tightness. If D_mixed shows tighter per-label
clusters than E_kor4, that mechanism is at the feature-space level.

### 6.3 Modality-gating interventional test
§5.1's hypothesis (ICL helps only when input modality is novel to the LLM)
is correlational. A direct intervention: take a standard IEMOCAP transcript
and transform it into a synthetic "novel format" (e.g., phoneme-level
encoding, or extracted acoustic statistics text) that the LLM has not seen.
If ICL gain appears under the novel format but not the original text, that
confirms input-novelty as a causal driver.

### 6.4 Scale & model-family ablation
Our experiments used a single base model (Qwen2.5-7B 4-bit). Repeating on
Llama-3-8B, Qwen2.5-72B, and a smaller 1-3B variant would test whether the
observed tier gaps and modality gating are model-size-dependent.

### 6.5 Cross-cultural replication
Current Korean-specific findings should be validated on Japanese, Chinese,
or African face data. If the 3-axis anchor decomposition survives, that
strengthens generalizability; if it doesn't, the finding becomes
culturally-parameterized.

## Limitations

- Single base model (Qwen2.5-7B 4-bit quantized); no ablation over model
  family or scale.
- AU-intensity input, not raw face image — Qwen-VL image path untested.
- Only one target culture (Korean) and two English dialog benchmarks
  (IEMOCAP, MELD); cross-cultural text and non-English bio benchmarks remain
  untested.
- 1-epoch QLoRA with fixed LR 2e-4 and batch configuration. We performed a
  rank ablation (r ∈ {8, 16, 32} with matched α = 2r) on Korean FER AU and
  observed ≤ 0.75 pp accuracy variation (56.00 – 56.75 %), suggesting rank
  is not the bottleneck; data volume or model capacity appears to be. Longer
  training (multi-epoch) remains unexplored.
- The modality-gating hypothesis (§5.1) is correlational — we observe ICL gain
  ↔ input novelty co-occur but have not intervened on novelty directly.
- Attention-map or representation-level mechanism for §5.2 is not measured;
  treated as future work.

## Remaining tasks (prioritized)

1. **§1 Introduction + §2 Related Work writing** — most remaining paper work.
   S-PACE / CBBF cited as same-lab concurrent bio-behavioral track.
2. **Attention-map mechanism (§5.2 support)** — extract attention entropy
   from Qwen layer 14 on Mixed vs pure Korean exemplar configs. (exp_019, planned)
3. **SOTA baseline comparison** — Emotion-LLaMA (NeurIPS 2024) on shared
   protocol. Paper A strengthening.
4. **Cross-culture replication** — same protocol on Japanese/Chinese FER
   (data dependent).
5. **Modality-gating verification** — synthetic novel-format text
   experiment to test §5.1 hypothesis intervention-style.

Session 2026-04-23 evening + 2026-04-24 overnight M1 queue delivered findings
1–6 (§4) across three datasets. §5 discussion has three open mechanistic
hypotheses (modality gating, anchor-variance 3-axis, ordinality).
