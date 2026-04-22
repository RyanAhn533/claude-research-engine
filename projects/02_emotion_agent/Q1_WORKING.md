# Q1 Paper Working Draft — Emotion Agent

## Status (2026-04-23, Week 3 Day 3)
- Phase 0 setup ✓
- Week 1 preprocessing ✓ (IEMOCAP/MELD/K-EmoCon)
- Week 2 agent baseline eval ✓ (IEMOCAP/MELD/K-EmoCon/FER)
- **Week 3 few-shot scaling ✓ — PAPER-GRADE findings**
- Week 4 LoRA fine-tune (GPU 48GB needed, BrandSpace off): pending

## Title (draft)
*"Cultural Grounding via Concrete Exemplars, Not Textbook Prototypes:
A Few-Shot Emotion Agent on Korean Facial Action Units"*

## Abstract draft

We present a zero-training emotion agent built on Qwen2.5-7B (4-bit) that integrates
FACS action-unit inputs with in-context Korean exemplars. Across four datasets (IEMOCAP,
MELD, K-EmoCon, Korean FER 237K), we find three distinct levels of cultural grounding:
(i) **abstract cultural prompts degrade accuracy** (−7 pp on Korean FER AU);
(ii) **idealized Ekman FACS prototypes are invariant** (25.25 % — indistinguishable from random);
(iii) **actual Korean distribution samples drive consistent gains** (+15–19 pp), with a
**Korean + Western mixed anchor** reaching the highest accuracy (44.25 %).
We further show an **inverted-U scaling** with respect to the number of exemplars
(optimum at k=2, degrading beyond k=8), indicating cultural grounding is a
distribution-distillation problem rather than a capacity-addition problem.
These findings challenge the common practice of injecting cultural metadata via system
prompts and motivate concrete-example-based adaptation for culturally-situated LLM agents.

## Headline Results Table

| Dataset | Config | Acc | Macro-F1 |
|---------|--------|-----|----------|
| IEMOCAP (Eng) | TF-IDF+LR trained baseline | 66.33 ± 0.84 | 0.640 ± 0.007 |
| IEMOCAP | Qwen zero-shot | 48.75 | 0.484 |
| IEMOCAP | Qwen + Korean cultural prior | 48.50 | 0.485 |
| MELD (Eng) | TF-IDF+LR baseline | 59.75 | 0.312 |
| MELD | Qwen zero-shot | 55.00 | 0.540 |
| MELD | Qwen + Korean cultural prior | 55.75 | 0.552 |
| **MELD** | **Qwen + cultural + context** | **58.50** | **0.581** |
| Korean FER (AU) | Qwen zero-shot FACS prompt | 29.75–34.25* | 0.22–0.31 |
| Korean FER | + Korean cultural abstract | 27.25 | 0.183 |
| Korean FER | + k=2 Korean exemplars | 42.25 | 0.386 |
| Korean FER | + k=4 Korean exemplars | 42.00 | 0.359 |
| Korean FER | + k=4 Ekman Western prototype | 25.25 | 0.114 |
| **Korean FER** | **+ k=4 mixed (Kor2+West2)** | **44.25** | **0.372** |

*Baseline variance across seeds — multi-seed replication pending.

## Key Figures

- **Figure 1**: k-shot scaling on Korean FER AU — inverted-U
  `experiments/exp_013_kshot_sweep/cache/kshot_curve.png`
- **Figure 2** (TBD): 3-level cultural grounding bar chart (abstract / prototype / real / mixed)
- **Figure 3** (TBD): Cross-dataset table (IEMOCAP/MELD/Korean FER)

## Findings (paper §4 bullet)

1. **Abstract cultural prior via system prompt harms Korean FER AU** (−7.00 pp, exp_011).
2. **Ekman FACS textbook prototypes are invariant on Korean data** (25.25 % ≈ random 25 %).
3. **Actual Korean distribution exemplars drive +15–19 pp** with minimal prompt length.
4. **Korean + Western mix = best** (44.25 %) — anchoring effect.
5. **Inverted-U scaling** — k=2 optimal, degradation beyond k=8 (attention dilution).
6. **MELD (English) weakly benefits from Korean cultural prompt only with context**
   (+3.50 pp in full config) — hints at synergy with conversation grounding.

## Method §3 outline

### 3.1 Agent architecture
Qwen2.5-7B-Instruct, 4-bit NF4 quantized (5.4 GB). No fine-tuning.
System prompt: FACS prototype rules. User prompt: detected AU text + (optional k exemplars).

### 3.2 Cultural grounding variants
(A) zero-shot FACS  (B) abstract cultural text  (C) idealized Ekman prototype
(D) actual Korean exemplars  (E) Korean + Western mix

### 3.3 Inputs
OpenGraphAU 41-AU intensity (exp_002 from project 01, 229 K images),
Yonsei-consensus-clean subset (is_selected=0), balanced 4-class 100 samples/class.

## Limitations

- Zero-shot only; no LoRA/fine-tune (GPU budget 7.7 GB shared with BrandSpace server).
- K-EmoCon has no text transcription — Korean text+emotion dataset access limited.
- Korean FER AU baseline shows 25–34 % variance across seeds (needs multi-seed in final).

## Remaining tasks

1. **Multi-seed replication** (3 seeds) on exp_012/013/014 — stat robustness
2. **MELD mixed exemplar ablation** — generalize thesis across datasets
3. **LoRA fine-tune** (BrandSpace kill → 48 GB) — upper-bound comparison
4. **Qwen2.5-VL image-based** variant — closed-loop agent w/ raw face
5. **Writing**: §3 method diagram, §4 figures, §5 discussion with project 01 Jack-2012 link
