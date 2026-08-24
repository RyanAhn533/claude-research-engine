# exp_056 unlabeled-exemplar control — RESULTS (2026-07-16)

**Goal:** disentangle FORMAT GROUNDING from TASK RECOGNITION. Our zero-shot prompt already
names the task + label space (+ FACS prototypes for AU), so exp_040's `TR = random - zero`
does NOT cleanly measure Pan&Gao task recognition. Add an `unlabeled` condition (exemplar
inputs shown, LABEL lines removed) → 3-way decomposition:

```
GROUNDING = unlabeled - zero      (map alien input schema onto the already-known task)
RECOG     = random    - unlabeled (deranged label-space term)
TL        = gold      - random    (correct input→label mapping)
gain      = GROUNDING + RECOG + TL
```

**Setup:** 6 models × 3 datasets × 7 seeds × 4 conditions. AU=numeric no-FACS (novel),
IEMOCAP+MELD=English dialog (familiar). random = DERANGED labels (no exemplar keeps its true label).

## Verdict: GROUNDING is the mechanism, NOT task recognition. (cross-checked ✅)

### AU (novel) — GROUNDING dominates the gain
| model | gain | GROUNDING | (% of gain) | RECOG | TL |
|---|---|---|---|---|---|
| qwen7b  | +16.1 | +15.0 | 93% | −4.0 | +5.1 |
| falcon7b| +15.1 | +13.5 | 89% | −5.5 | +7.1 |
| qwen14b | +14.7 | +12.4 | 84% | −2.7 | +5.0 |
| yi6b    | +11.0 |  +6.9 | 62% | +0.8 | +3.4 |
| qwen3b  |  +8.1 |  +9.5 |116% | −4.1 | +2.8 |
| mistral |  +2.3 |  +2.4 | null| −4.3 | +4.1 |

- Label-free exemplars alone recover **62–116%** (median ~89%) of the full gold-ICL gain.
- Strong-gating models (qwen7b/14b/3b, falcon) have AU GROUNDING **positive in 7/7 seeds**.

### IEMOCAP + MELD (familiar) — GROUNDING ≈ 0
GROUNDING ∈ {−1.4 … +2.9} across all models/datasets. Nothing alien to ground.

### RECOG < 0 on AU for 5/6 models
On novel formats, DERANGED labels actively *hurt* vs. unlabeled → Min-et-al "random labels are
fine" breaks. The old `TR = random − zero` conflated grounding (large +) with label-damage (−).

## Robustness / integrity (independent cross-check, research-cross-checker)
- unlabeled truly label-free (LABEL line removed; no leakage via exemplar count/order). CONFIRMED
- derange_labels is a true derangement (4 distinct labels/class). CONFIRMED
- GR+RC+TL == gain exactly for all 18 (model×dataset) cells. CONFIRMED
- exemplar/test disjoint; test class-balanced (100/class). CONFIRMED

## ⚠️ Caveats for the paper
1. **Do NOT cross-compare exp_056 TEXT absolute gains to exp_035 published gains** — prompt_text
   here omits exp_035's "Here are N labeled examples:" header. Internal GR/RC/TL decomposition is
   valid (all 4 conditions share the builder); only cross-experiment absolute-value comparison is off.
2. **RECOG is a deranged-label term, not a clean "label-space-present-but-uninformative" control.**
   Anchor the mechanism claim on GROUNDING (label-free recovery), use RECOG<0 as a secondary
   "wrong labels hurt on novel formats" point — do NOT claim "task recognition = 0".

## Paper implications
- Retitle: drop "via Task Recognition" → **format grounding** framing.
- C2 rewrite: gain's bulk is label-free format grounding (62–93%), not label-mapping learning;
  correct labels add a little (TL +3–7), wrong labels hurt (RECOG −3–5).
- New figure (figE): stacked GR/RC/TL bars, AU vs IEMOCAP vs MELD across models.
