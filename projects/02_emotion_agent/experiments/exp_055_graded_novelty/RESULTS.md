# exp_055 graded format-novelty → ICL gain (dose-response) — RESULTS (2026-07-14)

**Goal:** rescue the fertility contribution — test whether, WITHIN one task (SST-2), ICL gain
rises monotonically (graded) with format novelty (graded leetspeak p∈{0,.25,.5,.75,1}), which
would upgrade fertility from a between-format signal to a graded causal predictor.

**Setup:** 6 models × 3 seeds × 5 novelty levels × {zero, gold-ICL, random-ICL}, N=200 SST-2.

## Verdict: rescue FAILED (honestly). Fertility is NOT a graded within-task predictor.
- within-task Spearman(fertility, gain) is inconsistent across models: +0.40,+0.50,+0.30,−0.90,+1.00,−0.50.
- raw-gain peak location is model-dependent and noisy (p=0.25 for 2, p=0.75 for 2, p=1.0 for 1).
- => the "fertility linearly predicts gain within a task" claim does NOT hold. The paper's existing
  demoted framing (between-format novelty signal, within-family ρ≈0) remains the correct ceiling.

## What IS robust (pooled, 5 real models; mistral excluded as null):
| p | fertility | zero-shot | gain | TR (recognition) | gain/headroom |
|---|---|---|---|---|---|
| 0.0 | 1.20 | 90.1 | +2.8 | +0.4 | 24% |
| 0.25| 2.37 | 77.9 | +8.1 | +6.9 | 36% |
| 0.5 | 3.34 | 70.6 | +5.2 | +3.8 | 20% |
| 0.75| 4.08 | 61.2 | +7.1 | +5.7 | 20% |
| 1.0 | 4.77 | 57.0 | +5.5 | +3.2 | 13% |

1. **p=0 (familiar) = minimum gain for 4/5 models.** Confirms core thesis at graded resolution.
2. **max novelty (p=1.0) is the peak for only 1/5** → gain does not keep rising with novelty.
3. **Recovery fraction declines with novelty (36→20→20→13%)** = ICL recovers a shrinking share of
   the novelty-induced accuracy drop → a *quantitative* hint of the "recoverable-novelty window"
   the paper currently only asserts (cf. cipher-reasoning cite).
4. **TR (task recognition) tracks gain at every level and is ~0 at p=0** → mechanism is unchanged
   and on-message across the whole novelty axis.

## ⚠️ Design confound (why NOT to headline this)
Leetspeak maps BOTH `i`→`1` and `l`→`1`: at high p the text becomes genuinely **lossy/ambiguous**,
not merely reformatted. So the p=1.0 collapse and the declining recovery are partly *information
loss*, not pure novelty. This confounds the "recoverable-novelty window" reading.

## Recommendation
- Do NOT add an inverted-U / graded-predictor claim to the paper (noisy + confounded). This would
  re-introduce exactly the kind of overclaim we removed this session.
- Optional honest 1–2 sentence support (only if JY wants it): "graded novelty confirms familiar=least
  gain and TR as the mechanism throughout; recovered fraction shrinks with novelty." Frame as support,
  not a new contribution.
- If we ever want the window as a real result: redo with an **information-preserving** novelty knob
  (e.g. bijective char cipher, byte-pair remap) so high novelty ≠ information loss.
