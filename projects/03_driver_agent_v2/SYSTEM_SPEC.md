# Driver Affective Agent v2 — Full System Specification

> **IP timestamp**: JY (Junyoung Ahn) — 2026-04-27.
> Post-graduation / company / startup stage scope. Recorded here for priority.
> This spec captures the full architecture intended for KMER-based commercial deployment.

## Overview

Real-time multimodal affective monitoring agent for Korean drivers, integrating
physiological signals, dual-camera face/scene video, and in-cabin audio into a
LoRA-adapted Qwen2.5-7B reasoning core. Outputs both classification labels
(emotion / drowsiness / stress) AND agentic intervention recommendations
(e.g., suggest break, adjust HVAC, trigger ADAS alert).

## Architecture (full v2)

```
Inputs (real-time @ ~0.5-2 Hz fusion rate):
  ┌───────────────────────────────────────────────────────────────────┐
  │ video_main (driver face)  → Vision encoder                        │
  │ video_sub (cabin/road)    → Vision encoder (shared weights)       │
  │ audio.wav (in-cabin)      → Whisper encoder (frozen)              │
  │ bio: GSR (EDA)            → NormWear bio encoder (S-PACE asset)   │
  │ bio: PPG (heart pulse)    → NormWear bio encoder                  │
  │ bio: Temp (skin)          → NormWear bio encoder                  │
  │ context: speed, GPS, time → simple feature embedding              │
  └───────────────────────────────────────────────────────────────────┘
                                ↓
               Per-modality MLP projection layers (trainable)
                                ↓
                   Token embeddings (unified token space)
                                ↓
       CBBF causal temporal fusion (S-PACE/CBBF prior, same-lab asset)
                                ↓
                Qwen2.5-7B-Instruct (4-bit NF4) + LoRA r=16
                                ↓
       Output: structured JSON
         {
           "emotion": "drowsy",
           "confidence": 0.87,
           "reasoning": "Bio HR variability ↓, eye closure rate ↑, ...",
           "intervention": {
             "tier": "warning",
             "action": "suggest_break_within_5min",
             "rationale": "Sustained drowsiness signal > 90s"
           }
         }
```

## Training pipeline

1. **Encoder pretraining**: skip — use pretrained Whisper/CLIP/NormWear frozen.
2. **Projection layer training** (~2-4h): MLP per modality, contrastive +
   classification objectives on KMER subset.
3. **CBBF temporal fusion**: import directly from same-lab CBBF code
   (`/home/ajy/CBBF-main/code/cbbf_conditioner.py`).
4. **LoRA fine-tune** (~5-10h): r=16, 1 epoch on KMER 79-subject train split,
   instruction-format target outputs.
5. **Test-time online adaptation**: per-driver short LoRA tuning on
   first-3-minutes calibration drive.

## Datasets

- **KMER (primary, JY's lab)**: 79 subjects, 486GB, bio/video/audio.
  Dataset paper to be published before this system goes external.
- **Public benchmarks for validation**: AffectiveROAD, PPB-Emo,
  Scientific Data 2024 multimodal driver emotion.

## Novelty axes (IP claims)

1. **Bio-first multimodal LLM agent for driver context** — most prior work
   uses bio as auxiliary signal, not first-class modality.
2. **CBBF causal temporal fusion** for bio-behavioral lag (Lazarus 1991,
   Kreibig 2010 grounded) integrated with LLM token space.
3. **Agentic intervention output** — emotion classification + reasoning chain
   + actionable safety recommendation in single forward pass.
4. **Test-time LoRA per-driver calibration** — driver-specific session adapter
   trained in-vehicle from first few minutes.
5. **Korean cultural context priors** — same-lab Yonsei 298-person consensus
   data for label calibration.

## Commercial scope

- **Partner targets**: Hyundai Mobis, 42dot (Hyundai Motor Group spin-off),
  Samsung Harman, KIA, Bosch (Korea), LG Electronics (VS).
- **Product form**: in-cabin monitoring SDK + driver app.
- **Possible spin-out**: small startup at JY's master's graduation
  (~2027-01).
- **Patent filings**: bio-first multimodal driver affective agent
  architecture (1), CBBF causal fusion adaptation to in-cabin sensors (2),
  test-time per-driver LoRA (3).

## Deployment phases

- **Phase 0 (graduation)**: master's thesis includes this as future work
  description. KMER dataset paper drafted but not submitted.
- **Phase 1 (post-grad month 1-3)**: 42dot job acceptance OR seed company
  formation; KMER dataset paper submit.
- **Phase 2 (month 4-9)**: full system implementation, on-device demo.
- **Phase 3 (month 10-18)**: pilot deployment with partner OEM.

## Why this stays out of master's thesis scope

- KMER dataset paper required first (without it, system unverifiable).
- Full multimodal training infra exceeds 6-9 month graduation budget.
- IP clarity: this v2 plan is JY's own commercial/startup IP, not a
  thesis contribution.

## Files / assets referenced

- `/home/ajy/CBBF-main/` — CBBF causal fusion code (S-PACE lab asset).
- `/mnt/ssd2/KMER_Sensing_Backup/` — 486GB KMER raw data.
- `/home/ajy/claude-research-engine/projects/02_emotion_agent/` — methodology
  (Paper B/A foundation that v2 builds upon).
- S-PACE NormWear bio encoder weights (lab internal).

---

**End of spec — IP timestamp commit. This file may be removed from working
tree but retained in git history for priority record.**
