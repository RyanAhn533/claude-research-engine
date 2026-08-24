# AAAI 2027 Draft v0 — Reframed for top-tier

> Vehicle: 02_emotion_agent. Thesis 단일화: **modality-gated ICL + 표현압축 메커니즘**.
> [MM] 표시 = 멀티모델 실험 결과 필요(현재 placeholder). 나머지는 기존 데이터로 작성 가능.

---

## Title (candidates)
1. **When Does In-Context Learning Help? Input Novelty Gates ICL Benefit via Representation Compaction**
2. *Modality-Gated In-Context Learning: ICL Helps Only on Inputs Novel to the LLM*

→ 감정/문화는 부제·testbed로. headline은 일반 ICL 명제.

## Abstract (v0)

In-context learning (ICL) is widely assumed to be a general-purpose lever for
adapting large language models (LLMs) to new tasks. We show this is conditional:
**ICL helps only when the input format is novel to the model, and the benefit is
governed by whether the in-context examples compact the model's hidden
representations.** Using emotion classification as a testbed across three inputs —
Korean facial action-unit (AU) intensity text (novel to LLMs) and two English
dialog benchmarks, IEMOCAP and MELD (familiar) — under a multi-seed protocol, we
find: (1) ICL (k=4) yields +12.75 pp on the novel AU-intensity input but ≤1 pp on
familiar English text, with a flat k-sweep confirming ICL is inert, not merely
saturating. (2) The mechanism is representational, not attentional: attention
entropy is identical across all five probed layers between high- and low-benefit
configurations, while last-layer hidden states are 22% more compact (within-class
L2) exactly where ICL helps. (3) The causal direction is confirmed across domains:
on IEMOCAP, where ICL does *not* help, the same in-context examples *expand*
within-class distance (+12.7%) — i.e., compaction tracks benefit with the right
sign. (4) The novel-input ICL advantage replicates across model families and
scales — Qwen2.5-7B (+10.1pp), Mistral-7B (+15.2pp), Qwen2.5-14B (+16.7pp) on
the novel input vs at most +5.5pp on familiar text — ruling out a single-model
artifact (novel/familiar gain ratio 3.3–15×). When familiar-input ICL gain does
appear (Qwen2.5-14B, +5pp), it is label-bias correction, not new task
information. As a
domain consequence, we also show Western Ekman FACS prototypes perform at chance
(25.4%) on Korean faces, and that parametric adaptation (LoRA) is the only tier
that reliably exceeds well-engineered zero-shot prompts. Our results refine when
practitioners should expect ICL to pay off and give a measurable predictor —
representation compaction — for it.

## Contributions (top-tier framing)

1. **A gating law for ICL**: input novelty (to the base LLM), not task or scale,
   determines whether ICL helps — demonstrated with a +12.75 vs ≤1 pp split and a
   flat k-sweep on the familiar side.
2. **A measurable mechanism**: ICL benefit ⟺ hidden-representation compaction
   (within-class L2 ↓), with attention-level explanation explicitly ruled out
   (5-layer entropy identical).
3. **Cross-domain causal confirmation**: where ICL fails, compaction reverses sign
   (IEMOCAP +12.7% expansion) — mechanism predicts benefit, not just correlates.
4. **Multi-model generality**: novel-input ICL advantage replicates across 3 models /
   2 families / 2 scales (Qwen2.5-7B/14B, Mistral-7B); novel-vs-familiar gain ratio
   3.3–15×. Where familiar-input ICL gain appears (14B, +5pp) it is label-bias
   correction (verified: zero-shot over-predicts 'neutral', LABEL-parse rate 100%),
   not new information — so the contrast, not a strict "familiar=flat," is the claim.
5. **Application finding**: Ekman prototypes ≈ chance on Korean faces; LoRA is the
   only robustly-winning adaptation tier — a concrete recipe for grounded emotion LLMs.

## Section plan (what moves)

- **§1 Intro**: lead with the general ICL puzzle (when does ICL help?), not emotion.
  Emotion = clean testbed because it gives one genuinely novel input (AU text) and
  two familiar ones (dialog), matched task.
- **§3 Method**: keep agent + tiers; foreground the novel-vs-familiar input contrast.
- **§4 Results**:
  - 4.1 Gating result (novel +12.75 vs familiar ≤1, k-sweep flat). CORE.
  - 4.2 Mechanism: attention ruled out + compaction. CORE / main figure.
  - 4.3 Cross-domain reversal (IEMOCAP expansion). CORE.
  - 4.4 [MM] Multi-model replication. CORE for top-tier.
  - 4.5 Application: Ekman≈chance, LoRA-only-robust. SUPPORTING.
  - **demote**: 3-axis anchor variance → appendix.
- **§5 Discussion**: why novelty gates ICL (label-space grounding for unfamiliar
  syntax); compaction as the lever; practitioner predictor.

## Main figure (design)
이중 패널: (좌) 도메인별 ICL gain(pp) bar; (우) 같은 도메인별 within-class L2 변화(%).
두 패널 정렬 → "압축되는 곳에서만 ICL이 듣는다"가 한 눈에. IEMOCAP은 우측에서 +방향(확장)으로 reversal 시각화.

## Risk ledger (정직)
- 단일모델 → [MM] 없으면 desk-reject 위험. **#1 우선순위 실험.**
- N=400 test/seed 작음 → seed 3 + paired stat로 방어, 가능하면 N 확대.
- "AU-intensity text가 진짜 novel modality인가" 반론 → §5에서 LLM embedding 거리로 novelty 정량화(제안된 추가 분석, exp).
- task가 niche → headline을 ICL 일반 명제로 올려 완화(이미 반영).
