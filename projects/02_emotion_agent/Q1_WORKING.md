# Q1 Paper Working Draft — Emotion Agent

> Live drafting. 실험 결과 쌓이면 §4 채움.

## Status (2026-04-23)
- Phase 0 setup ✓
- Phase 1 Week 1 (preprocessing): IEMOCAP, MELD, K-EmoCon ✓
- Phase 2 Week 2: Agent prototype + IEMOCAP/MELD eval ✓
- K-EmoCon bio-proxy eval: running
- Emotion-LLaMA/M3NET reproduction: Week 3+

## Draft Title
*"Culturally-aware Multimodal Emotion Agent with Bio-grounded Prompting: An ablation study across Western and Korean datasets"* (가제)

## Abstract skeleton

We present an LLM-driven emotion agent (Qwen2.5-7B, 4-bit quantized) that injects
culturally-situated priors and observable bio-signal proxies into the reasoning loop.
We evaluate on IEMOCAP (English conversational), MELD (English TV dialogue), and
K-EmoCon (Korean debate, bio-proxy). Without any fine-tuning, the agent achieves
competitive zero-shot emotion classification, with cultural-prior injection yielding
statistically distinct behavior between English and Korean subsets.
[To be completed]

## Results table (partial)

| Dataset | Config | Acc | Macro-F1 | N |
|---------|--------|-----|----------|---|
| IEMOCAP | TF-IDF+LR (baseline, trained) | 66.33 ± 0.84 | 0.640 ± 0.007 | 80/20 × 3 seed |
| IEMOCAP | Agent zero-shot baseline | 48.75 | 0.484 | 400 stratified |
| IEMOCAP | Agent + cultural (KO prior) | 48.50 | 0.485 | 400 stratified |
| IEMOCAP | Agent + prosody | 48.00 | 0.479 | 400 stratified |
| IEMOCAP | Agent full | 48.25 | 0.485 | 400 stratified |
| MELD    | TF-IDF+LR (baseline, trained) | 59.75 | 0.312 | official |
| MELD    | Agent zero-shot baseline | 55.00 | 0.540 | 400 stratified |
| MELD    | Agent + cultural (KO prior) | 55.75 | 0.552 | 400 stratified |
| MELD    | Agent + context (prev2) | 55.00 | 0.544 | 400 stratified |
| MELD    | **Agent full (+cultural+context)** | **58.50** | **0.581** | 400 stratified |
| K-EmoCon | Agent bio-proxy baseline | (running) | — | — |
| K-EmoCon | Agent bio-proxy + cultural | (running) | — | — |

Random 4-class baseline: 25%.

## Key findings so far

1. **English datasets (IEMOCAP/MELD)에서 Korean cultural prior는 null 또는 약한 positive**.
   - IEMOCAP: ±0.25%p (null, 예상대로).
   - MELD: +0.75%p alone, +3.5%p with context (synergy).

2. **Agent macro-F1 >> TF-IDF baseline F1** on MELD (0.58 vs 0.31).
   - Class imbalance 상황에서 LLM reasoning이 class floor 방어.

3. **Zero-shot accuracy는 trained baseline보다 낮음** (IEMOCAP 48 vs 66).
   - LoRA fine-tune 필요 (GPU 여유 시 Week 3+).

## Planned sections

§1 Introduction — LLM agent + culturally-situated emotion recognition gap
§2 Related Work — Emotion-LLaMA (NIPS24), M3NET (CVPR23), MER-in-Conv survey (EMNLP25), EEG Copilot (NeuralNet25)
§3 Method — 4-bit Qwen agent with cultural/bio-proxy prompt injection, inductive design
§4 Experiments —
   §4.1 Setup (IEMOCAP/MELD/K-EmoCon)
   §4.2 Main results (table above)
   §4.3 Ablation: cultural prior × dataset origin
   §4.4 Agent reasoning quality (qualitative)
§5 Discussion — cultural match condition, multi-layer cultural gap (project 01 reference)
§6 Limitations — zero-shot only (no fine-tune), Korean text missing
§7 Conclusion
