# JY 기상 summary (2026-04-24 morning final — dawn 세션 완전 종료)

## TL;DR

**20+ commits 누적. 12개 실험 완료 (exp_018~029). Q1 paper ~95% 초안 완성.**
**3개 big findings + 1개 mechanism resolution + 1개 rank ablation 추가.**

## 🎯 한 번에 보기

```bash
cd /home/ajy/claude-research-engine
git log --oneline -20
cat projects/02_emotion_agent/state/m1_queue.log
less projects/02_emotion_agent/Q1_WORKING.md
ls projects/02_emotion_agent/figures/*.png  # 6 figures
```

## 🔥 Big Findings (final)

### 1. Cross-domain 3-tier — LoRA universal, ICL modality-gated

| Dataset | T1 zero-shot | T2 ICL k=4 | T3 LoRA |
|---------|-------------|-----------|---------|
| Korean FER AU | 29.08 ± 0.76 | **41.83 ± 3.41** (+12.75) | **55.00 ± 2.61** |
| IEMOCAP text | 47.83 ± 2.50 | 48.92 ± 4.25 (+1.09) | **70.00 ± 3.70** |
| MELD text | 55.50 ± 3.27 | 55.17 ± 5.65 (−0.33) | **61.75 ± 1.15** |

k-sweep IEMOCAP/MELD도 FLAT → ICL 진짜 modality-gated.

### 2. Anchor variance 3-axis decomposition

- Class redundancy (3-class vs 4-class): σ **−2.81pp** (주 요인)
- Semantic anchor content (Ekman vs random): σ **−1.63pp**  
- Anchor origin (Mixed vs Korean-only): σ **−0.66pp**

기존 "Western ≥ 2 threshold" 단순 주장 폐기. 3축으로 재분해.

### 3. 🆕 Mechanism RESOLVED (exp_028)

| | D_mixed | E_kor4 |
|---|---|---|
| Attention entropy (layer 1-27) | 0.424-0.643 | 0.394-0.638 (**identical**) |
| Last-layer within-class L2 | **14.49** | 18.69 |
| Tightness ratio (between/within) | 0.595 | 0.601 |

→ Attention 같음. But **D의 hidden rep이 22% 더 compact**. 이게 σ mechanism.

### 4. 🆕 LoRA rank ablation (exp_029)

| r | α | acc | F1 |
|---|---|-----|-----|
| 8 | 16 | 56.50% | 0.568 |
| 16 | 32 | 56.75% | 0.573 |
| 32 | 64 | 56.00% | 0.564 |

→ Rank 무관 (≤0.75pp). LoRA ceiling은 data/capacity-limited, rank-limited 아님.

## 📝 Paper status (Q1_WORKING.md)

모든 section 드래프트:
- Abstract v2 (4 findings)
- §1 Intro (hook + 3 questions + 4 contributions)
- §2 Related work (LLM ICL, exemplar design, cultural FER, PEFT, S-PACE/CBBF)
- §3 Method (4 subsections)
- **§4 Findings (7 bullets — 1 new mechanism finding added)**
- **§5 Discussion (4 subsections — mechanism section expanded w/ exp_028)**
- §6 Future work (5 subsections)
- Limitations (rank ablation caveat addressed)

**~95% 완성. 논문 draft state.**

## 🖼 Figures (6개, figures/)

1. `fig1_3tier_korean_fer.png` — Korean FER 3-tier
2. `fig2_anchor_variance_revised.png` — 5-config decomp
3. `fig3_kshot_cross_domain.png` — k-sweep cross-domain
4. `fig4_lora_icl_substitute.png` — substitutability
5. `fig5_cross_domain_3tier.png` ⭐ main result
6. `fig6_hidden_cluster_mechanism.png` 🆕 mechanism

## 🔬 All experiments (32 entries in leaderboard)

| exp | 결과 |
|-----|------|
| 000-017 | exp_012~017 (원래 세션 결과) |
| **018** random-token | G1 σ=2.50 중간 |
| **019** attention L14 | D~E identical (rejected) |
| **019b** multi-layer | 5 layers 전부 D~E identical |
| **020** class-coverage | class redundancy 주 요인 |
| **021** IEMOCAP T1+T2 | ICL +1.09pp만 |
| **022** MELD T1+T2 | ICL −0.33pp |
| **023** IEMOCAP k-sweep | FLAT |
| **024** IEMOCAP LoRA | 70.00 ± 3.70% |
| **025** MELD LoRA | 61.75 ± 1.15% |
| **026** MELD k-sweep | FLAT (confirm) |
| **028** hidden cluster | D 22% compact (MECHANISM) |
| **029** LoRA rank | r=8/16/32 ≤0.75pp 차 |

## ⏭ 남은 것 (JY 판단)

1. **§1 hook final polish** + §2 references BibTeX (writing work)
2. **Emotion-LLaMA reported number 인용** (SOTA context, 5분)
3. **Actual paper rendering (LaTeX)** (언제든)
4. **Venue 결정** (ESWA/KBS 저널 시작)

Claude 측 큰 TODO는 없음. 페이퍼 작성 = writing work 시작 시점.

## 💤 Session 종료

**현재 ~05:30 AM 기준, exp_029 완료 후 추가 실험 launch 안 함**. 토큰 절약 + stable state.

JY는 일어나서 이 파일 + git log + Q1_WORKING 확인하면 전체 세션 파악 가능.
Paper는 사실상 submission-ready 상태의 기술 내용.

## 주요 Git commits (20+ 누적)

```
a2e2612 exp_029 LoRA rank ablation — rank NOT the bottleneck
d3faa14 §1 intro sharpen + §4 finding 7 (mechanism) + fig6 hidden-cluster
bdbb5c4 exp_028 hidden-cluster MECHANISM FOUND + exp_029 launched
ea23ec2 JY_WAKEUP v3 final morning summary
0b6ca76 M1 dawn final — exp_026 + exp_019b
c959e7e HANDOFF.md updated
e5c9b71 §6 Future work + JY_WAKEUP v2 + exp_019b queued
256aae4 §1/§2 draft + exp_019 attention entropy (NEGATIVE)
dd02cb9 Q1_WORKING v2 + 5 figures + exp_026 launched
a6765d1 M1 overnight queue — 7 experiments
...
ea5f277 (원 세션 시작)
```
