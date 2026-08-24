# PAPER v1 — strong framing (cross-checker 반영)

> 2026-06-11. AAAI_DRAFT_v0 supersede. emotion-forward + phenomenon-only claim +
> format-vs-task confound-control 중심 + 298-consensus anchor.
> [HARD] = 5-seed 하드닝 결과 들어오면 채움. 나머지는 확정 숫자.

## Title (phenomenon-only, mechanism 암시 제거)
**When Does Few-Shot Help? Format Novelty Gates In-Context Learning Gains in
Cross-Cultural Facial Emotion Recognition**

## Target
1차 forcing deadline: **AAAI 2027 (7/27)** — reach 제출(완성 강제용).
실착지(best-fit): **ACII 2027 / ACL Findings** — 같은 본문 recycle.

## Abstract (v1)
In-context learning (ICL) is widely treated as a general lever for adapting LLMs.
We show its benefit is sharply conditional in a cross-cultural facial-emotion
setting. Using Korean facial Action-Unit (AU) intensities — validated against a
298-annotator Yonsei consensus study — rendered as text, versus two familiar
English dialog benchmarks (IEMOCAP, MELD), under a matched 4-class protocol on
three LLMs (Qwen2.5-7B/14B, Mistral-7B), we find: (1) **few-shot ICL helps an order
of magnitude more on the unfamiliar AU input (+10 to +16.7pp) than on familiar
dialog text (≤5.5pp)** — contrast significant (Mann-Whitney p=0.012). We then
rule out the obvious explanations: the gain is **not** input perplexity (AU is the
*lowest*-perplexity input; Spearman(NLL,gain) rho=-0.75) and **not** headroom
(IEMOCAP has +22pp LoRA headroom yet ~0 ICL gain). A controlled
**format-vs-content** manipulation — the same facial signal as alien numeric codes
vs natural prose — isolates surface format as a contributing factor [HARD: Δ and
model-consistency]. We report honestly that a representation-compaction mechanism
does *not* survive cross-model replication, leaving the precise mechanism open.
Western FACS (Ekman) prototypes perform at chance (25.4%) on Korean faces,
underscoring the cultural specificity. Our contribution is a clean characterization
of *when* few-shot pays off, with the trivial explanations eliminated.

## Contributions
1. A robust **gating phenomenon**: ICL benefit is ~3–15× larger on input formats the
   LLM lacks a task-mapping for, replicated across 3 models / 2 families / 2 scales.
2. **Elimination of trivial explanations**: not perplexity (neg, rho=-0.75), not
   headroom (IEMOCAP LoRA +22pp vs ICL ~0).
3. A **format-vs-content control** (same facial signal, numeric vs prose) isolating
   surface format [HARD].
4. **Honest negative**: representation-compaction does not explain it cross-model
   (rho=0.31, ns) — mechanism left open rather than overclaimed.
5. **Cross-cultural anchor**: 298-consensus-validated Korean AU labels; Ekman ≈ chance.

## Section plan
- §1 Intro: practitioners assume ICL is general; we show it is format-gated, in a
  culturally-grounded emotion testbed with human-validated labels.
- §2 Related: ICL when-does-it-work; cultural bias in emotion (Jack 2012, Ekman);
  PEFT vs ICL. (LLM provider refs via claude-api if needed.)
- §3 Setup: 3 inputs (AU novel / IEMOCAP / MELD familiar), 4-class, 3 LLMs, 4-bit,
  n≥7 seeds [HARD], N=400. **298-consensus validation of AU labels = credibility.**
- §4 Results:
  - 4.1 Gating (novel +14.0 mean vs familiar +2.3, p=0.012). Table+Fig.
  - 4.2 Not perplexity (rho=-0.75) — negative control.
  - 4.3 Not headroom (IEMOCAP LoRA vs ICL).
  - 4.4 Format-vs-content control [HARD] — centerpiece ablation.
  - 4.5 Ekman≈chance on Korean (cultural).
- §5 Discussion: format-novelty as a *contributing* (not sole) driver; compaction
  ruled out; what "task-mapping familiarity" might mean. Honest limitations.
- §6 Limitations: single domain, n small (mitigated), mechanism open, prompt
  sensitivity reported transparently.

## 확정 숫자 (지금 보유)
- Gating: qwen7b AU+10.1/ie+1.0/meld+0.7; mistral +15.2/+2.1/+0.0; qwen14b +16.7/+5.5/+4.7
- contrast Mann-Whitney p=0.012; novel mean +14.0 vs familiar +2.3
- perplexity: AU 최저 NLL; pooled Spearman(NLL,gain) rho=-0.75 p=0.02
- compaction null: Spearman rho=0.31 p=0.54
- format-swap 3-seed (예비): qwen7b Δ+9.7 / mistral Δ-3.5 → [HARD] 5-seed로 결판 중
- Ekman on Korean ≈ 25.4%

## 남은 완성 작업
1. [HARD] 5-seed format-swap (running) → §4.4 확정 + Mistral 역전 노이즈/실재 판정
2. n≥7 seed gating 보강 (CI tighten) — 선택, light
3. figure 2장: Fig1 개념도(novel vs familiar + format-swap), Fig2 게이팅 막대
4. LaTeX (AAAI 2-col template) 본문 작성
5. 298-consensus 인용/표 1줄 anchor 삽입

---
# OVERNIGHT CONSOLIDATED RESULTS (2026-06-12 04:xx) — 본문 backbone

## 재정의된 thesis (밤샘 후 — 더 정밀)
> **In-context demonstrations help an LLM RECOGNIZE (not learn) a latent task when an
> unfamiliar input FORMAT blocks cold zero-shot access.** ICL gain은 (a) 모델이 task를
> latent하게 알고 + (b) 입력 포맷이 novel일 때만 크다. ~75% Task-Recognition (random-label robust).

## 핵심 결과 (전부 우리 실험, 4-bit, batched)
1. **게이팅 일반화 (exp_039, no-FACS, 6모델):** AU(novel) ICL gain이 Qwen-3B/7B/14B(+6.7/+16.4/+14.5), Yi-6B(+10.5), Falcon-7B(+14.7)에서 familiar 대화(≤~5)보다 훨씬 큼. **5/6 모델·3 패밀리.** Mistral만 예외(+1.2).
2. **메커니즘 = Task-Recognition (exp_040):** AU gain의 ~75%가 TR(random-label robust), TL 아님. qwen7b 90%/14b 75%/yi 63%/falcon 70%. (Pan&Gao 2023 / Min 2022 framework). 모델이 AU→감정을 가중치엔 알지만 낯선 포맷이라 cold 못 꺼냄 → 예시가 인식 unlock.
3. **예측자 = tokenizer fertility (exp_042):** 포맷이 토큰으로 쪼개지는 정도가 gain을 양으로 예측(Pearson r=0.88, p=0.002). perplexity는 음(−0.75)으로 실패 → fertility가 format-level novelty를 잡음.
4. **2nd 도메인 (exp_045, sentiment-leetspeak):** sentiment(LLM이 아는 task)+leetspeak(alien). natural zero92→gain+2.7 vs leet zero58→gain+5.7(qwen7b), qwen14b도 delta+1.5. **감정-AU 밖에서 재현.** TR-driven.
5. **경계조건 control (exp_044, exp_043):** diabetes(모르는 task)를 alien 포맷으로 줘도 gain 0/음(ICL 해침). 표데이터 자연어 직렬화(낮은 fertility)도 gain 0. → **둘 다 필요(2×2):**

| | familiar 포맷 | novel/alien 포맷 |
|---|---|---|
| **latent-known task** | gain 작음 (영어대화, 자연어 sentiment) | **gain 큼 (AU, leet-sentiment)** |
| **task 모름** | — | gain 0/음 (alien diabetes) |

## 배제 (negative controls)
- perplexity (음의상관), headroom (IEMOCAP LoRA+22 vs ICL~0), representation-compaction (멀티모델 null).

## 차별화 (positioning)
- Pan&Gao 2023 (TR/TL): 익숙한 NL만. 우리=novel FORMAT + fertility 예측자 + Mistral negative.
- IsoBench COLM2024: same-task different-repr 문서화하나 random-label ICL 안 함 — 우리가 메커니즘+recovery 추가.
- Ciphered-reasoning (Anthropic 2025): 너무 alien하면 recovery 실패 = 우리 "recoverable window" 상한.
- Mistral null 설명: pretraining corpus 구성(Shin NAACL'22) / model-selection-within-mixture(Yadlowsky 2023).

## 정직한 한계
- 단일 task가 아님(2도메인)이나 leet 효과는 AU보다 작음(+1.5~5.7 vs +10~17). Mistral 예외. n=3~5 seed. 단일 언어/문화 testbed.

## 확률 (정직, 최종)
AAAI main ~30-33% / **ACII·ACL-Findings ~62-67%**. figure: figA_gating_6model · figC_mechanism_TRTL · figB_perplexity_control.
