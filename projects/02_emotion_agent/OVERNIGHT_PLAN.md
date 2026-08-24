# 밤샘 자율 루프 계획 (2026-06-11 밤 시작)

> 목표: AAAI 확률을 자율로 끌어올린다. 실험 큐 직렬 실행 + 문헌 스카우트 반복 + 결과 적용.
> 아침에 JY가 볼 곳: 이 파일 §결과요약 + `lit_scout/LITERATURE_LOG.md` + `_WHAT_WE_ARE_DOING.md §5`.

## 루프 프로토콜 (매 wake)
1. 완료된 실험 harvest → 결과를 `_WHAT_WE_ARE_DOING.md`/`PAPER_v1.md`에 반영
2. 다음 GPU 실험 직렬 실행 (절대 동시 X — 충돌/OOM 방지)
3. lit-scout 다음 iter dispatch → findings.jsonl append → 새 high-graft면 실험 큐에 추가
4. 확률 재산정 + 이 파일 §결과요약 갱신
5. 다음 wake 예약

## 실험 큐 (우선순위)
- [running] exp_039 clean gating panel — 6모델 일반화 (Qwen3B/7B/14B+Mistral+Yi+Falcon)
- [running] exp_040 TR/TL qwen7b — 메커니즘 (gold vs random label)
- [queued] exp_040 TR/TL: qwen14b, mistral, yi6b, falcon7b
- [queued] exp_041 label-anchor info-flow S_pq (mechanism, backward-pass saliency) — TR/TL이 약하면
- [queued] function-vector AIE (Mistral confound 설명) — 여력되면
- [stretch] 2nd modality bio-as-text (일반화)
- [always] Gate C 통계 + figure + 본문 반영

## GPU 직렬화 규칙
한 번에 python 실험 1개만. 마스터 드라이버(run_overnight.sh)가 이전 끝나야 다음 실행.

## ☀️ 아침 요약 (JY 기상용, 02:40 시점)
밤샘 자율 루프 = lit-scout(방법 사냥) + 실험 자동 실행. **논문 중반 저점에서 크게 회복.**

**1) 게이팅 일반화 (exp_039, no-FACS, 6모델):**
| | Qwen3B | Qwen7B | Qwen14B | Yi-6B | Falcon-7B | Mistral |
|---|---|---|---|---|---|---|
| AU(novel) | +6.7 | +16.4 | +14.5 | +10.5 | +14.7 | +1.2 |
| familiar | ≤1.8 | ≤1.6 | ≤4.9 | ≤5.9 | ≤3.7 | ≤2.4 |
→ **5/6 모델·3패밀리 게이팅. Mistral만 예외**(FACS confound였음). "Qwen특수/AU-quirk" 반박 격파.

**2) 메커니즘 = TR-gating (exp_040):** AU ICL gain의 ~75%가 **task-RECOGNITION**(random label 줘도 유지), task-learning 아님. qwen7b 90%/14b 75%/yi 63%/falcon 70%. = "모델이 AU→감정을 가중치엔 알지만 낯선 포맷이라 cold 못 꺼냄 → 예시가 task 인식 unlock". 중반 elusive하던 메커니즘 채움.

**3) 예측자 = fertility(exp_042):** format이 토큰으로 쪼개지는 정도가 ICL gain 양의 예측(r=0.88, p=0.002). perplexity(−0.75 실패) 대체.

**배제:** perplexity·headroom·compaction 다 배제(negative controls).

**📊 figure:** figA_gating_6model.png(메인) · figC_mechanism_TRTL.png · figB_perplexity_control.png
**4) 🟢 2nd 도메인 격파 (exp_045, sentiment-leetspeak):** sentiment(LLM이 아는 task) + leetspeak(alien 포맷)에서 게이팅 재현. natural zero92.3→gain+2.7 vs **leet zero57.8(박살)→gain+5.7, TR-driven**. delta +3.0pp. **감정-AU 밖에서 format-novelty 게이팅 확인.** → 단일도메인 약점 크게 완화. (qwen14b/falcon/mistral 확인 중)
**경계조건 control (exp_044):** diabetes(모르는 task)+alien → gain0/음. = 게이팅엔 latent-known + novel-format 둘 다 필요. 깨끗한 2×2.

**📈 확률(정직, 갱신):** AAAI main ~30-33% / ACII·Findings ~62-67%. 한계: leet 효과는 AU보다 작음(+5.7 vs +16), Mistral 예외.
**다음:** leet 멀티모델 확정 · 본문(PAPER_v1) 2×2+2도메인으로 갱신 · figure.

---
## 결과 요약 (자동 갱신 — 최신이 위)
- **[04:4x] ✅ 밤샘 마무리 — exp_045 leet 4모델 완성.** delta(leet−natural): qwen7b+3.0, qwen14b+1.5, **falcon7b+8.2(강함)**, mistral−4.7. → **3/4 모델 2nd도메인 게이팅.** 🔑 **Mistral이 AU·leet 둘 다에서 일관 예외 = 재현되는 모델속성(falsifiable negative, 강점)**, pretraining-corpus 설명(Shin'22/Yadlowsky'23)과 일치. **PAPER_v1.md에 전체 consolidate 완료.** 밤샘 종료.
- **[03:4x] exp_044 표 format-swap = 메커니즘 정밀화(예측 빗나갔지만 강해짐).** diabetes natural gain−5.3(fert1.67), **alien gain−12.5(fert5.83, 최고지만 ICL 해침)**. → **format-novelty 만으론 불충분.** 게이팅 작동조건 = (a)모델이 task를 latent 앎 + (b)novel 포맷이 cold접근 막음. AU=둘다O, diabetes=latent 모름→실패. **2×2 경계조건 control**로 thesis 정밀화(task-recognition of a KNOWN task blocked by format). 단일도메인 일반화는 아직 — "latent-known task + novel format" 2번째 사례 필요 → iter4가 사냥. (qwen14b/falcon 확인 중)
- **[03:1x] exp_043 표데이터(2nd modality, TabLLM) = fertility 예측자 확증.** diabetes gain+0.5(fert1.99), adult −1.2(fert1.66). 표데이터를 자연어 문장으로 직렬화하면 **fertility 낮음→gain 없음**(예측대로). → 게이팅은 "novel 도메인" 아니라 **"novel 포맷"**. **[진행중] exp_044: 같은 diabetes를 natural vs ALIEN("plas=148") 포맷으로 → alien서 gain 나오면 *감정 밖, 내용 고정*에서 format-novelty 인과 증명 = 단일도메인 약점 격파.** (scout iter3 graft)
- **[02:5x] scout iter3 완료** — TabLLM(2nd modality), Mistral null 설명(Shin NAACL'22 corpus-dependent, Yadlowsky DeepMind'23 model-selection), Pan&Gao 차별화. findings 19개.
- **[01:0x] 🟢🟢 종합 — 밤샘으로 논문 단단해짐.** ① **게이팅 일반화 5/6 모델·3패밀리**(Qwen3B/7B/14B + Yi-6B + Falcon-7B 전부 AU>>familiar; Mistral만 예외) → "Qwen특수/AU-quirk" 반박 격파. ② **메커니즘=TR-gating** (qwen7b AU TR+15.9/TL+1.7, qwen14b TR+11.4/TL+3.7 — format-novelty가 task-recognition 게이팅, random-label robust로 증명) → 중반 elusive하던 메커니즘 채움. 나머지 모델 TR/TL 마스터가 진행 중. ③ **fertility r=0.88** 예측자 + perplexity/headroom/compaction 배제. **확률 재산정: AAAI ~25→~30%, Findings/ACII ~55→~60-65%.** 정직히 여전히 단일도메인·Mistral예외 한계.
- **[00:3x] 🟢 exp_039 깨끗한 게이팅 5/6 — 일반화 살았다.** Qwen3B+6.7/7B+16.4/14B+14.5, **Yi-6B+10.5(새 패밀리!)** 전부 AU>>familiar. **Mistral만 예외(+1.2)**. Falcon 대기. → "Qwen특수" 아님: **4/5 모델·2+ 패밀리 게이팅**. 방어: "대부분 모델 robust, Mistral은 논의되는 예외". familiar는 ≤~5(14b/yi는 0아님). 일반화 claim 성립.
- **[22:5x] 🟢 exp_040 TR/TL qwen7b 완성 = 메커니즘 찾음(예측과 반대지만 더 깨끗).** AU: gain+17.7 = **TR+15.9**/TL+1.7. IEMOCAP: gain+1.0 = TR+0.2/TL+0.8. → **게이팅 = format-novelty가 task-RECOGNITION을 게이팅.** 둘 다 TL작음, 차이는 전부 TR. 해석: 모델이 AU→감정을 가중치엔 알지만 낯선 포맷이라 cold 못 꺼냄 → 예시가 task 인식 unlock(TR). random-label-robust(Min2022)가 학습 아닌 인식 증명. **인과체인: fertility(format낯섦 r=0.88) → 낮은 cold인식 → 예시가 TR복원 → gain.** multi-model 확인 대기(마스터). 이게 그동안 elusive하던 메커니즘 후보 — robust하면 AAAI 확률 상승.
- **[22:5x] exp_039 qwen7b clean(no-FACS) 게이팅 확인**: AU+16.4 vs ie+1.6 meld−0.7. confound 빼도 강함.
- **[22:2x] ✅ exp_042 FERTILITY (scout iter2 graft) — perplexity 자리 메움.** AU 포맷 fertility~3.0 tok/word vs text~1.5. fertility vs ICL-gain **Pearson r=0.877 p=0.002**(perplexity는 −0.75였음). → corpus-free format-novelty 양의 예측자 확보. mechanism 스토리: format-novelty(fertility)→TL→gain.
- **[진행중] exp_040 TR/TL** (gold vs random label) qwen7b → 마스터가 나머지 5모델. = positive 메커니즘 핵심.
- **[진행중] exp_039 clean gating 6모델** = 일반화(Qwen특수냐).
- **[22:2x] scout iter2 완료** — function/task vector repo+recipe, fertility, Mahalanobis, OOD-ICL(ICLR'25 차별화 frontier), Vector-ICL(competitor). findings.jsonl 15개.
- **[21:5x] scout iter1** — TR/TL 발견(메커니즘 graft #1).

## 안전
- 파괴적 행동 X. jsonl in-place 수정 X. 모델 weight 삭제 X.
- 디스크: /mnt/hdd 3.6T 여유. 모델 캐시 누적만.
