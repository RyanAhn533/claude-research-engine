# Insights — 03_ht_dar (MOSEI stripped ladder 재현)

## exp_001_mosei_stripped_ladder (2026-06-03)

### 재현 셋업
- 데이터: CMU-MOSEI `aligned_50.pkl` (Self-MM 공개 미러, SHA256 검증 `45eccfb7…`), text BERT768/audio74/vision35, seq50
- 코드: HT-DAR-main.zip → `/home/ajy/HT-DAR`, base env(torch2.6/transformers4.50), BERT 로컬
- 누락 패치: pynvml 설치, 최상위 `utils.py` shim
- env/data/git 해시 → reproducibility_manifest 기록

### 단일시드(1111) 사다리 — reference 정확 재현
| 변형 | best-val test acc_2 | reference |
|---|---|---|
| M0 baseline | 0.8525 | 0.8504 |
| **M1 Hier** | **0.8608** | 0.8559~0.8609 |
| M2 DAR only | 0.8539 | 0.8539 ✓ |
| M3 Hier+DAR | 0.8550 | 0.8550 ✓ |
| M4 full | 0.8528 | 0.8528 ✓ |
→ M2/M3/M4 소수점 4자리까지 일치. (단일seed라 leaderboard 미등재, exploratory_single_seed.json)

### W1. ★ 5-seed에서 M1의 우위가 사라진다 (단일seed는 운)
- M0 5-seed: 0.8528 ± 0.0036
- M1 5-seed: 0.8524 ± 0.0066  (s1114=0.8404 큰 outlier)
- 단일seed M1=0.8608은 lucky seed였음.
- **Gate C (M1 vs M0, paired n=5)**: Δ=-0.0004, bootstrap CI95=[-0.0099,+0.0060](0 포함),
  Cohen d=-0.04, Wilcoxon p=0.875, paired-t p=0.931, TOST±0.5pp p=0.196(동등성도 미입증)
- verdict=**neutral** → H_htdar_m1 outcome=**neutral**
- **교훈**: 단일seed로 ablation ranking 신뢰 금물. HT-DAR의 "MOSEI 게인은 noise 범위"
  honest disclosure를 재확인(오히려 더 강하게 — nominal 우위조차 없음).

## exp_002_iemocap_stripped_ladder (2026-06-03)

### W2. ★ IEMOCAP도 within-noise (MOSEI와 동일 결론)
- 데이터: MulT iemocap_data.pkl (Dropbox Archive), [300,74,35] seq20, 4-class
- 5-seed test wF1: M0 0.5820±0.011, M1 0.5827, M2 0.5839, M3 0.5799, M4 0.5898±0.018, A1 0.5904±0.019
- 단일시드(1111)에선 M4=+1.7pp로 우세해 보였으나 5-seed에서 +0.78pp로 축소
- Gate C: M4 vs M0 t_p=0.50, A1 vs M0 t_p=0.57 → 둘 다 **neutral**
- reference IEMOCAP(A1 +1.13pp p=0.32 INCONCLUSIVE)과 동일
- **교훈**: MOSEI(텍스트지배)뿐 아니라 IEMOCAP(균형)에서도 fusion delta가 noise.
  → 표준 벤치 정확도 논문 불가 확정. K-EmoCon(bio) 또는 reframe만 남음.

### F1. MOSI stripped misconfigured
- test acc_2 0.57 고정 (표준 0.84). reference도 MOSI stripped 미사용. config 미튜닝. 보류.

### 종합
- 표준 MSA 벤치마크(MOSEI/IEMOCAP) 둘 다 HT-DAR가 baseline 통계적으로 못 이김.
- 절대 성능은 SOTA band. 구조는 novel. 하지만 정확도 gain은 within-noise.
- 다음 결정: K-EmoCon HT-DAR 구현(유일한 "이긴다" 후보, Phase-D 신규) vs reframe(robustness/해석/효율).

### Open
- MOSEI M2/M3/M4 5-seed (백그라운드 진행중) → 완료시 ladder 전체 Gate C
- K-EmoCon Phase-D 구현 여부 (JY 결정 대기)
- 상세 종합: SESSION_REPORT_2026-06-03.md
