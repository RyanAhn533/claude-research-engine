# exp_002 — Fixed-DAR router re-run (MOSEI stripped ladder)

status: running
date: 2026-06-03
supersedes_runs: exp_001 M2/M3/M4 (broken-DAR Co-Belief)

## 배경 — 왜 재실행하나
exp_001의 M2/M3/M4 (DAR 변형) 결과 = "within-noise"는 **고장난 라우터**로 얻은 것.
코드 교차검증에서 3가지 결함 발견·수정:

1. **Co-Belief Holo 항 사망**: `c_holo`가 모달리티 무관 글로벌 스칼라 1개를 M축으로
   broadcast → `softmax_m(γ·c_mono + (1-γ)·c_holo)`에서 상수항이 정확히 상쇄.
   Holo-Confidence 브랜치가 라우팅에 기여 0 (dead code). README 수식 `-KL(p_m‖p̄)`
   미구현. → 모달리티별 KL로 복구.
2. **c_mono ≠ 스펙**: 학습 스칼라 `Linear(d,1)`. README는 `max_k softmax(MLP_m)_k`.
   → K-class 라우팅 분포 → max prob 으로 수정.
3. **라우터 grad vanishing**: cobelief 모달간 차이 ~1e-3 → softmax 균일 붕괴 →
   라우터 grad ~1e-7로 동결(init 균일에 영영 갇힘). → cross-modal 정규화(unit spread)
   + learnable temperature. 검증: mono_mlp grad 0.0→3.67, r std 0→0.45.
4. **per_modality 무력화**: M2/M3/M4가 `output_mode='per_modality'` →
   modality m 출력 = `r[m]·pooled_anchor[m]` (자기 스칼라 곱) → downstream LayerNorm이
   양의 스칼라 곱을 정규화로 제거 → r이 출력에 인과영향 ~0. → `mixed`-mixture
   `broadcast` (`H_mix=Σ_m r_m·anchor_m`, r이 출력 지배)로 전환.

코드: `DLF_HTCMA.py` DynamicAnchorCMA.__init__/forward, train_stripped.py EXP_CFG.

## hypothesis (재등록 필요 — broken-DAR verdict와 분리)
- **H_htdar_m2_fixed**: fixed-DAR M2(DAR only, mixed-broadcast) > M0 on MOSEI Acc2, ≥0.5pp.
  null: M2 - M0 <= 0.
- **H_htdar_m3_fixed**: fixed-DAR M3(Hier+DAR) > M0, ≥0.5pp.
- **H_htdar_m4_fixed**: fixed-DAR M4(full) > M0, ≥0.5pp.
- delta_threshold 0.005, min_seeds 5, paired bootstrap CI + Cohen d + Wilcoxon + TOST.

config_fingerprint: M2/M3/M4 모두 NEW (code_version=fixed-dar-router-v1, output_mode=broadcast).
옛 fingerprint(CONFIG_FINGERPRINTS.json M2/M3/M4)는 broken-DAR 전용으로 보존.

## 예상 (pre-registered 직관)
- MOSEI는 text 지배 → 라우터가 제대로 작동해도 거의 항상 text를 고를 것 → M2/M3/M4가
  M0/M1과 여전히 tie일 가능성 높음. 그래도 **이번엔 "라우터가 안 돈 게 아니라, text가
  지배적이라 동적 라우팅이 필요없다"**는 깨끗한 해석 가능 (dar_r 분포로 입증).
- 진짜 이득은 modality 지배가 불안정한 K-EmoCon(Phase D)에서 기대.

## 검증 포인트 (결과 나오면)
1. 실데이터에서 dar_r이 균일(0.33) 탈출하는가 — 라우터 작동 증거.
2. MOSEI Acc2 5-seed: M0 대비 fixed-DAR M2/M3/M4 delta + 통계.
3. broken-DAR(exp_001) vs fixed-DAR(exp_002) 대조 — 논문 ablation 재료.

## 로그
- 실행: `run_m234_fixed_dar.sh` (M2/M3/M4 × seed 1111-1115)
- 로그: `HT-DAR/code/logs_run_fixed/` (옛 broken `logs_run/`과 분리)
