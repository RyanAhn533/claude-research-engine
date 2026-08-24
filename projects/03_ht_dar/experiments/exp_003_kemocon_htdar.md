# exp_003 — K-EmoCon HT-DAR: dynamic vs fixed anchor (MAIN conference result)

status: running (launched 2026-06-03 22:09)
hypotheses: H_htdar_kemocon_dar_arousal, H_htdar_kemocon_dar_valence (pre-registered, hash-chained)
target: ICASSP 2027 (~Sep 2026)

## 동기 (exp_001/002 → exp_003 논리)
- exp_001/002 (MOSEI/IEMOCAP): HT-DAR가 baseline과 within-noise tie. 이유 = **text 지배** →
  동적 앵커링이 결정할 게 없음 (라우터가 항상 text 고름).
- 따라서 thesis 검증의 정당한 전장 = **지배 modality가 없는 데이터**.
- K-EmoCon = video+audio+bio, **text 없음** → 동적 앵커링이 작동해야 할 곳.
- CBBF(KBS 저널)는 "bio가 앵커"(고정 bio-anchor + Beta gate)를 주장. 본 실험은 그걸
  "fixed vs dynamic anchor" 축의 한 점으로 흡수하고, **학습된 앵커**가 더 나은지 검증.

## 셋업
- 코드: `/home/ajy/02_multimodal_emotion/03_visionmer_v3_binary/`
  - `model_mm_v3_htdar.py` — V3 인코더(Vid/Aud/Bio)+BioCond+TSE+head 재사용,
    융합부(gate+PerceiverFusion)만 HT-DAR `HTCMABLF_Fusion`으로 swap.
    modality l/v/a = video/audio/bio.
  - `train_binary_all.py --htdar --attn_module {static|dar}` (6-fold GroupKFold,
    arousal+valence, BCE+MSE+CCC loss, CCC metric).
  - 실행: `run_htdar_kemocon.sh` → `logs_htdar/{static,dar}.log`, `ckpt_htdar_{static,dar}/`
- 데이터: `/mnt/hdd/ajy/datasets/kemocon_precessed_data` (segments_index.csv, 3577 seg)
- HF_HOME=/mnt/hdd/ajy/caches/huggingface (AST 캐시)

## ★ 라우터 수정 (exp_002에서 발견, 본 실험에 반영됨) — 코드 교차검증 결과
원본 DAR 라우터는 구조적으로 죽어있었음 (exp_001/SESSION_REPORT의 "DAR tie"는 이 buggy 라우터):
1. Co-Belief **Holo 항이 softmax에서 상쇄** (모달 무관 글로벌 스칼라 broadcast) → dead branch.
2. c_mono가 스펙(max softmax K-class) 아닌 학습 스칼라.
3. **라우터 grad vanishing** (cobelief 모달간 차 ~1e-3 → grad ~1e-7, 균일 동결).
4. `per_modality` 출력 = r·self → downstream LayerNorm이 양의 스칼라곱 제거 (no-op).
수정: 모달리티별 -KL(p_m‖p̄) / max-softmax c_mono / cross-modal 정규화+learnable temp /
mixed-mixture broadcast. 검증: 실 K-EmoCon에서 mono_mlp grad **0→81.9** (라우터 작동).

## 가설 (pre-registered)
- dar CCC − static CCC ≥ +0.02 (arousal / valence 각각). null: ≤0.
- 통계: 6-fold paired t-test + Wilcoxon + bootstrap 95% CI (analyze_htdar_kemocon.py).

## 판정 시나리오
- **dar > static 유의** → win. "학습된 앵커 > 고정 앵커 (지배 modality 없을 때)" = 컨퍼런스 코어.
- **tie** → fallback: 라우팅 시각화(r_m 샘플별 분포) interpretability angle. 여전히 publishable.

## 결과 (분석 스크립트 출력 붙일 자리)
[PENDING — 학습 완료 후 analyze_htdar_kemocon.py 결과 + leaderboard append]
