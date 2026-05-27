# exp_001_congruence_head — KILLED at 2026-05-23 ~30min in

## Why killed

3 페르소나 SELF_ATTACK 결과 + 1 critical data bug:

### Bug (확정)
- 새 master_v_yonsei_pathfix.csv 에 옛 컬럼명 `mean_is_selected` 없음
- dataset_v2.py 가 default 0.0 fallback → 전체 batch reject_rate=0
- → congruence head 는 "모두 no reject" 만 학습. **학습 garbage**

### 3 페르소나 결과
| 페르소나 | severity | 핵심 |
|---|---|---|
| method_skeptic | high | Stage 6 baseline = 옛 csv → fair comparison 불가; n_seeds<7; fillna leakage |
| novelty_critic | high | Peterson 2019 / DMUE CVPR21 / MTAC TMM23 이미 함; Stage 11 humankl 동일 채널 |
| reviewer_simulator | 3.5/10 Reject | 8-region zero-mask=0.0001 → confession, not contribution |

### 가장 아픈 한 줄 (reviewer_simulator)
> "Adding a single MLP head on the global feature does not transform a refuted interpretability claim into a publishable one; it merely obscures it."

## Lessons (다음 exp 적용)

1. **Dataset 컬럼 매핑 검증 필수** — 학습 전에 batch sample 1개 print 로 mean_is_selected 값 확인
2. **Baseline 도 같은 csv 로 재학습 필수** — 옛 baseline F1=0.9256 / C2=0.114 무효
3. **Crossed ablation 필요**: `{regions on/off} × {head on/off} × {Stage 11 humankl on/off}` 8 cells
4. **n_seeds ≥ 7** (v2 paper_ready Gate C)
5. **Thesis operationalize 필수** — architecture ablation ≠ "self-report ≠ self-validation" thesis
6. **Region-conditioned congruence** 가 진짜 contribution 가능성 (g_feat 만 X, patch token 도 입력)

## Files preserved (참고용)
- design.md, attack_method_skeptic.md, attack_novelty_critic.md, attack_reviewer_simulator.md
- configs/, results/train_seed999.log (30분 진행)
- pcfer_wrapper.py + trainer.py patch 는 src/ 에 남아있음 (다음 exp 재사용)
