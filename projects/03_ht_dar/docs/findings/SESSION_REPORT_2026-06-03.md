# HT-DAR 재현·평가 세션 종합 기록 (2026-06-02 ~ 06-03)

> 프로젝트: `projects/03_ht_dar` (research-engine v2 편입)
> 목적: HT-DAR (JY가 설계한 DLF fusion 교체 모듈)를 표준 MSA 벤치마크에 적용, 성능을 통계적으로 평가하고 투고처 판단.

---

## 0. 한 줄 결론
**HT-DAR는 MOSEI·IEMOCAP 두 표준 벤치마크에서 baseline(DLF-stripped)을 통계적으로 못 이긴다 (둘 다 within-noise, p>0.5).** 절대 성능은 AAAI 2025 SOTA band(0.85~0.86 acc_2) 안에 있으나, ablation delta는 noise. 원 논문 저자의 honest disclosure와 일치(오히려 더 강하게 확인).

---

## 1. 환경·셋업 (재현용)

| 항목 | 값 |
|---|---|
| 머신 | /home/ajy, RTX A6000 48GB (타 유저 sc와 공유) |
| conda | base (torch 2.6.0+cu124, transformers 4.50, scipy, sklearn) |
| 코드 | `/home/ajy/HT-DAR/` (HT-DAR-main.zip 해제) |
| BERT | `/mnt/hdd/huggingface_cache` 로컬 오프라인 |
| 데이터 루트 | `/mnt/hdd/ajy/HT-DAR-data/` → `HT-DAR/dataset` 심볼릭 링크 |

### 데이터 출처·검증
| 데이터셋 | 파일 | 출처 | 검증 |
|---|---|---|---|
| CMU-MOSEI | `MOSEI/Processed/aligned_50.pkl` (4.4G) | Self-MM gdrive 미러 (id 1Z8XJBUz…) | SHA-256 `45eccfb7…` 일치 ✓ |
| CMU-MOSI | `MOSI/Processed/aligned_50.pkl` (351M) | Self-MM gdrive (id 1CKYtrKh…) | 로드 OK |
| IEMOCAP | `mult_archive/iemocap_data.pkl` (291M) | MulT Dropbox Archive.zip | train/valid/test, labels(N,4,2), [300,74,35] seq20 ✓ |
| (보너스) | mosi/mosei MulT pkl도 같은 archive에 있음 | | |
| K-EmoCon | `/mnt/hdd/ajy/datasets/kemocon_precessed_data` (S-PACE 포맷) | 기존 보유 | HT-DAR 미통합 |

### 코드 패치 (zip 누락분)
1. `pip install pynvml` (GPU 할당 유틸 의존성 누락)
2. `code/utils.py` shim 생성 — `from trains.utils.functions import assign_gpu, setup_seed, …` (run.py가 최상위 utils import하나 zip에 없었음)
3. `train_stripped.py` — 4번째 인자 `DATASET` 추가 (mosei/mosi 전환, 기존 호출 호환)
4. `train_iemocap.py` — `DATA_PATH` 우리 경로로, split 키 `dev`→`valid` (MulT pkl 구조 맞춤)

### 보조 스크립트 (projects/03_ht_dar/)
- `log_results.py` — MOSEI 로그 → best-val test acc_2 파싱 → leaderboard 집계 append (≥3seed만)
- `gate_c_runner.py` — M1 vs M0 5-seed Gate C (bootstrap CI/Cohen's d/Wilcoxon/TOST)
- `CONFIG_FINGERPRINTS.json` — M0~M4 config 지문

### 실행 스크립트 (HT-DAR/code/)
`run_ladder_mosei.sh`, `run_multiseed_ext_v2.sh`, `run_m234_multiseed.sh`(MOSEI), `run_ladder_iemocap.sh`, `run_iemocap_ms5.sh`(IEMOCAP)

---

## 2. 결과

### 2.1 MOSEI 단일시드(1111) 사다리 — reference 정확 재현
| 변형 | best-val test acc_2 | reference |
|---|---|---|
| M0 baseline | 0.8525 | 0.8504 |
| **M1 Hier** | **0.8608** | 0.8559~0.8609 |
| M2 DAR only | 0.8539 | 0.8539 ✓ |
| M3 Hier+DAR | 0.8550 | 0.8550 ✓ |
| M4 full | 0.8528 | 0.8528 ✓ |
M2/M3/M4 소수점 4자리 일치 → 셋업/데이터 정상 입증.

### 2.2 MOSEI 5-seed (M0/M1) + Gate C ★
| 변형 | 5-seed mean | std | per-seed |
|---|---|---|---|
| M0 | 0.8528 | 0.0036 | 0.8525/0.8487/0.8500/0.8591/0.8539 |
| M1 | 0.8524 | 0.0066 | 0.8608/0.8536/0.8533/**0.8404**/0.8539 |

**Gate C (M1 vs M0, paired n=5)**: Δ=-0.0004, bootstrap CI95=[-0.0099,+0.0060](0 포함), Cohen d=-0.04, Wilcoxon p=0.875, paired-t p=0.931, TOST±0.5pp p=0.196(동등성도 미입증) → **neutral**.
→ 단일시드 M1=0.8608은 lucky seed. 5-seed에선 M0와 동급(약간 낮음). 가설 `H_htdar_m1` outcome=**neutral**.
→ MOSEI M2/M3/M4 × seed1112-1115 멀티시드는 본 세션 종료 시점 백그라운드 진행 중(within-noise 확정 보강용).

### 2.3 IEMOCAP 5-seed 사다리 (test wF1) ★
| 변형 | mean | std | vs M0 | Gate C |
|---|---|---|---|---|
| M0 | 0.5820 | 0.0106 | — | — |
| M1 Hier | 0.5827 | 0.0072 | +0.08pp | neutral |
| M2 DAR | 0.5839 | 0.0087 | +0.20pp | neutral |
| M3 Hier+DAR | 0.5799 | 0.0052 | -0.21pp | neutral |
| **M4 full** | 0.5898 | 0.0184 | +0.78pp | neutral (t_p=0.50) |
| **A1 parallel** | 0.5904 | 0.0185 | +0.85pp | neutral (t_p=0.57) |

→ 단일시드에선 M4=+1.7pp로 좋아 보였으나 5-seed에서 +0.78pp로 축소, **유의성 없음(p=0.50)**. reference IEMOCAP(A1 +1.13pp p=0.32 INCONCLUSIVE)과 동일 패턴.

### 2.4 MOSI — 보류(misconfigured)
stripped 셋업에서 test acc_2 0.57 고정(30ep, 표준 0.84). reference도 MOSI stripped 미사용. config 미튜닝. 죽임. (MulT mosi_data.pkl로 재시도 여지 있음)

---

## 3. 종합 평가 (구조 vs 성능)
- **구조**: HT-DAR(동적앵커라우팅+계층피라미드+Beta게이트)는 DLF의 고정 language-anchor(LFA)를 일반화한 새 메커니즘. 차별성 有.
- **성능(절대)**: MOSEI ~0.853, IEMOCAP ~0.582 — 둘 다 baseline/SOTA band. 망한 거 아님.
- **성능(ablation delta)**: **두 벤치 모두 within-noise (p>0.5).** "정확도로 SOTA 갱신" 서사 불가.
- **데이터셋별 패턴**: 단일시드에선 MOSEI=M1우세 / IEMOCAP=M4우세로 달랐으나, 멀티시드에선 둘 다 noise로 수렴 → "데이터셋 따라 다른 단이 산다"는 단일시드 관찰도 통계적으론 약함.

---

## 4. 경쟁·투고 지형 (조사 결과)
- **Baseline = DLF (AAAI 2025**, Wang et al., arXiv 2412.12225). top-tier.
- **MOSEI Acc-2 SOTA ~86.4~86.7 포화** (HTRN 86.7, DLF 86.4, MSAmba). 6년간 +4pp뿐. sub-1pp가 정상.
- **남은 2026 투고처** (6/3 기준, 4월 마감 ICMI/ACII/ACMMM 지남):
  - AAAI 2027: full 7/28 (top-tier, baseline 출신, 정확도 SOTA 필요 → 현 상태론 무리)
  - ICASSP 2027: 9/16 (~15주, 현실적, 4p)
  - ICLR 2027 ~9-10월, WACV/CVPR(주제 안 맞음)
  - Neurocomputing(저널, 상시)

---

## 5. 결론 & 남은 길
표준 벤치마크 정확도 논문은 불가(둘 다 noise). 의미 있는 길:
1. **K-EmoCon** — 유일하게 큰 게인(옛 HT-CMA-BLF 10× cccA) 보고된 곳. HT-DAR 미구현(Phase-D). 구현해서 진짜 이기면 메인 펀치 → ICASSP/ACII급. **유일한 "이긴다" 카드.** (데이터 보유, 신규 구현 필요)
2. **Reframe** — robustness(모달리티 결손)/해석(라우팅)/효율. 정확도 SOTA 없이 mid-tier 가능.
3. **정직한 분석 논문** — "고정 앵커가 언제 충분한가" negative result. 워크샵/숏페이퍼.

→ 미결정: K-EmoCon 승부 vs reframe. (JY 결정 대기)

---

## 6. 엔진 상태 (projects/03_ht_dar)
- hypothesis_registry: H_htdar_m1/m2/m4 + m1 outcome(neutral) — 해시체인 검증
- leaderboard: MOSEI M0/M1 5-seed + M1 Gate C supersede + IEMOCAP 6변형 5-seed (모두 neutral)
- reproducibility_manifest: MOSEI 1건 (data_hash/env_lock/git_sha)
- exploratory_single_seed.json: MOSEI M0~M4 단일시드
- insights.md, 본 문서
- `python -m engine.cli.jy validate --project 03_ht_dar` 통과
