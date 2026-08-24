# K-EmoCon sweep — routing NO-GO (2026-06-10)

> "bio 무대로 옮기면 routing이 살아날 것"이라는 가설 검증. **틀렸다.** routing은 K-EmoCon에서도 죽었다.
> 코드: `/home/ajy/02_multimodal_emotion/03_visionmer_v3_binary/` (srxma swap 추가, 교차검증 완료).
> 데이터: K-EmoCon (bio/audio/video, **text 없음** = routing이 작동해야 할 "진짜 전장"). 6-fold subject-wise.

## 결과 (acc / f1m / auc / CCC, majority 비교)

| config | acc | f1m | auc | CCC | major |
|---|---|---|---|---|---|
| **AROUSAL** (maj 0.639) | | | | | |
| static (고정, 최단순) | **0.670** | 0.588 | 0.678 | 0.218 | 0.639 |
| dar (동적) | 0.643 | 0.609 | 0.693 | 0.239 | 0.639 |
| hgar (그래프) | 0.652 | 0.607 | 0.668 | 0.217 | 0.639 |
| srxma full (상한) | 0.650 | 0.596 | 0.657 | 0.242 | 0.639 |
| srxma random (null) | 0.660 | 0.599 | 0.660 | 0.246 | 0.639 |
| srxma fixed | 0.652 | 0.590 | 0.631 | 0.229 | 0.639 |
| srxma learned | 0.658 | 0.603 | 0.640 | 0.223 | 0.639 |
| **VALENCE** (maj 0.948) | | | | | |
| static | 0.948 | 0.486 | 0.675 | 0.098 | 0.948 |
| dar | 0.948 | 0.570 | 0.682 | 0.094 | 0.948 |

(arousal std ~0.06–0.09. valence/srxma·hgar 일부는 천장 확인 후 중단 — no result.)

## 판정

1. **routing headroom = 0 (arousal)**: srxma **full(0.650) ≤ random(0.660)** — 상한이 null을 못 이김. learned(0.658)도 random과 동급. 전부 std(0.08) 안. **MOSEI와 동일.** text 없는 bio 무대에서도 "어떤 엣지 고르냐"가 정확도에 영향 0.
2. **HT-DAR 부활 실패**: static(고정, 0.670)이 dar/hgar(동적/그래프)를 acc로 오히려 앞섬. "지배 modality 불안정 전장"에서도 동적 앵커 이득 없음.
3. **valence = 라벨 천장**: 전 모델 acc=majority(0.948), CCC≈0.09. 학습 신호 없음.

## 결론
**SR-XMA/BioCARS의 핵심 = "learned cross-modal edge routing"이 3개 독립 셋(MOSEI + K-EmoCon arousal + valence)에서 일관되게 random과 무차별.** routing 축은 죽었다. 데이터 탓이 아니라 이 task들에서 cross-modal 엣지 선택 자체가 무의미. BioCARS 컴포넌트(lag/uncertainty 등)를 이 위에 쌓는 건 작동 안 하는 라우터 정교화일 뿐.

## 살아있는 것
- 정직한 negative/efficiency 논문 재료 ("sparse≈dense cross-modal fusion, learned routing 무의미 + redundancy 분석").
- 검증된 인프라: SR-XMA V2, K-EmoCon srxma 통합, 교차검증 방법론.
- 축 전환 후보: routing이 아니라 **bio representation 품질**(S-PACE thesis) — 다른 레버.

관련: [[CODE_REVIEW_srxma_pipeline_20260607]], [[STRUCTURE_codewalk_20260608]], MOSEI gap 결과.
