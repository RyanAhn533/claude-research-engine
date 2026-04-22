# Insights — 먹히는 패턴 / 안 먹히는 패턴

매 5 iteration마다 업데이트.

---

## Iteration 0 (초기 상태, 2026-04-21 기준)

### 먹히는 패턴 ✅

1. **Linear probe가 embedding 평가의 gold standard** (Alain & Bengio 2016)
   - Silhouette/cos sim 같은 간접 metric보다 직관적 accuracy가 납득력 높음
   - Random baseline 25% 대비 %p 차이로 보고하면 해석 쉬움
   - → 다음 iteration에도 primary metric으로 유지

2. **선행연구 → 가설 → 실험 → 선행연구와 대조** 구조
   - Jack 2012 PNAS를 thesis seed로 삼아 정량화
   - Ekman FACS를 자연스러운 검증 subset으로 활용
   - WebSearch 병렬로 5분 내 선행연구 확보

3. **Group-level feature subset** (Eye-AU group, Mouth-AU group)
   - 개별 AU보다 group이 noise 흡수해서 signal 명확

### 안 먹히는 패턴 ❌

1. **고차원 embedding에 silhouette score 직접 적용** (Phase 0.1a 실수)
   - L2 norm 없이 raw cosine은 scale distortion
   - 6144-8192d에서 silhouette은 편향되어 작아짐
   - → embedding 평가는 linear probe로 통일

2. **Path resolution 미검증 시 데이터 손실 방치** (Phase 0.2 슬픔 9.3% 매칭)
   - 기쁨/분노/중립은 99% 매칭, 슬픔만 9.3% — structural path issue
   - → Iteration 시작 전 class 균형 체크 필수

3. **Python 환경 dependency 순차 설치** (Py-Feat 실패, LC006)
   - user-level site-packages가 conda env 우회
   - → PYTHONNOUSERSITE=1 + `--force-reinstall --ignore-installed` 첫 시도부터 적용

4. **Region-level과 AU-level 혼용** (LC001 step 5, LC004)
   - "AU embedding"이 실제로는 region embedding이었음
   - → FACS AU(근육)와 face region(공간) 층위 명시적 구분

### 개방 질문

- AU-level (~60%) < Region-level (~80%) gap — 원인?
  - 가설 A: AU intensity는 scalar, Region embedding은 high-dim → capacity 차이
  - 가설 B: OpenGraphAU는 서양 얼굴 편향 → 한국인 AU intensity 부정확
  - 가설 C: AU + Region의 joint가 필요 (방법론 novelty 후보)

- Jack 2012 재검증의 level-dependent nuance
  - Region level mouth +14%p, AU level mouth +1.2%p
  - 이 괴리가 논문 §4.3의 core finding 구조

---

## Iteration 5 지점 update (2026-04-22)

### 이번 5 iterations에서 확정된 것

1. **연세대 298명 사회적 합의는 실제 label noise signal이다** (exp_002)
   - Clean(is_selected=0) filter로 Region +2.78%p, AU +2.14%p
   - Rejected subset 62.7% acc (랜덤 25% 초과) — 완전 노이즈 아닌 '애매한 경계'
   - §4.5 Social consensus 논문 direct contribution

2. **Jack 2012 reversal은 robust** (exp_003)
   - Raw: Mouth-Eye gap +13.60%p
   - Clean: +12.77%p (약간 축소, 여전히 대 gap)
   - Eyes가 clean에서 더 많이 improve (+2.87%p vs mouth +1.76%p)
   - → "noise가 eye region에 더 컸다" 재해석 가능

3. **Nose region이 예상외로 강력** (exp_003)
   - Clean 77.55% > Mouth clean 76.96% — new per-region top
   - Phase 0.1c raw에서도 75.01%로 mouth(75.20%) 거의 같음
   - § 4.3 추가 discussion 포인트

4. **AU intensity는 Region embedding과 redundant** (exp_004)
   - Joint - Region = +0.06%p (within noise)
   - Phase 3 graph 설계 pivot: AU-as-node 폐기, sample-as-node + consensus 중심으로

5. **Forehead는 class-invariant** (exp_003)
   - Raw→Clean delta +0.06%p (거의 무변화)
   - 다른 region은 +1.2 ~ +2.9 범위. Forehead만 예외
   - 한국인 감정 표현에서 forehead는 실제로 약한 신호 (noise 아님)

### Meta-pattern (이번 5 iter)

- **Positive finding (accuracy 상승)**: exp_002 (social consensus filter)
- **Robustness finding (gap 유지)**: exp_003 (Jack 12.77% clean)
- **Negative finding (method novelty 재설계)**: exp_004 (AU redundant)
- 세 종류 모두 **논문에 valuable** — negative도 Phase 3 방향 결정에 기여

### 먹히는 패턴 ✅ (새로 추가)
- **Subset filter 비교** (raw vs clean vs rejected): 한 실험에서 3 variant → ablation + interpretation 동시 확보
- **Per-region matrix** (8 region × 2 condition): paper table + figure 직접 생성

### 안 먹히는 패턴 ❌ (새로 추가)
- **단순 concat (AU + Region)**: redundant feature는 linear probe에서 marginal. Phase 3는 structural interaction (graph) 필요.

### 다음 5 iter 탐색 방향
- **Primary 3 (Jack reversal AU level)** 올리기: 현재 +1.0%p (AU), 목표 ≥+10%p
  - 아이디어: 개별 AU 단독이 아닌 "region-grouped AU" (예: upper-face AU group vs lower-face AU group)
  - 아이디어: AU intensity × annotator disagreement interaction
- **§4.5 Social consensus 강화**:
  - Demographic × consensus interaction (성별별 noise 비율)
  - 3-annotator majority vote effect
- **Per-class F1 floor** 올리기: 현재 슬픔/분노가 약함
  - Focal loss / re-weighting on linear probe
  - Class-conditional PCA

---

## Iteration 10 지점 update
(pending)

---

## Meta-pattern (iteration 전반)

- **수치는 랜덤 baseline 대비 %p로 보고** (0.815 → 81.5% vs random 25%, +56%p)
- **Paper section 매핑** 없는 실험은 취소 (Q1_WORKING §X.X)
- **Logic chain (LC###)** 기록 — RLHF 데이터 축적
