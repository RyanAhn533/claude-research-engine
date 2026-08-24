# exp_057 — AU 사전 조건화 (AU Prior Conditioning)

**상태**: 2026-08-05 신설, 스모크 진행 중
**설계 근거**: `/home/ajy/AU사전_VLM실험_설계메모_2026-08-05.md`

---

## 무엇을 하는 실험인가

VLM/LLM에게 AU 값을 넘길 때 **어떤 AU→감정 사전(prior)을 함께 주느냐**가
판정 성능을 바꾸는지, 그리고 **감정별로 최적 사전이 갈리는지**를 본다.

동시에 exp_038 이래 프롬프트를 오염시켜온 **AU 스케일 버그를 고친다.**

---

## 1. ★버그 수정 (선행 조건)

### 문제
`exp_038/intervene.py:34-41`의 `numeric_fmt(row, thr=50)`은 AU가 **0~100**이라 가정하고
`row[au] >= 50`으로 필터한다. 그런데 parquet 실제 값은 **0~1**이다.

```
opengraphau_41au_237k_v2.parquet  (229,003 × 46)
AU min/max/mean = 0.0 / 0.9956 / 0.1337
is_selected==0 205,336행 중 어느 key-AU라도 50 이상인 행 = 0개
```

결과 연쇄:
1. 임계 분기가 **한 번도 발동하지 않음** → 항상 top-3 fallback
2. `f"{v:.0f}"`가 0~1 실수를 반올림 → 값이 **0 아니면 1**
3. 실측 렌더 분포(2000행 × 3토큰 = 6,000개): **1이 5,525개(92%) / 0이 475개(8%)**

즉 모델은 "상위 3개 AU가 뭔지"만 받고 **강도는 못 받았다.**
그런데 시스템 프롬프트는 `"AU intensities (0-100)"`라고 말하고 있었다.

`prose_fmt`의 `intensity_word`(>=70 strong / >=50 moderate / else slight)도
같은 이유로 **항상 "slight"** 를 반환한다.

### 수정 (`au_prior.py:numeric_fmt_fixed`)
```python
AU_SCALE = 100.0      # 0-1 → 0-100
ACTIVE_THR = 20.0     # 0-100 단위 (mean*100 = 13.4이므로 20이면 salient만)
MAX_AUS = 8           # 프롬프트 가독성 상한
MIN_AUS = 3           # 아무것도 안 넘으면 top-3 (빈 문자열 방지)
```

### 수정 전후 (2026-08-05 실측)

| | 수정 전 | 수정 후 |
|---|---|---|
| 값 범위 | 0 또는 1 (고유값 2개) | **19~99 (고유값 81개)** |
| 값 평균 | — | 61.6 (중앙값 59) |
| AU 개수 | 항상 3개 고정 | **3~8개 가변** (8개 1597 / 7개 243 / 6개 117 / 5개 40 / 4개 1 / 3개 2) |

수정 후 렌더 예시:
```
[기쁨] AU25 (lips part)=88, AU5 (upper lid raiser)=78, AU12 (lip corner puller)=63,
       AU10 (upper lip raiser)=60, AU7 (lid tightener)=47, AU2 (outer brow raiser)=47,
       AU26 (jaw drop)=44, AU1 (inner brow raiser)=38
```
(수정 전에는 `AU25 (lips part)=1, AU5 (upper lid raiser)=1, AU12 (lip corner puller)=1`)

### ⚠️ 기존 실험은 건드리지 않았다
exp_035/038/039/040/056의 렌더러는 **그대로 둔다**. 논문에 실린 숫자의 재현성을 지키기 위해서다.
따라서 exp_057 결과는 **기존 실험과 절대 수치를 직접 비교하면 안 된다.**
비교는 exp_057 내부(조건 간)에서만 유효하다.

---

## 2. 5개 사전 조건

전부 **시스템 프롬프트에 한 줄 삽입**만 다르고 나머지는 exp_038 `prompt_numeric`과 동일.

| 조건 | 삽입 내용 | 출처 |
|---|---|---|
| `A_none` | (없음) | baseline = 현행 no-FACS |
| `B_textbook` | `FACS prototypes (Ekman): Happy=AU6+12; Sad=AU1+4+15; Angry=AU4+5+7+23; Neutral=no strong AU activation.` | exp_035 문자열 + neutral 명시 |
| `C_derived` | `AU6 43%, AU12 43%, AU10 42%, AU7 41%, AU25 40%, AU15 37%, AU9 36%, AU4 34% … Weight them accordingly.` | **우리 데이터** `outputs/phase0/03_per_au_linear_probe/per_au_ranking.csv` |
| `D_mouth` | `Focus on the mouth and cheek region (AU6,10,12,14,15,20,23,25,26,27)` | 우리 linear probe (mouth 75.2) |
| `D_eyes` | `Focus on the eye and brow region (AU1,2,4,5,7)` | Jack 2012 PNAS (동아시아=눈) |

**C가 핵심.** 교과서 사전은 이미 졌지만(아래 §4) **데이터에서 유도한 사전은 안 넣어봤다.**
"심리학이 말하는 AU 조합" vs "이 데이터가 실제로 쓰는 AU 조합" 대결.

**D는 Q2 thesis 직결.** Phase 0.1 linear probe는 mouth 75.2 / eyes 61.5 (13pp 괴리)로
Jack 2012의 동아시아=눈 예측과 어긋난다. VLM에게 영역을 지시했을 때 어느 쪽이 맞는지가
그 괴리에 대한 독립 증거가 된다.

---

## 3. 설정

- 모델 6종: qwen7b, qwen14b, qwen3b, yi6b, falcon7b, mistral (4bit nf4, greedy, bs=32)
- seed 7개: 42, 123, 777, 2024, 31337, 1234, 9001
- N=400 (클래스당 100), k=4 (클래스당 exemplar 1)
- 데이터: AU parquet `is_selected==0` 205,336행 풀
- 지표: acc, macro-F1, **감정별 recall**, **parse_fail 카운트**

### exp_035 대비 추가한 것
- **parse_fail 명시 집계** — exp_035의 `parse_output`은 실패를 조용히 `"neutral"`로
  대체하고 실패율을 기록하지 않았다. 여기서는 라벨 문자열이 아예 없는 응답을 따로 센다.
- **감정별 recall** — Step 1의 핵심 질문(감정별로 최적 사전이 갈리는가)에 필요.

---

## 4. 선행 결과 (이 실험이 서 있는 위치)

**"교과서 사전을 넣는다 vs 안 넣는다"는 이미 돌렸고, 졌다.**

| 실험 | 결과 |
|---|---|
| exp_035 with-FACS vs exp_039 no-FACS | mistral **+15.17 → +1.15** (FACS 한 문장이 결과를 뒤집음) |
| `state/insights.md:21` 문화 사전 주입 | 34.25% → 27.25% = **−7.00pp** (7변형 중 2개만 양수) |
| exp_014 | "Ekman FACS textbook = **random on Korean (25.25%)**" (chance 25%) |
| exp_012 | 추상 사전 대신 **구체 예시 → +12pp** (ICL 논문의 출발점) |

→ 그래서 exp_057은 "사전을 넣나 마나"가 아니라 **"어떤 사전이냐"** 를 묻는다.

---

## 5. 실행

```bash
# 스모크 (1모델 1seed 2조건)
export USE_TF=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_CPP_MIN_LOG_LEVEL=3 \
       HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
python -u au_prior.py --model qwen7b --seeds 42 --priors A_none,B_textbook --shots 4

# 전체 패널 (6모델 × 5조건 × 7seed)
./run_panel.sh

# 부분 실행
PRIORS=A_none,C_derived SEEDS=42,123 ./run_panel.sh

# 분석
python analyze.py 4        # k=4
```

⚠️ 환경변수 필수 — 안 붙이면 `protobuf Descriptors cannot be created directly`로 죽는다
(exp_056 `run_panel.sh`와 동일).

---

## 6. 분석이 답하는 질문 (`analyze.py`)

| # | 질문 | 출력 |
|---|---|---|
| Q1 | 사전이 전체 정확도를 바꾸는가 | `delta vs A_none (pp)` 표 |
| **Q2** | **★감정별로 최적 사전이 갈리는가** | per-emotion recall 표 + 감정별 argmax |
| Q3 | 교과서(B) vs 데이터유래(C) | 모델별 승패 |
| Q4 | 입 vs 눈 (Jack 2012 검증) | mouth/eyes 승패 |

**Q2가 이 실험의 심장이다.** 감정별로 갈리면 → "나라 차이"보다 **"감정 차이"가 먼저**이고,
그건 데이터셋 하나 안에서 교락 없이 측정된다.

---

## 7. 왜 나라별(cross-dataset)을 지금 안 하는가

두 가지 지뢰가 이미 확인돼 있다.

**지뢰 1 — 한국인 특이성은 이미 기각됨 (EA-H007)**
서양 FER 부정감정 gap **0.49 ≥ 한국 0.446** → thesis에서 "문화" 제거,
format-vs-ambiguity(문화 무관)로 reframe.

**지뢰 2 — 2026-08-04 확인된 분노 라벨 오염**
- EmotiEffLib **통과군조차** Anger 0.06
- POSTER++ 통과군 Anger 0.20 < Fear 0.25
- 블라인드 판정단 합의제외율(원거부군): **angry 0.86** / sad 0.50 / happy 0.29 / neutral 0.04
- 결론: **"연기 기반 부정감정 수집의 구조적 실패"**

→ 한국 vs 서양 분노 AU를 비교하면 **반드시** 차이가 나오지만, 그건 문화가 아니라
**수집 방식(연기 vs in-the-wild)** 이다. `dataset ≡ country` 교락이고,
구조가 KMER의 `road_type ≡ 감정`과 동일하다.

**확장 조건**: Step 1에서 감정별 신호가 확인된 뒤, 서양 4개 데이터셋에
41-AU를 **v3 전처리(YOLO face crop) 거쳐** 추출하고 나서.
(현재 41-AU parquet은 한국 237K 한 벌뿐. crop 없이 추출하면 SFEW/AFEW는
얼굴이 프레임 5%뿐이라 v2 배경학습 버그를 재현한다.)

---

## 8. 파일

```
au_prior.py     렌더러 수정 + 5조건 프롬프트 + 러너
run_panel.sh    6모델 순차 실행 (cache 있으면 skip)
analyze.py      Q1~Q4 분석
cache/          au_prior_{model}.json
logs/           모델별 실행 로그
```

## 9. 주의

- 기존 실험 결과와 **절대 수치 직접 비교 금지** (렌더러가 다름)
- `ACTIVE_THR=20.0`은 임의 선택 — 필요하면 민감도 확인할 것
- 사전 문구 자체가 프롬프트 길이를 바꾸므로, 길이 효과가 걱정되면
  A_none에 동일 길이의 무의미 텍스트를 넣은 placebo 조건 추가 검토
