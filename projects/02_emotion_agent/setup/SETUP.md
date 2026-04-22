# SETUP — Emotion Agent Q1

모든 환경/경로/의존성 설정. Day 0에 한 번에 끝낸다.

---

## 0. 시스템 snapshot

| 항목 | 값 | 비고 |
|-----|-----|-----|
| GPU | NVIDIA RTX A6000 48GB | 타인 28.9GB 점유, **free 약 19.7GB** — JY 제약상 12GB는 남겨야 → **우리 사용 ≤ 7.7GB** |
| OS | Ubuntu | — |
| Conda | `/home/ajy/miniconda3/` | base에 torch 2.6 + cu124 |
| Disk work dir | `/data/heartlab/` or `/mnt/hdd/` | `/`는 87% 꽉참, 피함 |

---

## 1. Conda 환경

**전용 env 생성** (충돌 회피):

```bash
conda create -n cre_q1 python=3.10 -y
conda activate cre_q1

pip install --upgrade pip
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.48.0 accelerate bitsandbytes
pip install scikit-learn pandas numpy pyarrow tqdm einops
pip install librosa soundfile opencv-python pillow
pip install jupyter matplotlib seaborn
pip install wandb  # 선택
```

**Env 경로**: `/home/ajy/miniconda3/envs/cre_q1`

---

## 2. LLM — Qwen2.5 (local, 이미 다운로드됨)

**경로**:
- Qwen2.5-7B-Instruct: `/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct` (15GB)
- Qwen2.5-VL-7B-Instruct: `/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` (16GB)

**GPU 7.7GB 제약 → 4-bit quantization 필수**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir="/mnt/hdd/ajy/caches/huggingface",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir="/mnt/hdd/ajy/caches/huggingface")
```

**예상 VRAM**: ~4-5GB (4-bit). Batch 조정해서 7GB 내.

### Qwen2.5-VL (multimodal) 필요 시
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# 동일하게 4-bit 로드
```

---

## 3. 데이터 경로 매핑

| Dataset | 경로 | 상태 |
|---------|-----|-----|
| **IEMOCAP** | `/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_full_release/IEMOCAP_full_release/` | 18GB, Session1-5 전체 |
| (옵션) IEMOCAP preprocessed | `/mnt/hdd/huggingface_cache/datasets/AbstractTTS___iemocap` | HF 전처리 |
| **MELD** | `/data/heartlab/datasets/MELD/` 또는 `/mnt/hdd/BG/private_paper/datasets/MELD.Raw/` | |
| **CMU-MOSEI** | `/data/heartlab/datasets/CMU_MOSEI/` | |
| **CMU-MOSI** | `/data/heartlab/datasets/CMU_MOSI/` | |
| **DEAP** | `/mnt/hdd/BG/MER/DEAP_code/` | |
| **KEMDy20** | (SGMT pipeline 경로 — JY 확인) | S-PACE 재활용 |
| **K-EmoCon** | (SGMT pipeline 경로) | |
| **237K Korean FER** | `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/` | project 01 자산 |
| **Yonsei 298 consensus** | `/home/ajy/AU-RegionFormer/data/label_quality/all_photos.csv` | is_selected, n_evals, n_selected |

---

## 4. Preprocessing code 재활용

| 대상 | 위치 |
|------|-----|
| IEMOCAP preprocess | `/mnt/hdd/BG/MER/IEMOCAP_code/IEMOCAP_data_preprocessing.ipynb` (BG 랩원) |
| MELD preprocess | `declare-lab/MELD` repo clone 후 |
| DEAP preprocess | `/mnt/hdd/BG/MER/DEAP_code/` |
| S-PACE pipeline (bio+behavior) | JY 다른 서버의 코드 — 본 프로젝트로 import 필요 |

---

## 5. 작업 디렉토리

**Project root**: `/home/ajy/claude-research-engine/projects/02_emotion_agent/`

**실험별 디렉토리**: `experiments/exp_NNN_<desc>/`
**코드**: `src/` (agent, perception, reasoning, utils)
**결과**: `results/` (summary/figure — git에 커밋. 대용량은 gitignore)
**베이스라인 ckpt**: 외부 경로 (gitignore)

---

## 6. kill_switch

```bash
# 중단
touch /home/ajy/claude-research-engine/scripts/kill_switch
# 재개
rm /home/ajy/claude-research-engine/scripts/kill_switch
```

---

## 7. Sanity check (Day 0 완료 조건)

모두 통과해야 Phase 1 진입:

- [ ] `conda activate cre_q1` 성공
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` = True
- [ ] Qwen2.5-7B 4-bit 로드 + 한 prompt inference 성공 (VRAM < 7GB 확인)
- [ ] IEMOCAP Session1 WAV 하나 로드 성공
- [ ] MELD train.csv 로드 성공
- [ ] DEAP .mat 하나 로드 성공
- [ ] Yonsei consensus CSV 로드 성공 (`n_evals`, `n_selected` 컬럼 존재 확인)
