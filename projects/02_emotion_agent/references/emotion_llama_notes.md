# Emotion-LLaMA — Reproduction Notes

Source: https://github.com/ZebangCheng/Emotion-LLaMA
Local clone: `projects/02_emotion_agent/libs/Emotion-LLaMA/`
Paper: https://arxiv.org/pdf/2406.11161
Venue: **NeurIPS 2024** ✓ top-tier

## Overview
- Task: Multimodal emotion recognition + reasoning with instruction tuning
- Base: LLaMA-2 with **MiniGPT4**-style visual encoder + audio encoder
- Dataset: **MERR** (28,618 coarse + 4,487 fine)
- Benchmark results:
  - MER2023: **F1 = 0.9036** (winner)
  - DFEW zero-shot: UAR **45.59**, WAR **59.37**
  - EMER: Clue Overlap 7.83, Label Overlap 6.25

## Repo structure (essential)
```
Emotion-LLaMA/
├── train.py                      # main training entry
├── eval_emotion.py               # eval on benchmarks
├── eval_emotion_EMER.py          # EMER protocol
├── infer_api.py                  # inference server (API)
├── app.py / app_EmotionLlamaClient.py    # demo apps
├── environment.yaml              # conda env 'llama' 정의
├── requirements.txt
├── minigpt4/                     # MiniGPT4-based module
│   (core model code)
├── train_configs/
│   ├── Emotion-LLaMA_finetune.yaml       # 주 finetune config
│   └── minigptv2_tuning_stage_2.yaml
├── eval_configs/
├── checkpoints/                  # pretrained .pth 대기
├── examples/                     # demo inputs
└── docs/
```

## Env dependencies (주요)
- torch (LLaMA-2 호환)
- `decord` (video frames)
- bitsandbytes==0.41.2.post2 (4/8-bit quant)
- accelerate==0.24.1
- transformers (특정 버전 호환 필요)
- MiniGPT4 derived code

## GPU 요구
- **Fine-tune**: LoRA 기준 24-30GB VRAM (A6000 48GB 온전 필요)
- **Inference only**: 14-16GB (4-bit quant 10GB 추정)
- **BrandSpace 상주 (7.7GB budget) 현 상태**: 학습 불가, inference 가능성 타이트

## Reproduction 전략

### Phase A (Week 2 초반) — Inference-only
- Pretrained checkpoint 다운로드 → `checkpoints/`
- `infer_api.py` 또는 `eval_emotion.py` 로 MER2023/DFEW inference
- 결과 수치 재현 → baseline table
- BrandSpace 끄고 작업

### Phase B (Week 2 후반, 선택) — LoRA finetune 1 epoch
- 우리 데이터 (IEMOCAP/MELD Korean subset이 있으면) 으로 LoRA
- 최소 효과 검증
- Skip 가능 (시간 부족 시)

## 우리 Q1 논문에서 위치
- **§2 Related Work**: Emotion-LLaMA = closest SOTA. Cite.
- **§4 Comparison**: 같은 benchmark 위에서 Ours vs Emotion-LLaMA 수치
- **§3 Method 차별점**:
  - Emotion-LLaMA: MERR 데이터 instruction tune, Western emotions 중심
  - Ours: **Korean cultural prior + bio-grounded + agent tool-use** 차별화

## Known caveats
- `environment.yaml`의 `llama` env 이름 우리 env (`cre_q1`)와 충돌 피해 별도 만들기
- `minigpt4` 모듈은 외부 코드 포함. 라이선스 체크
- Checkpoint 다운로드 경로: Google Drive (folder: `1LSYMq2G-TaLof5xppyXcIuWiSN0ODwqG`)

## Next Week 2 tasks
1. Emotion-LLaMA용 별도 conda env 생성 (`conda env create -f environment.yaml -n emoLLaMA`)
2. Checkpoint 다운로드 (Google Drive)
3. MER2023 또는 DFEW 1 sample inference 시도
4. 통과 시 전체 eval 수치 재현 → leaderboard

## Ongoing
Local clone at `libs/Emotion-LLaMA/` (2026-04-23)
