# Phase 0 Setup — PASS

## 결과 요약

| Gate | 결과 |
|------|-----|
| `cre_q1` conda env (Python 3.10, torch 2.6.0+cu124) | ✅ |
| CUDA available | ✅ RTX A6000 |
| Qwen2.5-7B 4-bit 로드 (130s) | ✅ VRAM 5.4GB (budget 7.7GB 내) |
| Inference sanity ("crushed" → "sad") | ✅ 0.32s |
| Data path 12/12 (IEMOCAP/MELD/MOSEI/MOSI/DEAP/Yonsei/FER/Qwen caches) | ✅ |

## Install log

```
torch==2.6.0+cu124
transformers==4.48.0
accelerate==1.13.0
bitsandbytes==0.49.2
scikit-learn==1.7.2
pandas==2.3.3
librosa==0.11.0
soundfile==0.13.1
```

## Qwen snapshot path

```
/mnt/hdd/ajy/caches/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/
```

4 safetensors shards + config. `local_files_only=True` 없이 snapshot 경로 직접 지정 방식 확정.

## Lessons

- `cache_dir` + `local_files_only=True` 조합은 HF hub cache 구조 탐색 실패 → snapshot 경로 직접 지정이 안전
- `PYTHONNOUSERSITE=1` 강제 필수 (user-level pkg 간섭 회피)
- Qwen2.5-7B 4-bit NF4 = VRAM 5.4GB (실측). 7B로도 agent inference 여유 있음

## Next: Phase 1 Week 1

- Day 1: IEMOCAP preprocessing ipynb 재활용 확인
- Day 2-3: MELD + DEAP 로드
- Day 4-5: SGMT-style baseline 4 dataset 수치 확보
