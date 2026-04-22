---
exp_id: exp_000_phase0_setup
phase: 0
status: in_progress
date_planned: 2026-04-22
category: infra
---

# Phase 0 Setup — cre_q1 env + Qwen 4-bit sanity

## 목적
ROADMAP Phase 0의 4개 gate 통과:
1. `cre_q1` conda env (Python 3.10) 생성 성공
2. `torch.cuda.is_available() == True`
3. Qwen2.5-7B 4-bit 로드 + 1 prompt inference 성공 (VRAM < 7GB)
4. 주요 데이터 (IEMOCAP, MELD, DEAP, Yonsei consensus) 로드 sanity

## 제약
- GPU free 19.7GB (타인 28.9GB 점유) → **우리 사용 ≤ 7.7GB**
- PYTHONNOUSERSITE=1 사용 (user-level site-packages 충돌 회피 — pyfeat 실패 lesson)
- `/home/ajy/.local` path 간섭 주의

## 단계
1. `conda create -n cre_q1 python=3.10 -y`
2. `pip install` (torch, transformers, bnb, sklearn 등)
3. Qwen 4-bit 로드 스크립트: `scripts/qwen_sanity.py`
4. 데이터 경로 검증: `scripts/data_sanity.py`
5. 전체 결과 `results.json` + `summary.md` 저장

## Fallback
- Qwen 4-bit 로드 실패 → 8-bit 시도 (VRAM 타이트)
- 그래도 실패 → Qwen2.5-3B로 downscale
- 데이터 path 일부 누락 → 해당 dataset 만 skip, 나머지 진행
