# GPU Sharing — A6000 점유 현황

## Lab 환경
- **A6000 48GB × 1대** (Dell Precision 7960, JY 개인 머신)
- 연구실 전체 A6000만 있음
- Multi-GPU 불가. 단일 GPU 점유 관리 필요.

## 상시 점유 프로세스

### BrandSpace Engine FastAPI (약 28-29GB)
- Path: `/home/ajy/brandspace/`
- 용도: 웹사이트 데모 (live)
- Runtime: Qwen2.5-VL + Qwen2.5-7B + Flux.1 Dev
- Ops manual: `/home/ajy/brandspace/OPERATIONS.md`
- **종료**: `pkill -f "serve.py --port 8090"` → **GPU 48GB 전체 사용 가능**
- **재시작**: `cd /home/ajy/brandspace && nohup python serve.py --port 8090 > /tmp/brandspace_serve.log 2>&1 &`

## 실험 시 GPU 사용 레벨

| 상태 | Available VRAM | 가능 작업 |
|-----|-------------|---------|
| BrandSpace 상주 | ≤ 7.7GB | Qwen 4-bit inference, preprocessing, small head 학습 |
| BrandSpace 종료 | ~48GB | 7B LoRA fine-tune, Emotion-LLaMA 재현, 큰 multi-GPU 시뮬 |

## 실험 전 체크리스트

```bash
# GPU 상태
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 내 process인지 확인
ps -p <PID> -o user,cmd --no-headers

# 필요하면 종료 후 실험, 끝나면 재시작
```

## 프로젝트별 전략

| 프로젝트 | GPU 수요 | 권장 |
|---------|--------|-----|
| Project 01 (Q2, AU-RegionFormer) | 낮음 (linear probe) | 현 상황 (7.7GB)로 OK |
| Project 02 (Q1, Emotion Agent) — inference | 낮음 | 7.7GB OK |
| Project 02 — LoRA fine-tune | 30GB+ | BrandSpace 종료 후 실행 |
| Project 02 — Emotion-LLaMA 재현 | 30GB+ | BrandSpace 종료 후 실행 |

## 주의

- BrandSpace는 **웹사이트 live**이므로 종료 전에 웹사이트 트래픽 확인
- LoRA fine-tune은 **새벽/주말 등 저트래픽 시간에** 실행 권장
- 학습 끝나면 BrandSpace 재시작
