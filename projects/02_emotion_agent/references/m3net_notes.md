# M3NET — Reproduction Notes

Source: https://github.com/feiyuchen7/M3NET
Local clone: `projects/02_emotion_agent/libs/M3NET/`
Paper: Chen et al., CVPR 2023 — "Multivariate, Multi-frequency and Multimodal: Rethinking GNN for ERC"

## Overview
- Task: MERC (Multimodal Emotion Recognition in Conversation)
- Model: **Hypergraph GNN** + multi-freq + multi-modal
- Input: pre-extracted RoBERTa/GloVe text + audio + visual features
- Benchmarks: IEMOCAP, MELD

## Repo structure
```
M3NET/
├── train.py                 # main entry
├── model.py / model_GCN.py / model_hyper.py
├── dataloader.py
├── HypergraphConv.py        # core graph conv
└── high_fre_conv.py         # multi-freq filter
```

## Dependencies (constraints)
- **Python 3.8.5**
- **torch 1.7.1**  ← 우리 torch 2.6과 **호환 안 됨**
- CUDA 11.3
- torch-geometric 1.7.2

## 실행 옵션
### Option A (권장): 별도 env
```bash
conda create -n m3net python=3.8 -y
conda activate m3net
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==1.7.2
```
CUDA 11.0 빌드. 현재 GPU cu12.4 driver에서 backward compatible 가능성 있음 (cuFFT/cuDNN legacy).

### Option B: torch 2.x 포팅
- HypergraphConv 재구현 (torch-geometric 2.x)
- 시간 오버헤드 큼 → 비추천

## Dataset
- Pre-extracted features: [Dropbox](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0)
  - IEMOCAP / MELD RoBERTa + GloVe + audio + visual features (pkl)
  - 다운로드 수동 필요

## Pretrained checkpoints
- IEMOCAP (RoBERTa-based): [Dropbox](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0)
- Inference-only로 수치 재현 가능:
  ```
  python train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 \
    --batch-size 16 --graph_type='hyper' --epochs=0 \
    --graph_construct='direct' --multi_modal \
    --mm_fusion_mthd='concat_DHT' --modals='avl' \
    --Dataset='IEMOCAP' --norm BN --testing
  ```

## GPU 요구
- **적음** (graph NN, feature 이미 pre-extracted). 4-8GB VRAM 가능
- 현재 budget 7.7GB 내 실행 가능 (Emotion-LLaMA보다 가벼움)

## Training commands (paper 재현 시)
IEMOCAP:
```
--base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 \
--graph_type='hyper' --epochs=80 --graph_construct='direct' --multi_modal \
--mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' \
--norm BN --num_L=3 --num_K=4
```
MELD:
```
--base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 \
--graph_type='hyper' --epochs=15 --graph_construct='direct' --multi_modal \
--mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' \
--norm BN --num_L=3 --num_K=3
```

## 우리 Q1 논문에서 위치
- **§2 Related Work**: Graph-based MERC baseline
- **§4 Comparison**: 경쟁 baseline 1 (비-LLM graph approach)
- 차별화: Emotion-LLaMA(LLM)과 M3NET(graph) 양쪽 비교해 우리 agent의 위치 확인

## Known caveats
- torch 버전 차이로 modern env에서 직접 실행 불가. 별도 env 필수
- Dropbox 링크가 rate limit/권한 문제로 다운로드 실패 가능. Retry 필요

## Next
1. 별도 env `m3net` 생성 (Week 2 Day 2)
2. Pre-extracted features 다운로드 (Dropbox)
3. IEMOCAP pretrained ckpt로 testing 모드 실행
4. 수치 재현 → leaderboard
