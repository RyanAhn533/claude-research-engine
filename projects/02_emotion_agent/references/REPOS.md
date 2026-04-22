# Reference Repos — Emotion Agent Q1

단계별 reference할 github repo. 빠른 진행 목적.

---

## 🔴 Primary (직접 재현/비교 대상)

### Emotion-LLaMA — **NeurIPS 2024 (top-tier)**
- Repo: [`ZebangCheng/Emotion-LLaMA`](https://github.com/ZebangCheng/Emotion-LLaMA)
- Paper: Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning (NeurIPS 2024)
- **Why**: Multimodal emotion + LLM instruction-tuned. 우리 agent direction과 정면 비교
- **사용 단계**: Phase 2 (Week 2) SOTA reproduce
- **비고**: MERR 데이터셋 construction strategy 공개 (2025-07)

### M3NET — CVPR 2023
- Repo: [`feiyuchen7/M3NET`](https://github.com/feiyuchen7/M3NET)
- Paper: Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation
- **Why**: Graph NN 기반 MERC 표준. IEMOCAP/MELD preprocessing pattern 레퍼런스
- **사용 단계**: Phase 1 (preprocessing pattern), Phase 2 (baseline 비교)

### BeMERC — arXiv 2503.23990 (March 2025)
- Paper: [arxiv.org/html/2503.23990](https://arxiv.org/html/2503.23990)
- Repo: (공개 여부 확인 필요)
- **Why**: 가장 최근 MLLM-based MERC. LLM-emotion 선행
- **사용 단계**: Phase 2 — repo 공개되어 있으면 reproduce

### EEG Emotion Copilot — Neural Networks (IF 7.8)
- Paper: [ScienceDirect S0893608025007282](https://www.sciencedirect.com/science/article/abs/pii/S0893608025007282)
- Repo: (공개 여부 확인 필요)
- **Why**: Bio+LLM 가장 가까운 비교 대상
- **사용 단계**: Phase 2 — bio+LLM angle 비교

### MER Survey (EMNLP Findings 2025)
- Paper: [aclanthology.org/2025.findings-emnlp.332](https://aclanthology.org/2025.findings-emnlp.332)
- **Why**: 우리 Related Work의 backbone citation
- **사용 단계**: Phase 1-4 전반

---

## 🟡 Secondary (dataset/preprocessing)

### MELD (official)
- Repo: [`declare-lab/MELD`](https://github.com/declare-lab/MELD)
- **Why**: MELD benchmark 공식 schema + dev/test/train split
- **사용 단계**: Phase 1 (1.2)

### MISA (ACM MM 2020)
- Repo: [`declare-lab/MISA`](https://github.com/declare-lab/MISA) (있을 시)
- Paper: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
- **Why**: 전통 multimodal baseline
- **사용 단계**: optional baseline

### MMSA (multimodal sentiment analysis toolkit)
- Repo: [`thuiar/MMSA`](https://github.com/thuiar/MMSA)
- **Why**: MOSI/MOSEI 통합 benchmark toolkit
- **사용 단계**: Phase 2 CMU-MOSEI 필요 시

### MultiBench
- Repo: [`pliang279/MultiBench`](https://github.com/pliang279/MultiBench)
- **Why**: Multimodal learning benchmark 전반. fusion strategy 참고

---

## 🟢 Ecosystem (landscape / curated list)

### Awesome-Emotion-Reasoning — curated
- Repo: [`yuntaoshou/Awesome-Emotion-Reasoning`](https://github.com/yuntaoshou/Awesome-Emotion-Reasoning)
- **Why**: 최신 emotion reasoning (Qwen2.5-VL, Qwen2.5-Omni, Qwen3 등 포함) 전체 landscape
- **사용 단계**: Related work 조사, Phase 2 SOTA 추가 발굴

### Awesome-MMPs — physiological signals
- Repo: [`willxxy/awesome-mmps`](https://github.com/willxxy/awesome-mmps)
- **Why**: Bio + multimodal ML 리소스 — 우리 bio-grounded 축
- **사용 단계**: Phase 1 (DEAP), Phase 3 (bio-grounded)

### Awesome-Multimodal-LLMs
- Repo: [`BradyFU/Awesome-Multimodal-Large-Language-Models`](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- **Why**: MLLM landscape 전반

### Awesome-Unified-Multimodal
- Repo: [`AIDC-AI/Awesome-Unified-Multimodal-Models`](https://github.com/AIDC-AI/Awesome-Unified-Multimodal-Models)
- **Why**: Qwen3-Omni 포함 unified MLLM 추적

### Awesome-LLM-Apps (2025)
- Repo: [`mskj-apaas/awesome-llm-apps-2025`](https://github.com/mskj-apaas/awesome-llm-apps-2025)
- **Why**: LLM agent 설계 패턴 참고

---

## 🔧 Agent framework (우리 agent 구현)

### Qwen-Agent — **primary**
- Repo: [`QwenLM/Qwen-Agent`](https://github.com/QwenLM/Qwen-Agent)
- **Why**: 우리가 로컬에 둔 Qwen2.5-7B / VL-7B와 동일 family. Function calling, RAG 기본 지원
- **사용 단계**: Phase 3 agent prototype backbone

### Qwen2.5-VL (model family)
- Repo: [`QwenLM/Qwen2.5-VL`](https://github.com/QwenLM/Qwen2.5-VL)
- **Why**: Vision-language inference 패턴
- **사용 단계**: multimodal reasoning 필요 시

### Qwen2.5 (text family)
- Repo: [`QwenLM/Qwen2.5`](https://github.com/QwenLM/Qwen2.5)
- **Why**: text-only 기본

### OmAgent (multimodal agent)
- Repo: `OmAgent` (search 결과 링크)
- **Why**: multimodal 복잡 task agent framework 참고

---

## 💡 Approach/Methodology references

### VAEmo (ACM MM 2025) — **top-tier**
- Paper/Repo: "VAEmo: Efficient Representation Learning for Visual-Audio Emotion with Knowledge Injection" (ACM MM 2025)
- **Why**: Knowledge injection approach — 우리 cultural prior injection 과 유사 패턴

### LLaMAC (biosignal multimodal dataset)
- Paper: [PubMed 41345139](https://pubmed.ncbi.nlm.nih.gov/41345139/)
- **Why**: Low-cost biosensor + LLM dataset — 우리 Q1 방향 support

### MemoCMT (Scientific Reports 2025)
- Paper: [Nature article s41598-025-89202-x](https://www.nature.com/articles/s41598-025-89202-x)
- **Why**: Cross-modal transformer fusion 참고

### GS-MCC (2024)
- Paper: [arxiv 2404.17862](https://arxiv.org/html/2404.17862v1)
- **Why**: Graph spectrum multimodal consistency — MERC novel approach

---

## ⚠️ 재현 난이도 vs 우선순위

| Repo | 우선순위 | 재현 난이도 | 대안 |
|------|--------|---------|-----|
| Emotion-LLaMA | ⭐⭐⭐ | 중 (LLaMA-2, 4-bit 필요) | M3NET로 대체 |
| M3NET | ⭐⭐⭐ | 낮 (graph NN, 가벼움) | — |
| EEG Emotion Copilot | ⭐⭐ | 미확인 | arXiv E2-LLM (2601.07877) |
| BeMERC | ⭐ | 미확인 (repo 확인 필요) | skip OK |
| Qwen-Agent | ⭐⭐⭐ | 낮 (우리 Qwen 이미 있음) | — |

---

## 업데이트 규칙

- 매 phase 시작 전 **새 repo 발견되면 이 파일 append**
- 재현 성공/실패는 `state/paper_tried.jsonl`에 함께 기록
- 사용 안 한 repo는 "skipped" 표시
