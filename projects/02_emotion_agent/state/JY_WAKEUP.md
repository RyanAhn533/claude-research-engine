# JY 일어나서 확인 (2026-04-24 morning — M1 overnight + dawn 세션 완료)

## TL;DR

- M1 queue (exp_018~025) ✅ 완료 02:11
- Follow-up (exp_026 MELD k-sweep + exp_019 + exp_019b attention) 진행 중
- **Q1_WORKING v2 대폭 재작성** (§1/§2 draft + §4 revised + §5 expanded + §6 future)
- **5 figures** regenerated (cross-domain 3-tier 포함)
- Commits pushed: a6765d1, dd02cb9, 256aae4 (3 batches)
- **Major revision**: "ICL universal 3-tier" → "ICL modality-gated, LoRA universal"



## 확인 순서 (우선순위)

```bash
# 1) 모든 실험 종합 결과
cat /home/ajy/claude-research-engine/projects/02_emotion_agent/state/m1_queue.log

# 2) 최신 paper draft
less /home/ajy/claude-research-engine/projects/02_emotion_agent/Q1_WORKING.md

# 3) 새 figures (5개)
ls -la /home/ajy/claude-research-engine/projects/02_emotion_agent/figures/*.png

# 4) Git log 최근 추가 commit
cd /home/ajy/claude-research-engine && git log --oneline -10
```

## 핵심 Headline Results

| Dataset | T1 zero-shot | T2 ICL k=4 | T3 LoRA |
|---------|-------------|-----------|---------|
| Korean FER AU | 29.08 ± 0.76 | 41.83 ± 3.41 (**+12.75**) | 55.00 ± 2.61 (**+13.17**) |
| IEMOCAP text | 47.83 ± 2.50 | 48.92 ± 4.25 (**+1.09**) | 70.00 ± 3.70 (**+21.08**) |
| MELD text | 55.50 ± 3.27 | 55.17 ± 5.65 (**−0.33**) | 61.75 ± 1.15 (+6.58) |

**LoRA universal, ICL Korean-FER-only**.

## 추가 발견

- **exp_018 random-token anchor**: σ 2.50 (중간) → Ekman 의미 있음 (1.63pp σ 기여)
- **exp_020 class-coverage**: H1 3-class σ=1.53 vs H2 4-class σ=4.34 — **class redundancy가 σ 주 요인 (2.81pp)**
- **exp_023 IEMOCAP k-sweep**: FLAT (46.67/46.17/47.00) — ICL 진짜 flat
- **exp_019 attention entropy**: D vs E 거의 동일 (0.645 vs 0.639) — **attention level 아닌 mechanism**



## 전체 Queue 계획 (9AM까지 ~10.5h packed)

### 현재 러닝 중 (m1_queue.sh)
1. **exp_018** Random-token anchor control — Paper B mechanism (~22:55 종료)
2. **exp_020** Class-coverage dissociation — Paper B H_COVERAGE vs H_ORIGIN (~23:10)
3. **exp_021** IEMOCAP T1+T2 cross-domain 3-tier (~00:30)

### 대기 중 (m1_queue_ext.sh — 자동 이어받음)
4. **exp_022** MELD T1+T2 cross-domain (~01:50)
5. **exp_023** IEMOCAP k-sweep (k=0,4,8, 3 seeds) — saturation 일반화 (~03:20)
6. **exp_024** IEMOCAP LoRA × 3 seeds — 3-tier 완성 on IEMOCAP (~05:00)
7. **exp_025** MELD LoRA × 3 seeds — 3-tier 완성 on MELD (~06:40)
8. 9AM까지 ~2h 버퍼

## 각 실험 의미 요약

| exp | Paper | 질문 | 기대 |
|-----|-------|------|------|
| 018 | B §4 | random anchor 효과? | G1 σ<1.5pp면 "structure matters, content irrelevant" 확정 |
| 020 | B §4 | σ 차이가 class-coverage냐 anchor origin이냐 | 분리 |
| 021 | A §4 | 3-tier가 IEMOCAP text에서도? | T2-T1 ≥ +5pp면 cross-domain 확증 |
| 022 | A §4 | MELD text에서도? | 비슷한 tier gap |
| 023 | A §4.2 | k-saturation IEMOCAP에서 재현? | k=0<<k=4≈k=8 |
| 024 | A §4.3 | IEMOCAP LoRA가 ICL 뛰어넘나? | +13pp over T2면 3-tier 완성 |
| 025 | A §4.3 | MELD도? | 동일 패턴 |

## 결과 확인 스크립트

```bash
# 종합 결과
grep -A2 "FINAL\|EXT FINAL" /home/ajy/claude-research-engine/projects/02_emotion_agent/state/m1_queue.log

# 개별 실험 JSON
ls /home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/exp_{018,020,021,022,023,024,025}_*/cache/*.json 2>/dev/null

# 각 실험 마지막 DONE 라인
for e in exp_018_random_token_anchor exp_020_class_coverage exp_021_iemocap_multiseed \
         exp_022_meld_multiseed exp_023_iemocap_kshot; do
  echo "=== $e ==="
  grep -A5 "^\[DONE\]" /home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/$e/cache/run.log 2>/dev/null | head -8
done

# LoRA per-seed 확인
for s in 42 123 777; do
  for exp in exp_024_iemocap_lora exp_025_meld_lora; do
    rj=/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments/$exp/cache/seed_$s/result.json
    [ -f "$rj" ] && echo "$exp seed=$s:" && cat "$rj"
  done
done

# GPU 현황
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
```

## Process 확인

```bash
ps -ef | grep -E "m1_queue|run.py|run_seed" | grep -v grep
# m1_queue.sh (이어받기 대기 중인 EXT 포함)
# 현재 running python
```

## 긴급 중단

```bash
pkill -f m1_queue
pkill -f "run.py\|run_seed.py"
```

## 결과 해석 가이드 — Paper 작성 시

### Paper B (SCI journal — Anchor Decomposition)
- exp_018: random-token 실험 → §4.3 mechanism figure
  - G1 σ<1.5: "structural diversity" 가설 확정 → strong paper
  - G1 σ>3: semantic content 필요 → weaker but still interesting
- exp_020: class-coverage 실험 → §4.4 dissociation figure
  - H_COVERAGE 입증 시: 기존 two-effect 해석 수정 필요
  - H_ORIGIN 입증 시: 기존 주장 강화

### Paper A (Conference — Adaptation Hierarchy)
- exp_021, 022: Text domain T1+T2 → §4.2 cross-domain table
- exp_023: k-sweep → §4.2 saturation figure on IEMOCAP
- exp_024, 025: LoRA → §4.3 cross-domain 3-tier 확정

### 최종 paper Table (expected)
```
Dataset       | T1 zero-shot | T2 ICL k=4 | T3 LoRA
Korean FER AU |  29.08±0.76  |  41.83±3.41 |  55.00±2.61   (exp_012/015)
IEMOCAP       |  TBD         |  TBD        |  TBD          (exp_021/024)
MELD          |  TBD         |  TBD        |  TBD          (exp_022/025)
```

모든 domain에서 T1<<T2<<T3 패턴 재현되면 **"adaptation hierarchy is universal"** claim 가능.

## 주의

- GPU 48GB, 일부 실험 병렬 실행 가능 (exp_018 + exp_020 현재 병렬 중)
- LoRA 실험은 30min training으로 무겁지만 OOM 위험 낮음
- Queue 전체 log: `state/m1_queue.log`
- 각 실험 stdout: `experiments/exp_NNN/cache/run.log` (또는 `seed_N.log`)

## Plan 참조
- `/home/ajy/.claude/plans/woolly-sniffing-karp.md` — master plan
- S-PACE/CBBF 엮기 layer 포함 (Paper B/A 각각 어떻게 인용할지)
