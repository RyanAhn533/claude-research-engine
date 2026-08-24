#!/bin/bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_038_format_intervention
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3
for m in qwen7b mistral qwen14b; do
  echo "[hard] $m $(date +%H:%M:%S)"
  $PY -u intervene.py --model $m > logs/${m}_5seed.log 2>&1
  echo "[hard] $m: $(grep H1 logs/${m}_5seed.log | tail -1)"
done
echo "[hard] ALL DONE $(date +%H:%M:%S)"
