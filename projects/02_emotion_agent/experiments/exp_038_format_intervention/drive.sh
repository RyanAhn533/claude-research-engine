#!/bin/bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_038_format_intervention
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3
for m in qwen7b mistral; do
  echo "[drive] $m $(date +%H:%M:%S)"
  $PY -u intervene.py --model $m > logs/${m}.log 2>&1
  echo "[drive] $m done: $(grep H1 logs/${m}.log | tail -1)"
done
echo "[drive] ALL DONE $(date +%H:%M:%S)"
