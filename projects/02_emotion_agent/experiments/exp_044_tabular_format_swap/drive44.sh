#!/bin/bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_044_tabular_format_swap
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python; export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3
while pgrep -f "tab_swap.py --model qwen7b" >/dev/null; do sleep 30; done
for m in qwen14b falcon7b; do $PY -u tab_swap.py --model $m > logs/${m}.log 2>&1; echo "[d44] $m done"; done
echo "[d44] ALL DONE"
