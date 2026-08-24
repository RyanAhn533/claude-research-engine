#!/bin/bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_045_sentiment_leet
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python; export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3 HF_HOME=/mnt/hdd/ajy/caches/huggingface
while pgrep -f "tab_swap.py|sent_leet.py --model qwen7b" >/dev/null; do sleep 30; done
for m in qwen14b falcon7b mistral; do $PY -u sent_leet.py --model $m > logs/${m}.log 2>&1; echo "[d45] $m: $(grep 'H]' logs/${m}.log|tail -1)"; done
echo "[d45] ALL DONE"
