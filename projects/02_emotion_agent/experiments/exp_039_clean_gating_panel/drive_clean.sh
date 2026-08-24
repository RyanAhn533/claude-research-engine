#!/bin/bash
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_039_clean_gating_panel
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python
HUB=/mnt/hdd/ajy/caches/huggingface/hub
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3
declare -A SNAP=( [qwen7b]=Qwen--Qwen2.5-7B-Instruct [qwen14b]=Qwen--Qwen2.5-14B-Instruct [mistral]=mistralai--Mistral-7B-Instruct-v0.3 [qwen3b]=Qwen--Qwen2.5-3B-Instruct [yi6b]=01-ai--Yi-1.5-6B-Chat [falcon7b]=tiiuae--Falcon3-7B-Instruct )
for m in qwen7b qwen14b mistral qwen3b yi6b falcon7b; do
  d="$HUB/models--${SNAP[$m]}"
  # wait for download complete
  tries=0
  while [ "$(find "$d" -name '*.incomplete' 2>/dev/null | wc -l)" -gt 0 ] || [ ! -d "$d/snapshots" ]; do
    echo "[clean] waiting dl: $m"; sleep 30; tries=$((tries+1)); [ $tries -gt 80 ] && { echo "[clean] SKIP $m (dl timeout)"; continue 2; }
  done
  echo "[clean] === $m $(date +%H:%M:%S) ==="
  $PY -u clean_gating.py --model $m > logs/${m}.log 2>&1
  echo "[clean] $m: $(grep '>>>' logs/${m}.log | tail -1)"
done
echo "[clean] ALL DONE $(date +%H:%M:%S)"
