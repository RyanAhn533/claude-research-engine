#!/bin/bash
# Sequential multi-model gating panel. No GPU contention: waits for prior python to exit,
# and for each model's download to complete, before running.
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent/experiments/exp_035_multimodel_gating
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python
HUB=/mnt/hdd/ajy/caches/huggingface/hub
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3

declare -A SNAP=(
  [mistral]=models--mistralai--Mistral-7B-Instruct-v0.3
  [phi35]=models--microsoft--Phi-3.5-mini-instruct
  [qwen14b]=models--Qwen--Qwen2.5-14B-Instruct
)

wait_free() {  # wait until no gating python is running (avoid GPU contention)
  while pgrep -f "run_gating_batched.py" | grep -qv "^$$\$"; do sleep 15; done
}
wait_download() {  # wait until model fully downloaded (no .incomplete)
  local d="$HUB/${SNAP[$1]}"
  while [ "$(find "$d" -name '*.incomplete' 2>/dev/null | wc -l)" -gt 0 ] || [ ! -d "$d/snapshots" ]; do
    echo "[panel] waiting download: $1"; sleep 30
  done
}

for m in mistral phi35 qwen14b; do
  echo "[panel] ==== $m ===="
  wait_download "$m"
  wait_free
  echo "[panel] launching $m $(date +%H:%M:%S)"
  $PY -u run_gating_batched.py --model "$m" --batch_size 32 > "logs/${m}_batched.log" 2>&1
  echo "[panel] $m done $(date +%H:%M:%S): $(grep '\[SUMMARY\]' logs/${m}_batched.log | tail -1)"
done
echo "[panel] ALL DONE $(date +%H:%M:%S)"
