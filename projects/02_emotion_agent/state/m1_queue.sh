#!/bin/bash
# M1 experiment queue — 2026-04-23 night run (JY sleep)
# Sequence: exp_018 (running) -> exp_020 -> exp_021
set -u

LOGFILE=/home/ajy/claude-research-engine/projects/02_emotion_agent/state/m1_queue.log
EXP_DIR=/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments

echo "[$(date '+%Y-%m-%d %H:%M:%S')] M1 queue start" > "$LOGFILE"
echo "[$(date '+%H:%M:%S')] waiting for exp_018 (already running)..." >> "$LOGFILE"

while pgrep -f "exp_018_random_token_anchor/run.py" > /dev/null; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] exp_018 finished" >> "$LOGFILE"

# exp_020 class-coverage control
echo "[$(date '+%H:%M:%S')] exp_020 starting..." >> "$LOGFILE"
cd "$EXP_DIR/exp_020_class_coverage" && mkdir -p cache
PYTHONNOUSERSITE=1 /home/ajy/miniconda3/bin/conda run -n cre_q1 --no-capture-output python -u run.py > cache/run.log 2>&1
echo "[$(date '+%H:%M:%S')] exp_020 finished (exit=$?)" >> "$LOGFILE"

# exp_021 IEMOCAP Tier 1 + Tier 2
echo "[$(date '+%H:%M:%S')] exp_021 starting..." >> "$LOGFILE"
cd "$EXP_DIR/exp_021_iemocap_multiseed" && mkdir -p cache
PYTHONNOUSERSITE=1 /home/ajy/miniconda3/bin/conda run -n cre_q1 --no-capture-output python -u run.py > cache/run.log 2>&1
echo "[$(date '+%H:%M:%S')] exp_021 finished (exit=$?)" >> "$LOGFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] M1 queue complete" >> "$LOGFILE"

# Aggregate final results (best-effort)
{
  echo ""
  echo "=== FINAL RESULTS ==="
  for exp in exp_018_random_token_anchor exp_020_class_coverage exp_021_iemocap_multiseed; do
    rj=$(ls "$EXP_DIR/$exp/cache/"*.json 2>/dev/null | head -1)
    if [ -n "$rj" ]; then
      echo "--- $exp ---"
      /home/ajy/miniconda3/envs/cre_q1/bin/python -c "
import json, sys
d = json.load(open('$rj'))
for k, v in d.get('aggregate', {}).items():
    m = v.get('acc_mean', 0) * 100
    s = v.get('acc_std', 0) * 100
    fm = v.get('f1_mean', 0)
    fs = v.get('f1_std', 0)
    print(f'  {k}: acc={m:.2f}±{s:.2f}%, f1={fm:.3f}±{fs:.3f}')
" 2>&1
    fi
  done
} >> "$LOGFILE"
