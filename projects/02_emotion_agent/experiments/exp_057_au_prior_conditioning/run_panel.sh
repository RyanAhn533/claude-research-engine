#!/usr/bin/env bash
# exp_057 full panel: 6 models x 5 priors x 7 seeds x N=400, k=4.
# Sequential (single A6000, shared with other users). ~수 시간.
set -u
cd "$(dirname "$0")"
mkdir -p logs cache

PY=/home/ajy/miniconda3/envs/cre_q1/bin/python
export USE_TF=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1

PRIORS="${PRIORS:-A_none,B_textbook,C_derived,D_mouth,D_eyes}"
SEEDS="${SEEDS:-}"          # empty -> all 7
SHOTS="${SHOTS:-4}"

for m in qwen7b qwen14b qwen3b yi6b falcon7b mistral; do
  if [ -f "cache/au_prior_${m}.json" ]; then
    echo "[skip] $m (cache exists)"; continue
  fi
  echo "=== $m ==="
  $PY -u au_prior.py --model "$m" --priors "$PRIORS" --seeds "$SEEDS" --shots "$SHOTS" \
      > "logs/${m}.log" 2>&1
  tail -n 8 "logs/${m}.log"
done
echo "[done] $(ls cache/au_prior_*.json 2>/dev/null | wc -l) models"
