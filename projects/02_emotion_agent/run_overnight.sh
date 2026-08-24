#!/bin/bash
# Master GPU serializer for overnight queue. Runs TR/TL on remaining models AFTER
# current jobs (exp_039 panel, exp_040 qwen7b) free the GPU. No concurrency.
PY=/home/ajy/miniconda3/envs/cre_q1/bin/python
export PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=3
cd /home/ajy/CLAUDE_RESEARCH_ENGINE/projects/02_emotion_agent
TT=experiments/exp_040_tr_tl

wait_gpu_free() {  # wait until no other gating/tr_tl/clean python running
  while pgrep -f "clean_gating.py|tr_tl.py|mech.py|nll_novelty.py|intervene.py" | grep -qv "^$$\$"; do sleep 60; done
}

echo "[ON] start $(date +%H:%M:%S)"
# 1) wait for exp_039 panel + exp_040 qwen7b to finish
wait_gpu_free
echo "[ON] gpu free, TR/TL remaining models $(date +%H:%M:%S)"
for m in qwen14b mistral yi6b falcon7b qwen3b; do
  echo "[ON] tr_tl $m $(date +%H:%M:%S)"
  $PY -u $TT/tr_tl.py --model $m > $TT/logs/${m}.log 2>&1
  echo "[ON] tr_tl $m: $(grep -E '/au|/iemocap|gain=' $TT/logs/${m}.log | tail -2 | tr '\n' ' ')"
done
echo "[ON] ALL TR/TL DONE $(date +%H:%M:%S)"
