#!/bin/bash
# M1 extension queue — runs after m1_queue.sh finishes
# Sequence: exp_022 MELD -> exp_023 IEMOCAP k-sweep -> exp_024 IEMOCAP LoRA×3 -> exp_025 MELD LoRA×3
set -u

LOGFILE=/home/ajy/claude-research-engine/projects/02_emotion_agent/state/m1_queue.log
EXP_DIR=/home/ajy/claude-research-engine/projects/02_emotion_agent/experiments
CONDA=/home/ajy/miniconda3/bin/conda

echo "[$(date '+%H:%M:%S')] EXT queue waiting for m1_queue.sh..." >> "$LOGFILE"
while pgrep -f "m1_queue.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXT queue start" >> "$LOGFILE"

run_exp() {
    local name="$1" cmd="$2"
    echo "[$(date '+%H:%M:%S')] $name starting..." >> "$LOGFILE"
    eval "$cmd"
    echo "[$(date '+%H:%M:%S')] $name finished (exit=$?)" >> "$LOGFILE"
}

# exp_022 MELD T1+T2
cd "$EXP_DIR/exp_022_meld_multiseed" && mkdir -p cache
run_exp "exp_022" "PYTHONNOUSERSITE=1 $CONDA run -n cre_q1 --no-capture-output python -u run.py > cache/run.log 2>&1"

# exp_023 IEMOCAP k-sweep
cd "$EXP_DIR/exp_023_iemocap_kshot" && mkdir -p cache
run_exp "exp_023" "PYTHONNOUSERSITE=1 $CONDA run -n cre_q1 --no-capture-output python -u run.py > cache/run.log 2>&1"

# exp_024 IEMOCAP LoRA × 3 seeds
for s in 42 123 777; do
    cd "$EXP_DIR/exp_024_iemocap_lora" && mkdir -p cache
    run_exp "exp_024_seed${s}" "PYTHONNOUSERSITE=1 $CONDA run -n cre_q1 --no-capture-output python -u run_seed.py $s > cache/seed${s}.log 2>&1"
done

# exp_025 MELD LoRA × 3 seeds
for s in 42 123 777; do
    cd "$EXP_DIR/exp_025_meld_lora" && mkdir -p cache
    run_exp "exp_025_seed${s}" "PYTHONNOUSERSITE=1 $CONDA run -n cre_q1 --no-capture-output python -u run_seed.py $s > cache/seed${s}.log 2>&1"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXT queue complete" >> "$LOGFILE"

# Final aggregate
{
  echo ""
  echo "=== EXT FINAL RESULTS ==="
  for exp in exp_022_meld_multiseed exp_023_iemocap_kshot; do
    rj=$(ls "$EXP_DIR/$exp/cache/"*.json 2>/dev/null | head -1)
    [ -n "$rj" ] && {
      echo "--- $exp ---"
      /home/ajy/miniconda3/envs/cre_q1/bin/python -c "
import json
d = json.load(open('$rj'))
for k, v in d.get('aggregate', {}).items():
    print(f\"  {k}: acc={v.get('acc_mean',0)*100:.2f}±{v.get('acc_std',0)*100:.2f}%\")
" 2>&1
    }
  done
  echo "--- exp_024 (IEMOCAP LoRA) per-seed ---"
  for s in 42 123 777; do
    rj="$EXP_DIR/exp_024_iemocap_lora/cache/seed_${s}/result.json"
    [ -f "$rj" ] && /home/ajy/miniconda3/envs/cre_q1/bin/python -c "
import json
d = json.load(open('$rj'))
print(f\"  seed={d['seed']}: acc={d['finetuned']['acc']*100:.2f}%, f1={d['finetuned']['f1']:.3f}\")
" 2>&1
  done
  echo "--- exp_025 (MELD LoRA) per-seed ---"
  for s in 42 123 777; do
    rj="$EXP_DIR/exp_025_meld_lora/cache/seed_${s}/result.json"
    [ -f "$rj" ] && /home/ajy/miniconda3/envs/cre_q1/bin/python -c "
import json
d = json.load(open('$rj'))
print(f\"  seed={d['seed']}: acc={d['finetuned']['acc']*100:.2f}%, f1={d['finetuned']['f1']:.3f}\")
" 2>&1
  done
} >> "$LOGFILE"
