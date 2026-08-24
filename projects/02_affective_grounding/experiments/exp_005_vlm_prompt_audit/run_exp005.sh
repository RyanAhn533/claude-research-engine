#!/usr/bin/env bash
# exp_005 full autonomous chain: VLM audit (Qwen2.5-VL-7B 4bit) → analyze → engine log → validate.
set -u
CRE=/home/ajy/CLAUDE_RESEARCH_ENGINE
EXP=$CRE/projects/02_affective_grounding/experiments/exp_005_vlm_prompt_audit
LOG=$EXP/_chain.log
source /home/ajy/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate base-gpu-cu121 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date '+%F %T')] START exp_005 full run (3000 samples x 5 prompts) ===" | tee -a $LOG
python $EXP/run.py >> $LOG 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "=== [$(date '+%F %T')] run.py FAILED rc=$RC ===" | tee -a $LOG; exit $RC; fi

echo "=== [$(date '+%F %T')] analyze ===" | tee -a $LOG
python $EXP/analyze.py >> $LOG 2>&1

echo "=== [$(date '+%F %T')] engine log ===" | tee -a $LOG
python $EXP/log_exp005.py >> $LOG 2>&1 || echo "WARN: logging failed" | tee -a $LOG

echo "=== [$(date '+%F %T')] validate ===" | tee -a $LOG
python -m engine.cli.jy validate --project 02_affective_grounding >> $LOG 2>&1

echo "=== [$(date '+%F %T')] exp_005 CHAIN COMPLETE ===" | tee -a $LOG
