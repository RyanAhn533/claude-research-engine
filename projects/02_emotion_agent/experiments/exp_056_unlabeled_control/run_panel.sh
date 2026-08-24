#!/bin/bash
# exp_056 full panel: 6 models x 3 datasets x 7 seeds x 4 conditions.
# Sequential (one model in VRAM at a time) to avoid OOM on A6000.
cd "$(dirname "$0")"
export USE_TF=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_CPP_MIN_LOG_LEVEL=3 HF_HUB_OFFLINE=1
for m in qwen3b qwen7b qwen14b yi6b falcon7b mistral; do
  echo "======== $m ========"
  python3 unlabeled_control.py --model "$m" --datasets au_nofacs,iemocap,meld 2>&1 \
    | grep -vE "UserWarning|warnings.warn|do_sample|only used in"
done
echo "======== ALL DONE ========"
