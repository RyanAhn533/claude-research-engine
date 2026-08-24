#!/bin/bash
# SR-XMA E1+E0: 4모드 × 3seed (learned s1111은 smoke로 별도 진행 중 → 여기선 11런)
# 2개 동시(순차 큐). MOSEI, MDROP=0(clean efficiency), share=specific.
set -u
cd /home/ajy/HT-DAR/code
mkdir -p logs_srxma_e1
export MDROP=0.0

run_one() {
  local MODE=$1 SEED=$2
  echo "[START $(date +%H:%M:%S)] $MODE s$SEED" >> logs_srxma_e1/_queue.log
  python train_stripped.py "$MODE" specific "$SEED" mosei > "logs_srxma_e1/${MODE}_s${SEED}.log" 2>&1
  echo "[DONE  $(date +%H:%M:%S)] $MODE s$SEED (exit $?)" >> logs_srxma_e1/_queue.log
}
export -f run_one

# 11 jobs (SR_LEARN:1111 은 smoke 진행 중이라 제외)
JOBS="SR_LEARN:1112 SR_LEARN:1113 \
SR_RAND:1111 SR_RAND:1112 SR_RAND:1113 \
SR_FIXED:1111 SR_FIXED:1112 SR_FIXED:1113 \
SR_FULL:1111 SR_FULL:1112 SR_FULL:1113"

echo "[QUEUE START $(date)] 11 runs, -P2" > logs_srxma_e1/_queue.log
printf "%s\n" $JOBS | xargs -P 2 -I{} bash -c 'IFS=: read m s <<< "$1"; run_one "$m" "$s"' _ {}
echo "[QUEUE DONE $(date)] ALL 11 E1 RUNS FINISHED" >> logs_srxma_e1/_queue.log
