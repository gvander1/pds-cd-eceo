#!/usr/bin/env bash
# Feature analysis + transformer training + evaluation, for every encoder on both datasets.
# Usage (from ~/pdseceo):   bash scripts/run_all.sh          (seed 0)
#                           SEED=1 bash scripts/run_all.sh
cd "$(dirname "$0")/.."

ENCODERS="dinov2_vits14 resnet50_l3 dinov3_vits16 dinov3_vitl16 dinov3_vitl16_sat sam_vitb"
DATASETS="hiucd levir"
SEED=${SEED:-0}
declare -A HEADS=( [dinov2_vits14]=6 [dinov3_vits16]=6 [resnet50_l3]=16
                   [dinov3_vitl16]=16 [dinov3_vitl16_sat]=16 [sam_vitb]=4 )

run_one () {   # $1 = dataset, $2 = encoder
  local data=$1 enc=$2
  echo "========== $data | $enc | seed $SEED | $(date) =========="
  python scripts/extract_features.py --data $data --encoder $enc || return 1   # skipped if already cached
  python scripts/analyze_features.py --data $data --encoder $enc || return 1
  python scripts/train.py --data $data --encoder $enc --n_heads ${HEADS[$enc]} --seed $SEED || return 1
  local ckpt
  ckpt=$(ls -t checkpoints/${data}-${enc}-seed${SEED}-*.pt | head -1)
  python scripts/evaluate.py --ckpt "$ckpt" || return 1
}

mkdir -p logs
LOG=logs/run_all_seed${SEED}_$(date +%Y%m%d_%H%M).log
for enc in $ENCODERS; do
  for data in $DATASETS; do
    run_one $data $enc || echo "!!!!! FAILED: $data | $enc" 
  done
done 2>&1 | tee "$LOG"
echo "done, log in $LOG"