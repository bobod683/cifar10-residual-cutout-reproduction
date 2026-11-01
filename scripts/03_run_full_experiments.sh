#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

variants=(plain20 resnet20 resnet20_cutout)
seeds=(42 3407 2026)

for variant in "${variants[@]}"; do
  for seed in "${seeds[@]}"; do
    run_dir="outputs/full/${variant}_seed${seed}"
    if [[ -f "${run_dir}/summary.json" && -f "${run_dir}/final_model.pt" ]]; then
      echo "SKIP completed ${variant} seed=${seed}"
      continue
    fi
    python train.py \
      --variant "${variant}" \
      --seed "${seed}" \
      --epochs 200 \
      --batch-size 128 \
      --num-workers 4 \
      --learning-rate 0.1 \
      --momentum 0.9 \
      --weight-decay 0.0001 \
      --cutout-length 16 \
      --data-root data \
      --output-root outputs/full \
      --download \
      --deterministic
  done
done

echo FULL_EXPERIMENTS_OK
