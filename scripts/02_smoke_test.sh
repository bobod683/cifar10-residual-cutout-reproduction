#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train.py \
  --variant resnet20 \
  --seed 42 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2 \
  --limit-train 512 \
  --limit-test 256 \
  --data-root data \
  --output-root outputs/smoke \
  --download

test -f outputs/smoke/resnet20_seed42/summary.json
test -f outputs/smoke/resnet20_seed42/final_model.pt
echo SMOKE_TEST_OK
