#!/usr/bin/env bash
set -euo pipefail
python -u /root/autodl-tmp/ovr/scripts/run_mathvision_test.py \
  --model /root/autodl-tmp/ovr/models/qwen \
  --dataset /root/autodl-tmp/ovr/data/dataset_test \
  --output /root/autodl-tmp/ovr/results/qwen_mathvision20_4096 \
  --max-new-tokens 4096
