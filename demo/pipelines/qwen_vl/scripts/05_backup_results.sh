#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/ovr
mkdir -p "$ROOT/logs"
python -m pip freeze > "$ROOT/logs/requirements.txt"
nvidia-smi > "$ROOT/logs/gpu-info.txt"
tar -czf "$ROOT/qwen_pilot_backup.tar.gz" \
  -C "$ROOT" results scripts logs data/dataset_test
printf 'Backup: %s\n' "$ROOT/qwen_pilot_backup.tar.gz"
