#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
mkdir -p "$DEMO_DIR/results"
OUT="$(mktemp -d "$DEMO_DIR/results/reproduced.XXXXXX")"
"$PYTHON_BIN" "$DEMO_DIR/scripts/merge_mathvision_runs.py" \
 --runs "$DEMO_DIR/results/qwen_mathvision20_4096" "$DEMO_DIR/results/qwen_mathvision_extra80_4096" \
 --dataset "$DEMO_DIR/data/dataset_test100" --output "$OUT/qwen_merged"
"$PYTHON_BIN" "$DEMO_DIR/scripts/review_mathvision_pilot.py" \
 --run "$OUT/qwen_merged" --dataset "$DEMO_DIR/data/dataset_test100" --output "$OUT/qwen_review"
"$PYTHON_BIN" "$DEMO_DIR/scripts/review_mathvision_pilot.py" \
 --run "$DEMO_DIR/results/vl_rethinker_mathvision100_4096" \
 --dataset "$DEMO_DIR/data/dataset_test100" --output "$OUT/vl_review"
"$PYTHON_BIN" "$DEMO_DIR/scripts/compare_mathvision_pilots.py" \
 --qwen-run "$OUT/qwen_merged" --vl-run "$DEMO_DIR/results/vl_rethinker_mathvision100_4096" \
 --qwen-review "$OUT/qwen_review" --vl-review "$OUT/vl_review" \
 --dataset "$DEMO_DIR/data/dataset_test100" --output "$OUT/comparison"
printf 'Offline analysis saved to: %s\n' "$OUT"
