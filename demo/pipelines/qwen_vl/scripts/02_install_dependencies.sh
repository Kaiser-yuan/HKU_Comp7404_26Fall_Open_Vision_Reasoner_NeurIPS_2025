#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Disable the AutoDL academic proxy for package installation only.
unset http_proxy https_proxy
python -m pip install -r "$DEMO_DIR/requirements.txt"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; print('Import check passed')"
