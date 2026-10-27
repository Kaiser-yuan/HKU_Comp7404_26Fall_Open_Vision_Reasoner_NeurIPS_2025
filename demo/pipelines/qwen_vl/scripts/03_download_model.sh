#!/usr/bin/env bash
# Usage: source this script or execute it with bash; all settings apply to its process.
set -euo pipefail
MODEL_ID="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
DEST="${2:-/root/autodl-tmp/ovr/models/qwen}"
if [[ -f /etc/network_turbo ]]; then
    source /etc/network_turbo
fi
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
# The observed login saved its token at this location. Set it only if present.
if [[ -f /root/.cache/huggingface/token ]]; then
    export HF_TOKEN_PATH=/root/.cache/huggingface/token
fi
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_ENABLE_HF_TRANSFER
python - "$MODEL_ID" "$DEST" <<'DOWNLOAD'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=1)
print("Model download completed:", sys.argv[2])
DOWNLOAD
