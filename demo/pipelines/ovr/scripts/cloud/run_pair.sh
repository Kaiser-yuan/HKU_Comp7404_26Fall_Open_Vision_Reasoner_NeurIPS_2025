#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OvrPython="$PWD/.venv-cloud/bin/python"
if [[ ! -x "$OvrPython" ]]; then
  echo 'Run bash scripts/cloud/setup.sh first.' >&2
  exit 1
fi
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_ENDPOINT=https://huggingface.co
# Run inside tmux if the Mac SSH connection may disconnect. The provider's
# timed shutdown must be set by the user; this script never controls billing.
"$OvrPython" -m ovr.cloud check --gpu --access
"$OvrPython" -m ovr.cloud download
for OvrModel in coldstart rl; do
  "$OvrPython" -m ovr.cloud run --model "$OvrModel" --phase canary
done
"$OvrPython" -m ovr.cloud report --phase canary
for OvrModel in coldstart rl; do
  "$OvrPython" -m ovr.cloud run --model "$OvrModel" --phase pilot
done
"$OvrPython" -m ovr.cloud report
"$OvrPython" -m ovr.cloud export
echo 'Export path is printed above by ovr.cloud export. Check coverage.json before claiming completeness.'
echo 'This process has finished; the provider instance is still running until you stop it.'
