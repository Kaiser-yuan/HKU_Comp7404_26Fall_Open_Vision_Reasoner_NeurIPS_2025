#!/usr/bin/env bash
# Reuse the matching vendor image without changing its base environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OvrBasePython="${OVR_BASE_PYTHON:-/root/miniconda3/bin/python}"
"$OvrBasePython" - <<'PY'
import importlib.metadata,sys,torch
assert sys.version_info[:2] == (3,12), 'This image route requires Python 3.12'
assert importlib.metadata.version('torch') == '2.8.0+cu128'
assert importlib.metadata.version('torchvision') == '0.23.0+cu128'
assert torch.version.cuda == '12.8'
PY
OvrVenvDir="$(readlink -m "$PWD/.venv-cloud")"
"$OvrBasePython" -m venv --system-site-packages "$OvrVenvDir"
OvrPython="$PWD/.venv-cloud/bin/python"
if ! "$OvrPython" -m pip install --timeout 30 --retries 2 --index-url "${OVR_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" pip==25.3 -r requirements-cloud.txt; then
  "$OvrPython" -m pip install --timeout 30 --retries 2 --index-url https://pypi.org/simple pip==25.3 -r requirements-cloud.txt
fi
"$OvrPython" -m pip check
mkdir -p .build
"$OvrPython" -m unittest discover -s tests -p 'test_cloud*.py' -q > .build/setup-image-tests.stdout.txt 2> .build/setup-image-tests.stderr.txt
export OVR_CLOUD_CONFIG="$PWD/configs/cloud_shared100.json"
if [[ "${1:-}" == "--no-gpu-check" ]]; then
  "$OvrPython" -m ovr.cloud check
else
  "$OvrPython" -m ovr.cloud check --gpu
fi
"$OvrPython" -m pip freeze > .build/environment-image.freeze.txt
echo 'Image environment ready. Use OVR_CLOUD_CONFIG=configs/cloud_shared100.json with ovr.cloud or run_pair.sh.'
