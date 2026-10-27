#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "$(uname -s)" != "Linux" ]]; then
  echo 'Run this installer on the rented Linux GPU machine.' >&2
  exit 1
fi
# Temporary per-command sources only; do not change the user's global pip config.
OvrPypiIndex="${OVR_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
OvrTorchIndex="${OVR_TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu126}"
ovr_pip_install() {
  local interpreter="$1"; shift
  if ! "$interpreter" -m pip install --timeout 30 --retries 2 --index-url "$OvrPypiIndex" "$@"; then
    echo 'PyPI mirror unavailable; retrying official PyPI once.' >&2
    "$interpreter" -m pip install --timeout 30 --retries 2 --index-url https://pypi.org/simple "$@"
  fi
}
OvrVenvDir="$(readlink -m "$PWD/.venv-cloud")"
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m venv "$OvrVenvDir"
elif python3 -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
  python3 -m venv "$OvrVenvDir"
else
  # Bootstrap in an isolated directory; do not replace the provider's Python.
  python3 -m venv .bootstrap
  ovr_pip_install .bootstrap/bin/python uv==0.9.5
  .bootstrap/bin/uv venv --python 3.11 --managed-python "$OvrVenvDir"
fi
OvrPython="$PWD/.venv-cloud/bin/python"
"$OvrPython" -m ensurepip --upgrade
ovr_pip_install "$OvrPython" pip==25.3
# Exact +cu126 pins prevent a fallback to a CPU or different CUDA build.
if ! "$OvrPython" -m pip install --timeout 30 --retries 2 --index-url "$OvrTorchIndex" --extra-index-url "$OvrPypiIndex" 'torch==2.8.0+cu126' 'torchvision==0.23.0+cu126'; then
  echo 'CUDA wheel mirror unavailable; retrying the official cu126 index once.' >&2
  "$OvrPython" -m pip install --timeout 30 --retries 2 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple 'torch==2.8.0+cu126' 'torchvision==0.23.0+cu126'
fi
ovr_pip_install "$OvrPython" -r requirements-cloud.txt
"$OvrPython" -m pip check
mkdir -p .build
if ! "$OvrPython" -m unittest discover -s tests -p 'test_cloud*.py' -q > .build/setup-tests.stdout.txt 2> .build/setup-tests.stderr.txt; then
  cat .build/setup-tests.stderr.txt >&2
  exit 1
fi
echo 'Offline pipeline tests passed using temporary fixtures; no OVR generation has run.'
mkdir -p results/runs/cloud_pilot20
"$OvrPython" -m pip freeze > results/runs/cloud_pilot20/environment.freeze.txt
if [[ "${1:-}" == "--no-gpu-check" ]]; then
  "$OvrPython" -m ovr.cloud check
else
  "$OvrPython" -m ovr.cloud check --gpu
fi
echo 'Environment ready. Log in with HF_ENDPOINT=https://huggingface.co .venv-cloud/bin/huggingface-cli login (do not add token to git).'
echo 'Then run bash scripts/cloud/run_pair.sh.'
