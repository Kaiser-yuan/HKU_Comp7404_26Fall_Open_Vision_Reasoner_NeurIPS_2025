#!/usr/bin/env bash
set -euo pipefail
nvidia-smi
python - <<'CHECK'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
x = torch.randn(256, 256, device="cuda")
print("GPU test:", (x @ x).mean().item())
CHECK
df -h /root/autodl-tmp
python -m pip show transformers huggingface_hub accelerate pillow
