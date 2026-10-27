# Historical command archive

Not a sequential setup script. Includes failed, completed and pending commands.
Follow README.md for current run instructions. No secret token values are recorded.

## Block 1

```
nvidia-smi
```

## Block 2

```
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
    ))
    x = torch.randn(256, 256, device="cuda")
    print("GPU test:", (x @ x).mean().item())
PY
```

## Block 3

```
df -h /root/autodl-tmp
python -m pip show transformers vllm
```

## Block 4

```
hf auth login
```

## Block 5

```
curl -I --connect-timeout 10 --max-time 20 https://huggingface.co
```

## Block 6

```
source /etc/network_turbo
```

## Block 7

```
curl -I --connect-timeout 10 --max-time 20 https://huggingface.co
```

## Block 8

```
hf auth login
hf auth whoami
```

## Block 9

```
unset http_proxy https_proxy
```

## Block 10

```
python -m pip install "huggingface_hub[cli]==0.36.0"
hf --help
```

## Block 11

```
mkdir -p /root/autodl-tmp/ovr/{models,data,scripts,results,logs}
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
cd /root/autodl-tmp/ovr
```

## Block 12

```
export HF_TOKEN_PATH=/root/.cache/huggingface/token
hf auth whoami
```

## Block 13

```
python -m pip install \
  "transformers==4.57.1" \
  "huggingface_hub[cli]==0.36.0" \
  accelerate qwen-vl-utils pillow pandas tqdm
```

## Block 14

```
python -c "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; print('模型依赖检查通过')"
```

## Block 15

```
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /root/autodl-tmp/ovr/models/qwen
```

## Block 16

```
hf download Kangheng/OVR-7B-ColdStart \
  --local-dir /root/autodl-tmp/ovr/models/ovr-coldstart
```

## Block 17

```
hf download Kangheng/OVR-7B-RL \
  --local-dir /root/autodl-tmp/ovr/models/ovr-rl
```

## Block 18

```
python -m pip freeze > /root/autodl-tmp/ovr/logs/requirements.txt
nvidia-smi > /root/autodl-tmp/ovr/logs/gpu-info.txt
```

## Block 19

```
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
export HF_TOKEN_PATH=/root/.cache/huggingface/token
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_ENABLE_HF_TRANSFER
```

## Block 20

```
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="/root/autodl-tmp/ovr/models/qwen",
    max_workers=1,
)
print("Qwen 模型下载完成")
PY
```

## Block 21

```
unset http_proxy https_proxy
python -m pip install "transformers==4.57.1" accelerate pillow
python -c "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; print('模型依赖检查通过')"
```

## Block 22

```
python /root/autodl-tmp/ovr/scripts/qwen_smoke_test.py
```

## Block 23

```
"""Run on the AutoDL server after downloading Qwen. Not a benchmark."""
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

root = Path('/root/autodl-tmp/ovr')
model_dir = root / 'models/qwen'
for folder in ['data', 'results', 'logs']:
    (root / folder).mkdir(parents=True, exist_ok=True)

image = Image.new('RGB', (448, 224), 'white')
draw = ImageDraw.Draw(image)
for x in [24, 168, 312]:
    draw.ellipse((x, 64, x + 96, 160), fill='red')
image_path = root / 'data/smoke_test.png'
image.save(image_path)
question = 'How many red circles are in the image? Answer with one number.'
messages = [{'role': 'user', 'content': [
    {'type': 'image'}, {'type': 'text', 'text': question},
]}]

print('Loading local model...', flush=True)
started = time.perf_counter()
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, dtype=torch.bfloat16, device_map='auto',
    attn_implementation='sdpa', local_files_only=True,
).eval()
torch.cuda.synchronize()
load_seconds = time.perf_counter() - started
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
generation_seconds = time.perf_counter() - started
new_ids = output[:, inputs.input_ids.shape[1]:]
answer = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
result = {
    'model_path': str(model_dir), 'image': str(image_path),
    'question': question, 'expected_answer': '3', 'answer': answer,
    'load_seconds': load_seconds, 'generation_seconds': generation_seconds,
    'generated_tokens': int(new_ids.shape[1]),
    'peak_allocated_vram_gib': torch.cuda.max_memory_allocated() / 1024**3,
    'torch_version': torch.__version__, 'dtype': 'bfloat16',
    'attention': 'sdpa', 'max_new_tokens': 64, 'do_sample': False,
}
result_path = root / 'results/qwen_smoke_test.json'
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
print('Result saved:', result_path)
```

## Block 24

```
ls -l /root/qwen_smoke_test*
```

## Block 25

```
mkdir -p /root/autodl-tmp/ovr/scripts
mv -i /root/qwen_smoke_test.py /root/autodl-tmp/ovr/scripts/
python /root/autodl-tmp/ovr/scripts/qwen_smoke_test.py
```

## Block 26

```
python -m pip freeze > /root/autodl-tmp/ovr/logs/requirements.txt
nvidia-smi > /root/autodl-tmp/ovr/logs/gpu-info.txt
df -h /root/autodl-tmp
```

## Block 27

```
python project/scripts/prepare_mathvision20.py --output /path/to/new_empty_directory
```

## Block 28

```
mkdir -p /root/autodl-tmp/ovr
unzip -n /root/mathvision20_test_bundle.zip -d /root/autodl-tmp/ovr
```

## Block 29

```
python -u /root/autodl-tmp/ovr/scripts/run_mathvision_test.py \
  --model /root/autodl-tmp/ovr/models/qwen \
  --dataset /root/autodl-tmp/ovr/data/dataset_test \
  --output /root/autodl-tmp/ovr/results/qwen_mathvision20_4096 \
  --max-new-tokens 4096
```

## Block 30

```
cd /root/autodl-tmp/ovr
python -m pip freeze > logs/requirements.txt
nvidia-smi > logs/gpu-info.txt
tar -czf qwen_pilot_backup.tar.gz results scripts logs data/dataset_test
```

## Block 31

```
python project/scripts/review_mathvision_pilot.py \
  --run project/test/dataset_result/qwen_pilot_backup/results/qwen_mathvision20_4096 \
  --dataset project/test/dataset_test \
  --output project/test/dataset_result/qwen_review
```

## Block 32

```
df -h /root/autodl-tmp
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
export HF_TOKEN_PATH=/root/.cache/huggingface/token
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_ENABLE_HF_TRANSFER
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Kangheng/OVR-7B-ColdStart",
    local_dir="/root/autodl-tmp/ovr/models/ovr-coldstart",
    max_workers=1,
)
print("OVR-ColdStart download completed")
PYTHON
```

## Block 33

```
unzip -n /root/mathvision_extra80_test_bundle.zip -d /root/autodl-tmp/ovr
python -u /root/autodl-tmp/ovr/scripts/run_mathvision_test_v2.py \
  --model /root/autodl-tmp/ovr/models/qwen \
  --dataset /root/autodl-tmp/ovr/data/dataset_test_extra80 \
  --output /root/autodl-tmp/ovr/results/qwen_mathvision_extra80_4096 \
  --max-new-tokens 4096
```

## Block 34

```
cd /root/autodl-tmp/ovr
tar -czf qwen_100_pilot_backup.tar.gz results scripts logs data/dataset_test data/dataset_test_extra80
```

## Block 35

```
cd /root/autodl-tmp/ovr
python -m pip freeze > logs/requirements.txt
nvidia-smi > logs/gpu-info.txt
tar -czf qwen_100_pilot_backup.tar.gz results scripts logs data/dataset_test data/dataset_test_extra80
```

## Block 36

```
unzip -n /root/vl_rethinker100_test_bundle.zip -d /root/autodl-tmp/ovr
df -h /root/autodl-tmp
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
export HF_TOKEN_PATH=/root/.cache/huggingface/token
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_ENABLE_HF_TRANSFER
python /root/autodl-tmp/ovr/scripts/download_checkpoint.py \
  --repo TIGER-Lab/VL-Rethinker-7B \
  --output /root/autodl-tmp/ovr/models/vl-rethinker
```

## Block 37

```
python -u /root/autodl-tmp/ovr/scripts/run_mathvision_test_v3.py \
  --model /root/autodl-tmp/ovr/models/vl-rethinker \
  --dataset /root/autodl-tmp/ovr/data/dataset_test100 \
  --output /root/autodl-tmp/ovr/results/vl_rethinker_mathvision100_4096 \
  --max-new-tokens 4096
```

## Block 38

```
cd /root/autodl-tmp/ovr
tar -czf qwen_vl_rethinker_backup.tar.gz results scripts logs data/dataset_test data/dataset_test_extra80 data/dataset_test100 models/vl-rethinker/download_metadata.json
```

## Block 39

```
find /root -maxdepth 5 -type f -name '*rethinker*zip' 2>/dev/null
```

## Block 40

```
unzip -n /root/vl_rethinker100_test_bundle.zip -d /root/autodl-tmp/ovr
ls -l /root/autodl-tmp/ovr/scripts/download_checkpoint.py /root/autodl-tmp/ovr/scripts/run_mathvision_test_v3.py /root/autodl-tmp/ovr/data/dataset_test100/questions.jsonl
```

## Block 41

```
python -u /root/autodl-tmp/ovr/scripts/run_mathvision_test_v3.py \
  --model /root/autodl-tmp/ovr/models/vl-rethinker \
  --dataset /root/autodl-tmp/ovr/data/dataset_test100 \
  --output /root/autodl-tmp/ovr/results/vl_rethinker_mathvision100_4096 \
  --max-new-tokens 4096
```

## Block 42

```
cd /root/autodl-tmp/ovr
python -m pip freeze > logs/requirements.txt
nvidia-smi > logs/gpu-info.txt
tar -czf qwen_vl_rethinker_backup.tar.gz results scripts logs data/dataset_test data/dataset_test_extra80 data/dataset_test100 models/vl-rethinker/download_metadata.json
```

## Block 43

```
cd project/technical_demo
PYTHON=python3 bash scripts/06_reproduce_analysis.sh
```