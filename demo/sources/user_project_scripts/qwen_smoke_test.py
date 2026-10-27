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
