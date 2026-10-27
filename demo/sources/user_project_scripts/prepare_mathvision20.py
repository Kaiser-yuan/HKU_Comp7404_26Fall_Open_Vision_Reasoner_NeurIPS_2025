"""Extract a reproducible 20-question sample from the official MATH-V repository."""
import argparse
import base64
import hashlib
import io
import json
import random
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REVISION = 'a1f5cc3add200c0cd080fad463e500f44ef1fb41'
SEED = 7404
COUNT = 20
API = 'https://api.github.com/repos/mathllm/MATH-V'

def fetch(url):
    return subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '3',
        '--retry-all-errors', '--max-time', '60', url],
        check=True, capture_output=True).stdout

def repo_file(path):
    obj = json.loads(fetch(f'{API}/contents/{path}?ref={REVISION}'))
    if obj.get('encoding') == 'base64':
        return base64.b64decode(obj['content'])
    blob = json.loads(fetch(obj['git_url']))
    return base64.b64decode(blob['content'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-jsonl', type=Path,
        help='Optional local copy of data/test.jsonl at the recorded revision')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Output directory must be empty; refusing to overwrite files')
    raw = args.source_jsonl.read_bytes() if args.source_jsonl else repo_file('data/test.jsonl')
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    assert len(rows) == 3040
    indices = sorted(random.Random(SEED).sample(range(len(rows)), COUNT))
    selected = [rows[i] for i in indices]
    (out / 'images').mkdir()
    def download_image(row):
        data = repo_file(row['image'])
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        target = out / row['image']
        target.write_bytes(data)
        return {'id': row['id'], 'image': row['image'],
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        images = list(pool.map(download_image, selected))
    def jsonl(name, records):
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    jsonl('questions.jsonl', [{k:r[k] for k in ['id','question','options','image']} for r in selected])
    jsonl('answers.jsonl', [{k:r[k] for k in ['id','answer','solution','level','subject']} for r in selected])
    jsonl('samples_full.jsonl', selected)
    manifest = {
        'dataset':'MathVision', 'repository':'https://github.com/mathllm/MATH-V',
        'revision':REVISION, 'source_file':'data/test.jsonl', 'source_split':'test',
        'source_count':len(rows), 'source_sha256':hashlib.sha256(raw).hexdigest(),
        'sample_count':COUNT, 'seed':SEED,
        'sampling':'Uniform random without replacement using Python random.Random(seed).sample; sorted by original row index',
        'row_indices_zero_based':indices, 'ids':[r['id'] for r in selected],
        'level_counts':dict(Counter(str(r['level']) for r in selected)),
        'subject_counts':dict(Counter(r['subject'] for r in selected)),
        'images':images, 'license':'MIT', 'created_date':'2026-10-22',
    }
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (out/'LICENSE').write_bytes(repo_file('LICENSE'))
    (out/'README.md').write_text('''# MathVision 固定 20 题试跑集

来源：MathVision 官方仓库 https://github.com/mathllm/MATH-V
原始划分：test（3040 题），不是官方 testmini。
版本：'''+REVISION+'''
抽样：固定种子 7404，无放回均匀抽取 20 题，再按原数据行序排列。

## 文件

- questions.jsonl：推理输入，含 id、question、options、image，不含标准答案或解答。
- answers.jsonl：评分与分析用标准答案、原始解答、难度及学科；部分原始解答为空。
- samples_full.jsonl：20 条完整原始记录，供核查。
- images/：原始图片，未裁剪或重编码。
- manifest.json：源版本、抽样行号、题目 ID、图片校验值及类别统计。
- LICENSE：原仓库 MIT 许可。

## 推理注意事项

图片路径相对于此目录。将 question 和 options 正确组成提示词，并单独向视觉处理器提供对应图片。
question 中原有的 <image1> 等标记予以保留；不应直接将其当作框架的视觉 token。正式推理脚本需统一处理这些标记及选项格式。
不要把 answers.jsonl 或 samples_full.jsonl 中的 answer、solution 传入模型。
这是用于流程调试和配对比较的 20 题子集，不代表完整基准，不能与论文完整测试集分数直接比较。
所有模型必须使用相同的 questions.jsonl 和图片。不要根据模型回答结果重新筛题。

## 复现抽样

在包含 Pillow 与 curl 的环境运行项目脚本：

```bash
python project/scripts/prepare_mathvision20.py --output /path/to/new_empty_directory
```

源仓库版本已经固定，不随 main 分支变化；本包中的完整版本号与校验值见 manifest.json。
''')
    assert len({r['id'] for r in selected}) == COUNT
    print(json.dumps({'count':COUNT,'ids':manifest['ids'],
        'levels':manifest['level_counts'],'subjects':manifest['subject_counts']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
