"""Extend the original fixed MathVision sample without using model outcomes."""
import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image

REVISION = 'a1f5cc3add200c0cd080fad463e500f44ef1fb41'
SEED = 740480


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository-zip', type=Path, required=True)
    parser.add_argument('--original20', type=Path, required=True)
    parser.add_argument('--extra80', type=Path, required=True)
    parser.add_argument('--combined100', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.original20/'manifest.json').read_text())
    assert old['revision'] == REVISION and len(set(old['ids'])) == 20
    with zipfile.ZipFile(args.repository_zip) as z:
        prefix = f'MATH-V-{REVISION}/'
        raw = z.read(prefix+'data/test.jsonl')
        assert hashlib.sha256(raw).hexdigest() == old['source_sha256']
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        pool = [i for i,r in enumerate(rows) if r['id'] not in set(old['ids'])]
        indices = sorted(random.Random(SEED).sample(pool,80))
        extra = [rows[i] for i in indices]
        combined_indices = sorted(indices + old['row_indices_zero_based'])
        combined = [rows[i] for i in combined_indices]
        assert len(extra)==80 and not set(r['id'] for r in extra)&set(old['ids'])
        assert len({r['id'] for r in combined})==100
        for out, selected, selected_indices, label in [
            (args.extra80,extra,indices,'extra80'),
            (args.combined100,combined,combined_indices,'combined100')]:
            out.mkdir(parents=True,exist_ok=True)
            if any(out.iterdir()):raise RuntimeError(f'Refusing to overwrite {out}')
            (out/'images').mkdir()
            image_meta=[]
            for row in selected:
                image=z.read(prefix+row['image'])
                with Image.open(io.BytesIO(image)) as im:im.verify()
                (out/row['image']).write_bytes(image)
                image_meta.append({'id':row['id'],'image':row['image'],
                    'sha256':hashlib.sha256(image).hexdigest(),'bytes':len(image)})
            def jsonl(name,records):
                (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            jsonl('questions.jsonl',[{k:r[k] for k in ['id','question','options','image']} for r in selected])
            jsonl('answers.jsonl',[{k:r[k] for k in ['id','answer','solution','level','subject']} for r in selected])
            jsonl('samples_full.jsonl',selected)
            manifest={
                'dataset':'MathVision','repository':'https://github.com/mathllm/MATH-V',
                'revision':REVISION,'source_split':'test','source_file':'data/test.jsonl',
                'source_count':len(rows),'source_sha256':old['source_sha256'],
                'sample_count':len(selected),'sample_group':label,
                'additional_sampling_seed':SEED,'original_sampling_seed':old['seed'],
                'original20_ids':old['ids'],'new80_ids':[r['id'] for r in extra],
                'sampling':'Keep original20; uniformly sample 80 without replacement from the remaining 3020 source rows using Random(740480); sort by original row index.',
                'row_indices_zero_based':selected_indices,'ids':[r['id'] for r in selected],
                'level_counts':dict(Counter(str(r['level']) for r in selected)),
                'subject_counts':dict(Counter(r['subject'] for r in selected)),
                'images':image_meta,'license':'MIT','created_date':'2026-10-22',
            }
            (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
            (out/'LICENSE').write_bytes(z.read(prefix+'LICENSE'))
            (out/'README.md').write_text(f'''# MathVision {label}

Source: official MATH-V `test` split, 3040 questions.
Revision: {REVISION}
This directory contains {len(selected)} questions and original images; it is NOT official testmini.
The original 20 IDs are retained; 80 new IDs were drawn without replacement from the
remaining 3020 rows using seed {SEED}. No model outputs, scores, subjects or difficulty
labels were used to choose the new rows. The two-stage selection is documented in manifest.json.

questions.jsonl contains no answers; answers.jsonl is for post-generation scoring.
Images are relative to this directory. samples_full.jsonl preserves source records.
Do not send answer or solution fields to the model. Keep the previous 20-question result
unchanged. Use identical prompt, precision, image processing and generation settings for
additional questions and all model checkpoints before combining results.
''')
            print(label, len(selected),'levels',manifest['level_counts'])

if __name__=='__main__':main()
