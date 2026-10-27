"""Import the teammate's frozen questions without using model outcomes."""
import collections
import hashlib
import io
import json
import random
import re
import stat
import zipfile
from pathlib import Path
from .common import ROOT, digest, read_jsonl, unique, write_json, write_jsonl

BUNDLE_SHA256 = 'ea39a4232591b4dffa41bde55a6288d3e0a1be075620bfed22ffc7308d12d19e'


def build_question(row):
    question = re.sub(r'<image\d+>', '', row['question']).strip()
    options = row.get('options', [])
    if options:
        question += '\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {text}' for i, text in enumerate(options))
    return question + '\nSolve the problem step by step. Put your final answer in \\boxed{}. ' + (
        'For this multiple-choice question, put only the option letter in the box.'
        if options else 'Put only the final value or expression in the box.')


def import_bundle(archive_path, root=ROOT):
    from PIL import Image
    if digest(archive_path) != BUNDLE_SHA256:
        raise ValueError('Not the supplied frozen teammate bundle')
    out = root / 'workstreams/shared_test100'
    if out.exists():
        raise ValueError('Shared100 already exists; validate it instead of overwriting')
    with zipfile.ZipFile(archive_path) as z:
        entries = z.infolist()
        if len({i.filename for i in entries}) != len(entries) or sum(i.file_size for i in entries) > 15_000_000:
            raise ValueError('Unexpected archive structure or size')
        for i in entries:
            p = Path(i.filename)
            if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(i.external_attr >> 16) or not i.filename.startswith('dataset_test100/'):
                raise ValueError('Unsafe archive member')
        if z.testzip():
            raise ValueError('Archive CRC mismatch')
        prefix = 'dataset_test100/'
        def get(name):
            return z.read(prefix + name)
        def rows(name):
            return [json.loads(s) for s in get(name).decode('utf-8-sig').splitlines() if s.strip()]
        questions, answers, full = rows('questions.jsonl'), rows('answers.jsonl'), rows('samples_full.jsonl')
        manifest = json.loads(get('manifest.json'))
        ids = [str(r['id']) for r in questions]
        if len(ids) != 100 or len(set(ids)) != 100 or ids != manifest['ids']:
            raise ValueError('Question IDs differ from the frozen list')
        refs, full_by_id = unique(answers, 'id'), unique(full, 'id')
        if set(refs) != set(ids) or set(full_by_id) != set(ids):
            raise ValueError('Reference/full-record coverage differs')
        original = {str(i+1) for i in random.Random(7404).sample(range(3040), 20)}
        remaining = [i for i in range(3040) if str(i+1) not in original]
        additional = {str(i+1) for i in random.Random(740480).sample(remaining, 80)}
        if original != set(manifest['original20_ids']) or additional != set(manifest['new80_ids']) or set(ids) != original | additional:
            raise ValueError('Two-stage seeded selection cannot be reproduced')
        if [int(q)-1 for q in ids] != manifest['row_indices_zero_based'] or ids != sorted(ids, key=int):
            raise ValueError('Question order differs from source-row order')
        images = unique(manifest['images'], 'id')
        inputs, references, sizes = [], [], []
        official = unique(read_jsonl(root / 'data/questions.jsonl'))
        official_checked = 0
        for q in questions:
            qid = str(q['id']); a = refs[qid]; f = full_by_id[qid]
            if set(q) != {'id', 'question', 'options', 'image'}:
                raise ValueError('Input record has unexpected fields, possibly references')
            if any(q[k] != f[k] for k in ('question', 'options')) or any(a[k] != f[k] for k in ('answer', 'level', 'subject')):
                raise ValueError('Source question/reference mismatch')
            if official:
                o = official[qid]
                if any(o[k] != f[k] for k in ('question', 'options', 'answer', 'level', 'subject')):
                    raise ValueError('Question metadata disagrees with the locally pinned official benchmark')
                official_checked += 1
            image_name = q['image']
            if Path(image_name).is_absolute() or '..' in Path(image_name).parts or image_name != images[qid]['image']:
                raise ValueError('Invalid image reference')
            blob = get(image_name)
            sha = hashlib.sha256(blob).hexdigest()
            if sha != images[qid]['sha256'] or len(blob) != images[qid]['bytes']:
                raise ValueError('Original image hash/size mismatch')
            with Image.open(io.BytesIO(blob)) as im:
                sizes.append(im.size); im.verify()
            inputs.append({'question_id': qid, 'question': q['question'], 'options': q['options'],
                           'prompt': build_question(q), 'image': 'source/' + image_name, 'image_sha256': sha,
                           'level': a['level'], 'subject': a['subject']})
            references.append({'question_id': qid, **{k: f[k] for k in ('question', 'options', 'answer', 'solution', 'level', 'subject')}})
        levels = dict(collections.Counter(str(r['level']) for r in inputs))
        if levels != manifest['level_counts']:
            raise ValueError('Difficulty counts differ from source manifest')
        out.mkdir(parents=True)
        for i in entries:
            if i.is_dir():
                continue
            dest = out / 'source' / i.filename[len(prefix):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(i))
    write_jsonl(out / 'inputs.jsonl', inputs)
    write_jsonl(out / 'references.jsonl', references)
    write_json(out / 'question_ids.json', ids)
    result = {'questions': len(ids), 'unique_ids': len(set(ids)), 'readable_original_images': len(sizes),
              'bundle_sha256': BUNDLE_SHA256, 'seeded_selection_reproduced': True,
              'official_metadata_matches': official_checked, 'level_counts': levels,
              'largest_image_side': max(max(s) for s in sizes), 'image_bytes_preserved': True,
              'inputs_sha256': digest(out / 'inputs.jsonl'), 'references_sha256': digest(out / 'references.jsonl'),
              'qwen_raw_responses_present': False, 'real_ovr_generations': 0,
              'scoring_alignment': 'Qwen reference uses a custom scorer. Obtain its raw outputs and rescore all models consistently before comparing accuracies.'}
    write_json(out / 'validation.json', result)
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser();p.add_argument('archive');args = p.parse_args()
    print(json.dumps(import_bundle(args.archive), indent=2))
