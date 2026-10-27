"""Verify and re-score the distributed OVR evidence on CPU, without changing it."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMESPACES = ('cloud_vllm_shared1008192_v1', 'cloud_vllm_budget20_prefix_v1',
              'teacher_student_lengths_v1')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify():
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError('Distributed file missing or changed: ' + name)
    print(f'Verified {len(manifest)} distributed files', flush=True)
    return manifest


def copy_into(work, name):
    source, dest = ROOT / name, work / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, dest)
    else:
        shutil.copyfile(source, dest)


def reproduce():
    if os.name == 'nt':
        raise SystemExit('Use WSL2 for this offline entry point; the preserved runner imports POSIX file locks.')
    original_manifest = verify()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work = ROOT / 'results/reproduced' / (stamp + '-' + uuid.uuid4().hex[:8])
    work.mkdir(parents=True, exist_ok=False)
    for name in ['configs', 'workstreams/shared_test100', 'workstreams/budget20',
                 'workstreams/teacher_lengths', '.build/contract-assets/tokenizer.json']:
        copy_into(work, name)
    for ns in NAMESPACES:
        copy_into(work, 'results/raw/' + ns)
    for ns in NAMESPACES[:2]:
        copy_into(work, 'results/runs/' + ns)

    from ovr.cloud import report as report100
    from ovr.budget_compare import report as report20
    from ovr.trace_lengths import analyze
    config = json.loads((work / 'configs/cloud_vllm_shared1008192.json').read_text())
    print('Official re-scoring: two models, 100 questions each', flush=True)
    hundred = report100(config, root=work)
    assert hundred['complete'] and hundred['paired_n'] == 100
    print('Official re-scoring: 20 questions, two prefix budgets per model', flush=True)
    twenty = report20(root=work)
    assert twenty['complete']
    print('Re-tokenizing 1,000 saved teacher examples and 40 student traces', flush=True)
    lengths = analyze(root=work, from_sample=True)
    assert lengths['complete'] and lengths['teacher_observed_n'] == 1000

    compared = []
    reports = {
        NAMESPACES[0] + '/pilot': ['summary.csv', 'paired.csv', 'coverage.json'],
        NAMESPACES[1]: ['summary.csv', 'paired.csv', 'comparison.json'],
        NAMESPACES[2]: ['summary.csv', 'student_lengths.jsonl', 'teacher_lengths.jsonl', 'comparison.json'],
    }
    for namespace, filenames in reports.items():
        for name in filenames:
            relative = Path('results/reports') / namespace / name
            if sha(ROOT / relative) != sha(work / relative):
                raise ValueError('Recomputed report differs: ' + str(relative))
            compared.append(str(relative))
    scored_paths = [f'results/scored/{NAMESPACES[0]}/pilot/{m}.jsonl'
                    for m in ('coldstart', 'rl')]
    scored_paths += [f'results/scored/{NAMESPACES[1]}/{m}/{b}-scored.jsonl'
                     for m in ('coldstart', 'rl') for b in (8192, 16384)]
    for relative in scored_paths:
        if sha(ROOT / relative) != sha(work / relative):
            raise ValueError('Official per-question scoring differs: ' + relative)
        compared.append(relative)
    assert all(sha(ROOT / name) == expected for name, expected in original_manifest.items())
    result = {'status': 'OFFLINE_REPRODUCTION_VERIFIED', 'new_generations': 0,
              'gpu_used': False, 'paid_api_calls': 0, 'original_files_unchanged': True,
              'distributed_files_verified': len(original_manifest),
              'byte_identical_recomputed_files': compared,
              'scored_views': 280, 'teacher_sample_n': 1000, 'student_trace_n': 40}
    (work / 'REPRODUCTION_VERIFICATION.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('OFFLINE_REPRODUCTION_VERIFIED\n' + str(work), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else reproduce()
