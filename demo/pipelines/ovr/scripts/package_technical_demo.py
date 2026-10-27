"""Build the OVR technical handoff from an explicit list of existing evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/OVR_实验操作与复现包_20261027.zip'
NS = ('cloud_vllm_shared1008192_v1', 'cloud_vllm_budget20_prefix_v1',
      'teacher_student_lengths_v1', 'cloud_vllm_preflight8192_probability_review_v1')


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected_files():
    files = set()
    directories = ['ovr', 'scripts', 'tests', 'configs', 'sources', 'vendor/mathvision',
                   'workstreams/shared_test100', 'workstreams/budget20',
                   'workstreams/teacher_lengths', 'results/reports/vllm_deployment_audit']
    directories += [f'results/{kind}/{ns}' for ns in NS
                    for kind in ('raw', 'scored', 'runs', 'reports')]
    for name in directories:
        for p in (ROOT / name).rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            if '__pycache__' in p.parts or p.name.startswith('.') or p.suffix in {'.pyc', '.lock', '.log'}:
                continue
            if p.name.startswith('approval.local'):
                continue
            files.add(p)
    files.update(ROOT.glob('requirements*.txt'))
    files.update(ROOT / name for name in ['CLOUD_START_HERE.md', 'CLOUD_BUDGET20.md',
                 '.build/contract-assets/tokenizer.json', 'demo/template.html',
                 'output/OVR_教师冷启动RL推理长度_完整对比.docx',
                 'output/OVR_教师冷启动RL推理长度_完整对比.md'])
    files.update((ROOT / 'output').glob('OVR_teacher_coldstart_rl_reasoning_lengths.*'))
    return sorted(files)


def build():
    files = selected_files()
    assert all(p.is_file() for p in files)
    secret = re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|'
                        rb'BEGIN (?:OPENSSH|RSA) PRIVATE KEY|X-Amz-Signature=[0-9a-fA-F]{40,}|[?&]Signature=[A-Za-z0-9%+/]{30,}')
    with tempfile.TemporaryDirectory(prefix='ovr-technical-build-') as folder:
        stage = Path(folder) / 'OVR_technical_demo'
        for source in files:
            name = source.relative_to(ROOT)
            assert not any(part in {'models', '.venv', '.git', 'node_modules'} for part in name.parts)
            if source.suffix in {'.py', '.json', '.jsonl', '.md', '.txt', '.csv', '.sh', '.ps1'}:
                assert not secret.search(source.read_bytes()), 'Credential marker: ' + str(name)
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        docs = ROOT / 'docs/technical_demo'
        shutil.copyfile(docs / 'README.md', stage / 'README.md')
        (stage / 'docs').mkdir(exist_ok=True)
        for source in docs.glob('*.md'):
            if source.name != 'README.md':
                shutil.copyfile(source, stage / 'docs' / source.name)
        shutil.copyfile(ROOT / 'artifacts/OVR_SHARED100_CASES.html', stage / 'demo/OVR_SHARED100_CASES.html')
        provenance = stage / 'provenance'
        provenance.mkdir()
        source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
        source_hashes['artifacts/OVR_SHARED100_CASES.html'] = digest(ROOT / 'artifacts/OVR_SHARED100_CASES.html')
        identity = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'new_packaging_files': ['scripts/package_technical_demo.py', 'scripts/reproduce_technical_demo.py',
                                           'docs/technical_demo/'],
                    'source_file_sha256': source_hashes,
                    'scope': 'Existing OVR evidence and a new CPU-only reproduction wrapper; no new model generation.'}
        (provenance / 'source_identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
        (provenance / 'reference_package.json').write_text(json.dumps({
            'reference_name': 'technical_demo 2.zip',
            'use': 'Directory and command-documentation format only; teammate results not attributed to OVR.',
            'reference_files_executed': False}, indent=2) + '\n')
        environments = {}
        for model in ('coldstart', 'rl'):
            path = next((ROOT / f'results/runs/{NS[0]}/pilot/{model}').glob('session-*.json'))
            data = json.loads(path.read_text())
            environments[model] = {'session': str(path.relative_to(ROOT)),
                'started_utc': data['started_utc'], 'ended_utc': data['ended_utc'],
                'state': data['state'], 'hardware_and_packages': data['identity']['hardware'],
                'runtime': data['identity'].get('runtime')}
        (stage / 'logs').mkdir()
        (stage / 'logs/environment_summary.json').write_text(json.dumps(environments, ensure_ascii=False, indent=2) + '\n')
        sums = {p.relative_to(stage).as_posix(): digest(p)
                for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage / 'SHA256SUMS.json').write_text(json.dumps(sums, ensure_ascii=False, indent=2) + '\n')
        OUT.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():
                    z.write(p, p.relative_to(stage.parent).as_posix())
        with zipfile.ZipFile(OUT) as z:
            assert z.testzip() is None
            assert all(hashlib.sha256(z.read('OVR_technical_demo/' + name)).hexdigest() == value
                       for name, value in sums.items())
        receipt = {'file': str(OUT), 'bytes': OUT.stat().st_size, 'sha256': digest(OUT),
                   'members_verified': len(sums), 'source_files_verified': len(source_hashes),
                   'model_weights_included': False, 'credentials_included': False,
                   'new_generations': 0, 'gpu_started': False}
        OUT.with_suffix('.receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    build()
