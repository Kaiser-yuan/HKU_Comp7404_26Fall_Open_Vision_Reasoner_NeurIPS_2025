"""CPU-only allowlisted transfer package; never includes model weights/results/secrets."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ovr.budget_compare import check
from ovr.common import digest, write_json


def build():
    ready = check()
    patterns = [
        'CLOUD_BUDGET20.md', 'configs/cloud_vllm_budget20.json',
        'configs/cloud_vllm_shared1008192.json', 'configs/protocol.json',
        'requirements-local.txt', 'requirements-vllm-audit-lock.txt',
        'ovr/*.py', 'sources/manifest.json', 'vendor/mathvision/*.py',
        'vendor/mathvision/LICENSE', 'vendor/mathvision/README.upstream.md',
        'workstreams/budget20/*', 'workstreams/shared_test100/inputs.jsonl',
        'workstreams/shared_test100/references.jsonl', 'workstreams/shared_test100/question_ids.json',
        'workstreams/shared_test100/source/images/*.jpg',
        'tests/test_budget_compare.py', 'scripts/cloud/build_budget_starter.py']
    files = sorted({p for pat in patterns for p in ROOT.glob(pat)
                    if p.is_file() and '__pycache__' not in p.parts})
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {'git_reference': revision, 'working_tree_code_sha256_is_authoritative': True,
                'purpose': 'Prepared 20-question shared-trajectory budget comparison; not yet run',
                'sha256': hashes, 'offline_validation': ready}
    out = ROOT / 'artifacts/OVR_BUDGET20_STARTER.zip'
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'OVR_CLOUD_STARTER/'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, prefix + p.relative_to(ROOT).as_posix())
        z.writestr(prefix + 'BUDGET20_TRANSFER_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip() or any(hashlib.sha256(z.read(prefix + k)).hexdigest() != v for k, v in hashes.items()):
            raise ValueError('Transfer archive CRC or member SHA mismatch')
    receipt = {'path': str(out), 'bytes': out.stat().st_size, 'sha256': digest(out),
               'verified_members': len(files), 'gpu_runtime_verified': False,
               'real_generations_started': 0, 'git_reference': revision}
    write_json(out.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    build()
