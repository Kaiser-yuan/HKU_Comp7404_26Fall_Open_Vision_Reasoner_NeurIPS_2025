"""Small, explicit allowlist for transfer from the Mac to a rented Linux host."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ovr.common import digest, write_json


def build():
    patterns = ["requirements-cloud.txt", "requirements-local.txt", "requirements-vllm-audit-lock.txt", "CLOUD_START_HERE.md",
                "configs/cloud*.json", "configs/protocol.json", "sources/manifest.json",
                "ovr/*.py", "vendor/mathvision/*.py", "vendor/mathvision/LICENSE", "vendor/mathvision/README.upstream.md",
                "scripts/cloud/*.sh", "scripts/cloud/contract_smoke.py", "tests/test_cloud*.py",
                "workstreams/shared_test100/**/*", "workstreams/shared_pilot20/**/*", "workstreams/shared_pilot20/*"]
    files = sorted({p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = ROOT / "artifacts/OVR_CLOUD_STARTER.zip"
    out.parent.mkdir(exist_ok=True)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, "OVR_CLOUD_STARTER/" + p.relative_to(ROOT).as_posix())
        z.writestr("OVR_CLOUD_STARTER/TRANSFER_MANIFEST.json", json.dumps({"git_reference": revision, "sha256": hashes}, indent=2))
        z.writestr("OVR_CLOUD_STARTER/README.txt", "Start with CLOUD_START_HERE.md. No model weights or real inference results included.\n")
    with zipfile.ZipFile(out) as z:
        if z.testzip():
            raise ValueError("Starter ZIP CRC failed")
    receipt = {"file": str(out), "bytes": out.stat().st_size, "sha256": digest(out), "git_reference": revision,
               "source_files": len(files), "crc_verified": True, "gpu_runtime_verified": False}
    write_json(out.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
