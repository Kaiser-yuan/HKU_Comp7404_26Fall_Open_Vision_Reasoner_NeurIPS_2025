"""Offline host and shared-pilot checks; no installs, credentials or model loads."""
import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": type(exc).__name__}


def inspect():
    shared = ROOT / "workstreams/shared_pilot20"
    rows = [json.loads(s) for s in (shared / "inputs.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = json.loads((shared / "question_ids.json").read_text(encoding="utf-8"))
    errors = []
    if len(rows) != 20 or len(set(ids)) != 20 or [r["question_id"] for r in rows] != ids:
        errors.append("Shared input IDs differ from the fixed 20-question list")
    for r in rows:
        p = (shared / r["image"]).resolve()
        if not p.is_relative_to(shared.resolve()) or not p.is_file():
            errors.append("Missing/invalid image: " + r["question_id"])
        elif hashlib.sha256(p.read_bytes()).hexdigest() != r["image_sha256"]:
            errors.append("Image hash mismatch: " + r["question_id"])
        if "answer" in r or "correct" in r:
            errors.append("Reference answer present in model inputs")
    if any(sum(int(r["level"]) == n for r in rows) != 4 for n in range(1, 6)):
        errors.append("Shared pilot must have four questions at every difficulty")
    # No torch import: the first check must work before any GPU environment exists.
    gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpu_rows = []
    if gpu.get("returncode") == 0:
        for row in csv.reader(io.StringIO(gpu["stdout"]), skipinitialspace=True):
            if len(row) == 3:
                gpu_rows.append({"name": row[0], "memory_mib": float(row[1]), "driver": row[2]})
    memory = None
    if sys.platform == "win32":
        ram = command(["powershell.exe", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)"])
        if ram.get("returncode") == 0:
            memory = ram["stdout"]
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "python_311": sys.version_info[:2] == (3, 11), "ram_gib": memory,
            "project_disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
            "gpus": gpu_rows, "nvidia_smi": gpu, "shared_pilot": {"questions": len(rows), "errors": errors, "valid": not errors},
            "inference_status": "NOT RUN", "model_downloads": "NOT ATTEMPTED",
            "next_step": "Verify CUDA PyTorch and checkpoint access, then validate a quantized image canary. Do not run the legacy BF16/vLLM command on native Windows."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/local/windows_host.json")
    args = parser.parse_args()
    report = inspect()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["shared_pilot"]["valid"] else 1)
