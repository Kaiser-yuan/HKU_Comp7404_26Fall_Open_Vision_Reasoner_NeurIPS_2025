"""Preparation archive is allowed; final submission archive fails closed."""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ovr.common import MODELS, digest, read_json, write_json


def submission_errors(root=ROOT):
    errors = []
    team = read_json(root / "configs/team.json")
    if not team.get("group_id") or not team.get("exhibition_date") or len(team.get("members", [])) != 3 or not all(m.get("name") for m in team.get("members", [])):
        errors.append("Group ID, exhibition date and three author names are required")
    if not team.get("eligibility_verified") or not team.get("selection_registration_verified"):
        errors.append("Course eligibility and selection registration need verification")
    coverage = read_json(root / "results/reports/full/coverage.json")
    if not coverage.get("complete") or any(coverage.get("counts", {}).get(m) != 3040 for m in MODELS):
        errors.append("Three complete 3040-question evaluations are required for this full-evaluation submission")
    delivery = read_json(root / "configs/delivery.json")
    if delivery.get("status") != "FINAL" or not all(delivery.get(k) for k in ["human_content_review", "human_visual_review", "offline_demo_review", "ai_disclosure_verified", "talk_rehearsal_completed", "paper_result_sources_verified"]):
        errors.append("Final poster, offline demo, source/disclosure review and rehearsal sign-offs are pending")
    for key in ["poster_pptx", "poster_pdf"]:
        p = root / delivery[key]
        if not p.exists() or "DRAFT" in p.name:
            errors.append(f"Final non-draft {key} is missing")
    provenance_path = root / "output/pdf_provenance.json"
    if not provenance_path.exists():
        errors.append("PDF-from-PPTX provenance is missing")
    else:
        pv = read_json(provenance_path)
        for key, file_key in [("pptx_sha256", "poster_pptx"), ("pdf_sha256", "poster_pdf")]:
            p = root / delivery[file_key]
            if not p.exists() or pv.get(key) != digest(p):
                errors.append("PDF provenance does not match current final files")
    annotated = root / "analysis/annotations60.completed.csv"
    if not annotated.exists():
        errors.append("Human annotations have not been completed")
    else:
        rows = list(csv.DictReader(annotated.open(encoding="utf-8")))
        ids = set(read_json(root / "data/splits/analysis60.json"))
        review = set(read_json(root / "data/splits/review20.json"))
        expected = {(qid, m) for qid in ids for m in MODELS}
        if len(rows) != 180 or {(r.get("question_id"), r.get("model")) for r in rows} != expected:
            errors.append("Annotation rows do not match the fixed 60-question three-model sample")
        for row in rows:
            valid = row.get("annotation_status") == "COMPLETE" and bool(row.get("annotator")) and bool(row.get("error_type"))
            for b in ["visual_reflection", "divide_and_conquer", "visual_verification", "goal_driven_visual_tracing"]:
                valid &= row.get(b) in {"0", "1", "uncertain"}
                if row.get(b) in {"1", "uncertain"}:
                    valid &= bool(row.get(b + "_evidence"))
            if row.get("question_id") in review:
                valid &= bool(row.get("reviewer")) and row.get("reviewer") != row.get("annotator") and bool(row.get("adjudication")) and bool(row.get("disagreement"))
                valid &= all(row.get("reviewer_" + b) in {"0", "1", "uncertain"} and row.get("final_" + b) in {"0", "1", "uncertain"} for b in ["visual_reflection", "divide_and_conquer", "visual_verification", "goal_driven_visual_tracing"])
            if not valid:
                errors.append("At least one annotation/reviewer record is incomplete")
                break
    return errors


def package(kind, lite=False):
    if lite and kind != "preparation":
        raise ValueError("Lightweight archives are for preparation only")
    if kind == "submission":
        # Recompute rather than trusting a stale coverage file.
        from ovr.report import build_report
        build_report("full")
        errors = submission_errors()
        if errors:
            raise SystemExit("SUBMISSION BLOCKED:\n- " + "\n- ".join(errors))
    filename = "OVR_TEAM_STARTER_20261021.zip" if lite else f"OVR_{kind.upper()}_20261021.zip"
    target = ROOT / "artifacts" / filename
    target.parent.mkdir(exist_ok=True)
    exclude_dirs = {".venv", ".build", ".git", "__pycache__", "node_modules", "artifacts"}
    files = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(ROOT)
        if any(part in exclude_dirs for part in rel.parts) or rel.parts[:2] == ("data", "cache"):
            continue
        if lite and rel.parts[:2] == ("data", "images"):
            continue
        if p.name == ".DS_Store" or p.suffix in {".pyc", ".token", ".lock"} or p.name.startswith(".env") or p.name.startswith("approval.local"):
            continue
        files.append(p)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, "OVR_Project/" + p.relative_to(ROOT).as_posix())
        note = "PREPARATION ONLY; NO MODEL INFERENCE RUN. NOT A FINAL SUBMISSION.\n" if kind == "preparation" else "FINAL SUBMISSION PACKAGE; upload to Moodle manually.\n"
        if lite:
            note += "LIGHTWEIGHT TEAM STARTER: full-resolution dataset images omitted. Run python -m ovr.cli prepare-data before evaluation. The demo already embeds its six preview images.\n"
        z.writestr("OVR_Project/ARCHIVE_STATUS.txt", note)
    with zipfile.ZipFile(target) as z:
        if z.testzip():
            raise RuntimeError("Archive CRC failed")
    receipt = {"kind": kind, "lightweight": lite, "file": str(target.relative_to(ROOT)), "sha256": digest(target),
               "files": len(files) + 1, "bytes": target.stat().st_size, "crc_verified": True}
    write_json(target.with_suffix(".receipt.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["preparation", "submission"], required=True)
    parser.add_argument("--lite", action="store_true", help="Omit full-resolution dataset images for a small team handoff")
    args = parser.parse_args()
    print(json.dumps(package(args.kind, args.lite), indent=2))
