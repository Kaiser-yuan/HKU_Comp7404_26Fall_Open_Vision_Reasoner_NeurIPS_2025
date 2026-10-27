"""Execute the unmodified upstream evaluator in a disposable working directory."""
import subprocess
import sys
import tempfile
from pathlib import Path
from .common import ROOT, MODELS, digest, read_json, read_jsonl, unique, write_json, write_jsonl


def verify_upstream():
    manifest = read_json(ROOT / "sources/manifest.json")
    for row in manifest["files"]:
        if row["path"].startswith("vendor/mathvision/") and digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Upstream source changed: {row['path']}")


def score(raw_path, questions_path, scored_path):
    raw_path, scored_path = Path(raw_path), Path(scored_path)
    if raw_path.resolve() == scored_path.resolve():
        raise ValueError("Scored file must not overwrite raw output")
    verify_upstream()
    raw_hash = digest(raw_path)
    rows = read_jsonl(raw_path)
    if not rows:
        raise ValueError("No generated responses to score")
    unique(rows)
    questions = unique(read_jsonl(questions_path))
    models = {r["model"] for r in rows}
    if len(models) != 1 or not models <= set(MODELS):
        raise ValueError("Each raw file must contain exactly one known model")
    if set(str(r["question_id"]) for r in rows) - questions.keys():
        raise ValueError("Raw output contains unknown question IDs")
    with tempfile.TemporaryDirectory(prefix="ovr-scoring-") as tmp:
        work = Path(tmp)
        write_jsonl(work / "data/test.jsonl", [dict(q, id=q["question_id"]) for q in questions.values()])
        target = work / "outputs/predictions.jsonl"
        write_jsonl(target, [{"id": str(r["question_id"]), "response": r["response"]} for r in rows])
        process = subprocess.run([sys.executable, str(ROOT / "vendor/mathvision/evaluate.py")],
                                 cwd=work, capture_output=True, text=True, timeout=900)
        if process.returncode:
            raise RuntimeError("Official scoring failed: " + process.stderr[-3000:])
        official = unique(read_jsonl(target), "id")
    scored = [dict(r, extracted_answer=official[str(r["question_id"])]["model_answer"],
                   correct=official[str(r["question_id"])]["correct"]) for r in rows]
    if digest(raw_path) != raw_hash:
        raise ValueError("Raw output changed during scoring")
    write_jsonl(scored_path, scored)
    write_json(str(scored_path) + ".provenance.json", {
        "raw_sha256": raw_hash, "scored_sha256": digest(scored_path),
        "evaluator_commit": read_json(ROOT / "configs/protocol.json")["upstream_mathvision_commit"],
        "count": len(scored), "correct": sum(r["correct"] for r in scored)})
    return {"count": len(scored), "correct": sum(r["correct"] for r in scored)}
