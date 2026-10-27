"""Prepare shared inputs for the revised two-people/four-model pilot.

This script only prepares public data and manifests. It never loads models,
accepts access conditions, rents compute, or calls an inference API.
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ovr.common import digest, fingerprint, read_json, read_jsonl, stratified_ids, unique, write_json, write_jsonl
from ovr.run import question_prompt


def prepare():
    from PIL import Image
    if any(p.stat().st_size for p in (ROOT / "results/raw/cloud_pilot20").rglob("*.jsonl")):
        raise ValueError("Cloud outputs already exist. Reuse the frozen shared inputs instead of regenerating them.")
    cfg = read_json(ROOT / "configs/protocol.json")
    questions = unique(read_jsonl(ROOT / "data/questions.jsonl"))
    ids = stratified_ids(list(questions.values()), per_level=4, seed=42)
    out = ROOT / "workstreams/shared_pilot20"
    (out / "images").mkdir(parents=True, exist_ok=True)
    inputs, references = [], []
    for qid in ids:
        question = questions[qid]
        image_path = out / "images" / (Path(question["image"]).stem + ".png")
        with Image.open(ROOT / question["image"]) as im:
            im = im.convert("RGB")
            # Shared physical inputs, not just a nominal resolution setting.
            # Preserve aspect ratio and do not enlarge small source images.
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            im.save(image_path)
        inputs.append({"question_id": qid, "prompt": question_prompt(question, cfg),
                       "image": image_path.relative_to(out).as_posix(),
                       "image_sha256": digest(image_path), "level": question["level"],
                       "subject": question["subject"]})
        references.append({"question_id": qid, "answer": question["answer"],
                           "options": question["options"], "level": question["level"],
                           "subject": question["subject"]})
    write_jsonl(out / "inputs.jsonl", inputs)
    write_jsonl(out / "references.jsonl", references)
    write_json(out / "question_ids.json", ids)
    proposed_models = {
        "coldstart": {**cfg["models"]["coldstart"], "responsible": "user", "selection": "our recommended pair"},
        "rl": {**cfg["models"]["rl"], "responsible": "user", "selection": "our recommended pair"},
        "base": {**cfg["models"]["base"], "responsible": "teammate", "selection": "suggested; teammate confirmation pending"},
        "rethinker": {"repo": "TIGER-Lab/VL-Rethinker-7B", "revision": "5439d11fc9fa1474ab015104e6604223f165ccf5",
                      "responsible": "teammate", "selection": "suggested; teammate confirmation pending"}
    }
    protocol = {
        "protocol_id": "shared-pilot20-proposal-v1",
        "status": "PROPOSED; both experiment owners must align before inference",
        "purpose": "Engineering feasibility pilot, not full benchmark reproduction",
        "models": proposed_models,
        "questions": 20, "seed": 42, "per_difficulty": 4,
        "dataset_revision": cfg["dataset"]["revision"],
        "inputs_sha256": digest(out / "inputs.jsonl"),
        "reference_sha256": digest(out / "references.jsonl"),
        "generation_proposal": {"temperature": 0, "max_new_tokens": 4096, "batch_size": 1},
        "images": {"longest_side_max_pixels": 1024, "aspect_ratio_preserved": True,
                   "same_prepared_image_bytes_for_every_model": True},
        "quantization_proposal": {"language_weights_bits": 4, "group_size": 64,
                                  "status": "candidate only; conversion/runtime not verified",
                                  "note": "Record actual quantized modules and conversion; do not assume vision encoder is 4-bit"},
        "reference_answer_usage": "Scoring only. Never concatenate references.jsonl into the model prompt.",
        "failures": "Retain OOM, image failures, no-final-answer and truncation; do not silently drop questions",
        "comparison_limits": "4096-token and quantized pilot differs from the earlier BF16/32768-token plan. Do not merge outputs from those protocols."
    }
    write_json(out / "protocol.proposed.json", protocol)
    (out / "README.txt").write_text(
        "Shared 20-question MathVision engineering pilot\n"
        "Status: proposed protocol; no model inference has been run.\n"
        "Both experiment owners should confirm model assignment, quantization, image handling and token limits before running.\n"
        "inputs.jsonl contains model prompts and image paths. references.jsonl is for scoring only; never send it to the model.\n"
        "question_ids.json fixes the same 20 IDs for every model, four per difficulty level.\n"
        "Model/run configuration and pinned source revisions: protocol.proposed.json.\n"
        "Images are derived from MathLLMs/MathVision; resized only to a maximum side of 1024 with aspect ratio preserved.\n"
        "Source: https://huggingface.co/datasets/MathLLMs/MathVision and https://github.com/mathllm/MATH-V\n"
        "Current user pair: configs/cloud_pilot20.json specifies the prepared Linux BF16 cloud runner.\n"
        "The older 4-bit proposal is not used by that runner. Actual OVR CUDA inference remains to be verified.\n",
        encoding="utf-8")
    write_json(ROOT / "workstreams/user_ovr/assignment.json", {
        "owner": "user", "models": {k: proposed_models[k] for k in ["coldstart", "rl"]},
        "shared_inputs": "workstreams/shared_pilot20", "planned_generations": 40,
        "actual_generations": 0, "state": "CLOUD RUNNER PREPARED; OVR CUDA INFERENCE NOT RUN",
        "cloud_protocol": "configs/cloud_pilot20.json",
        "gates": ["HF account access to OVR-RL", "User-provisioned Linux GPU with 48/80GB VRAM",
                  "Both checkpoint image canaries pass", "Align protocol with the other experiment owner"],
        "legacy_note": "configs/protocol.json and ovr.cli run are the historical three-model BF16 plan. Use ovr.cloud and CLOUD_START_HERE.md for the current user pair."})
    zip_path = ROOT / "artifacts/OVR_SHARED_PILOT20.zip"
    zip_path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "validation.json":
                z.write(p, "shared_pilot20/" + p.relative_to(out).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip():
            raise ValueError("Shared pilot archive CRC failed")
    # Mechanical input checks are independent of checkpoint inference.
    assert len(inputs) == len(set(ids)) == 20
    assert all(sum(r["level"] == level for r in inputs) == 4 for level in range(1, 6))
    assert all("answer" not in r and "options" not in r for r in inputs)
    for r in inputs:
        assert digest(out / r["image"]) == r["image_sha256"]
        with Image.open(out / r["image"]) as im:
            assert max(im.size) <= 1024
            im.verify()
    receipt = {"status": "PREPARATION VERIFIED; INFERENCE NOT RUN", "questions": 20,
               "four_models_planned_responses": 80, "user_pair_planned_responses": 40,
               "protocol_sha256": fingerprint(protocol), "archive": str(zip_path.relative_to(ROOT)),
               "archive_sha256": digest(zip_path), "archive_bytes": zip_path.stat().st_size,
               "actual_model_generations": 0}
    write_json(ROOT / "workstreams/shared_pilot20/validation.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
