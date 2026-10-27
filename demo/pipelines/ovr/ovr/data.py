"""Download pinned public benchmark only; never download model weights."""
import hashlib
import io
import urllib.request
from collections import Counter
from pathlib import Path
from .common import ROOT, digest, read_json, read_jsonl, stratified_ids, unique, write_json, write_jsonl


def prepare(config=ROOT / "configs/protocol.json"):
    import pyarrow.parquet as pq
    from PIL import Image
    cfg = read_json(config)
    ds = cfg["dataset"]
    cache = ROOT / "data/cache/mathvision-test.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or digest(cache) != ds["sha256"]:
        url = f"https://huggingface.co/datasets/{ds['repo']}/resolve/{ds['revision']}/{ds['file']}"
        temp = cache.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if digest(temp) != ds["sha256"]:
            raise ValueError("Dataset SHA256 mismatch; refusing to prepare")
        temp.replace(cache)
    source = pq.read_table(cache).to_pylist()
    rows = []
    (ROOT / "data/images").mkdir(parents=True, exist_ok=True)
    for row in source:
        qid = str(row["id"])
        encoded = row.get("decoded_image") or row.get("image")
        if not isinstance(encoded, dict) or not encoded.get("bytes"):
            raise ValueError(f"Expected embedded image for {qid}; got {type(encoded)}")
        data = encoded["bytes"]
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        # Use a stable safe filename, preserving source image bytes and resolution.
        rel = "data/images/" + hashlib.sha256(qid.encode()).hexdigest()[:24] + ".png"
        # Convert through Pillow so filename/content agree, without resizing.
        with Image.open(io.BytesIO(data)) as im:
            im.convert("RGB").save(ROOT / rel, format="PNG")
        rows.append({"question_id": qid, "question": row["question"],
                     "options": row.get("options") or [], "answer": str(row["answer"]),
                     "subject": row["subject"], "level": int(row["level"]),
                     "image": rel, "image_sha256": digest(ROOT / rel)})
    unique(rows)
    if len(rows) != ds["count"]:
        raise ValueError(f"Expected {ds['count']} questions, got {len(rows)}")
    write_jsonl(ROOT / "data/questions.jsonl", rows)
    write_json(ROOT / "data/splits/full.json", [r["question_id"] for r in rows])
    write_json(ROOT / "data/splits/pilot.json", stratified_ids(rows, 6, cfg["seed"]))
    analysis = stratified_ids(rows, 12, cfg["seed"] + 1)
    write_json(ROOT / "data/splits/analysis60.json", analysis)
    import random
    write_json(ROOT / "data/splits/review20.json", random.Random(cfg["seed"] + 2).sample(analysis, 20))
    result = {"dataset": ds, "questions_sha256": digest(ROOT / "data/questions.jsonl"),
              "count": len(rows), "levels": dict(Counter(r["level"] for r in rows)),
              "subjects": dict(Counter(r["subject"] for r in rows))}
    write_json(ROOT / "data/manifest.json", result)
    return validate()


def validate():
    from PIL import Image
    cfg = read_json(ROOT / "configs/protocol.json")
    rows = read_jsonl(ROOT / "data/questions.jsonl")
    by_id = unique(rows)
    if len(rows) != cfg["dataset"]["count"]:
        raise ValueError("Wrong dataset count")
    for row in rows:
        path = ROOT / row["image"]
        if digest(path) != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {row['question_id']}")
        with Image.open(path) as im:
            im.verify()
    for split, count in [("full", len(rows)), ("pilot", 30), ("analysis60", 60), ("review20", 20)]:
        ids = read_json(ROOT / f"data/splits/{split}.json")
        if len(ids) != count or len(set(ids)) != count or set(ids) - by_id.keys():
            raise ValueError(f"Invalid {split}")
    if not set(read_json(ROOT / "data/splits/review20.json")) <= set(read_json(ROOT / "data/splits/analysis60.json")):
        raise ValueError("Review sample is not a subset of analysis sample")
    if digest(ROOT / "data/questions.jsonl") != read_json(ROOT / "data/manifest.json")["questions_sha256"]:
        raise ValueError("Question file changed since preparation")
    return {"status": "PASS", "questions": len(rows), "images_verified": len(rows), "model_inference": "NOT RUN"}
