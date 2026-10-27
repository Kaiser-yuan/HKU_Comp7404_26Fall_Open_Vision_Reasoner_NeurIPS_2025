import base64
import json
from pathlib import Path
from .common import ROOT, MODELS, read_json, read_jsonl, unique, write_csv, write_json
from .report import load_scored

BEHAVIORS = ("visual_reflection", "divide_and_conquer", "visual_verification", "goal_driven_visual_tracing")


def annotation_template():
    questions = unique(read_jsonl(ROOT / "data/questions.jsonl"))
    ids = read_json(ROOT / "data/splits/analysis60.json")
    review = set(read_json(ROOT / "data/splits/review20.json"))
    rows = []
    for qid in ids:
        for model in MODELS:
            row = {"question_id": qid, "model": model, "level": questions[qid]["level"],
                   "subject": questions[qid]["subject"], "second_review": qid in review,
                   "annotator": "", "annotation_status": "UNANNOTATED"}
            for b in BEHAVIORS:
                row[b] = ""
                row[b + "_evidence"] = ""
                row["reviewer_" + b] = ""
                row["final_" + b] = ""
            row.update(error_type="", notes="", reviewer="", disagreement="", adjudication="")
            rows.append(row)
    path = ROOT / "analysis/annotations60.template.csv"
    write_csv(path, rows)
    reviewer_rows = []
    for row in rows:
        if row["question_id"] in review:
            r = {"question_id": row["question_id"], "model": row["model"], "reviewer": ""}
            for b in BEHAVIORS:
                r[b] = ""
                r[b + "_evidence"] = ""
            reviewer_rows.append(r)
    write_csv(ROOT / "analysis/review20.template.csv", reviewer_rows)
    return {"questions": 60, "trajectories": len(rows), "second_review_questions": 20, "path": str(path)}


def agreement(first, second):
    pairs = [(a, b) for a, b in zip(first, second) if a in {"0", "1"} and b in {"0", "1"}]
    n = len(pairs)
    if not n:
        return {"binary_pairs": 0, "raw_agreement": None, "cohen_kappa": None}
    observed = sum(a == b for a, b in pairs) / n
    p1 = sum(a == "1" for a, b in pairs) / n
    p2 = sum(b == "1" for a, b in pairs) / n
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    return {"binary_pairs": n, "raw_agreement": observed,
            "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def analyze():
    import csv
    path = ROOT / "analysis/annotations60.completed.csv"
    if not path.exists():
        result = {"status": "NOT ANNOTATED", "expected_trajectories": 180}
        write_json(ROOT / "analysis/summary.json", result)
        return result
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ids = set(read_json(ROOT / "data/splits/analysis60.json"))
    review = set(read_json(ROOT / "data/splits/review20.json"))
    if len(rows) != 180 or {(r["question_id"], r["model"]) for r in rows} != {(q, m) for q in ids for m in MODELS}:
        raise ValueError("Annotations do not match fixed sample")
    if any(r["annotation_status"] != "COMPLETE" for r in rows):
        raise ValueError("Human annotations are not complete")
    metrics = []
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model]
        for b in BEHAVIORS:
            values = [r["final_" + b] if r["question_id"] in review else r[b] for r in subset]
            if any(v not in {"0", "1", "uncertain"} for v in values):
                raise ValueError("Missing or invalid finalized behavior labels")
            present, absent, uncertain = (values.count(v) for v in ["1", "0", "uncertain"])
            metrics.append({"model": model, "behavior": b, "n": len(values), "present": present,
                            "absent": absent, "uncertain": uncertain,
                            "presence_rate_all": present / len(values),
                            "presence_rate_known": present / (present + absent) if present + absent else None})
    review_rows = [r for r in rows if r["question_id"] in review]
    for r in review_rows:
        if not r["reviewer"] or r["reviewer"] == r["annotator"]:
            raise ValueError("Independent second reviewer is missing")
        if any(r["reviewer_" + b] not in {"0", "1", "uncertain"} for b in BEHAVIORS):
            raise ValueError("Independent reviewer labels are missing")
    agreements = [{"behavior": b, "reviewed_trajectories": len(review_rows),
                   "uncertain_pairs": sum(r[b] == "uncertain" or r["reviewer_" + b] == "uncertain" for r in review_rows),
                   **agreement([r[b] for r in review_rows], [r["reviewer_" + b] for r in review_rows])} for b in BEHAVIORS]
    write_csv(ROOT / "analysis/behavior_rates.csv", metrics)
    write_json(ROOT / "analysis/agreement.json", agreements)
    from collections import Counter
    errors = {m: dict(Counter(r["error_type"] for r in rows if r["model"] == m)) for m in MODELS}
    result = {"status": "HUMAN LABELS ANALYZED", "n": len(rows), "error_types": errors,
              "interpretation": "Descriptive trace analysis; not a causal or exact BTR reproduction"}
    write_json(ROOT / "analysis/summary.json", result)
    return result


def choose_cases(data, order):
    selected = []
    for name, pred in [("RL improvement", lambda a, b: not a and b),
                       ("RL regression", lambda a, b: a and not b),
                       ("Shared failure", lambda a, b: not a and not b)]:
        found = []
        for qid in order:
            if not all(qid in data[m] for m in MODELS):
                continue
            a, b = data["coldstart"][qid]["correct"], data["rl"][qid]["correct"]
            if pred(a, b) and (name != "Shared failure" or not data["base"][qid]["correct"]):
                found.append({"question_id": qid, "category": name})
                if len(found) == 2:
                    break
        selected.extend(found)
    return selected


def build_demo():
    import re
    questions = unique(read_jsonl(ROOT / "data/questions.jsonl"))
    data = load_scored()
    order = read_json(ROOT / "data/splits/full.json")
    selected = choose_cases(data, order)
    actual = any(data.values())
    if not selected:
        selected = [{"question_id": qid, "category": "Benchmark preview - no inference"} for qid in read_json(ROOT / "data/splits/analysis60.json")[:6]]
    cases = []
    for item in selected:
        q = questions[item["question_id"]]
        encoded = base64.b64encode((ROOT / q["image"]).read_bytes()).decode()
        display_options = q["options"]
        if display_options == [chr(65+i) for i in range(len(display_options))]:
            display_options = []  # Letter-only choices are already labeled in the original image.
        cases.append({**item, "question": q["question_id"] + ": " + re.sub(r"<image\d+>", "", q["question"]).strip(),
                      "options": display_options, "answer": q["answer"], "level": q["level"], "subject": q["subject"],
                      "image": "data:image/png;base64," + encoded,
                      "models": {m: data[m].get(q["question_id"]) for m in MODELS}})
    payload = {"status": "REAL SAVED OUTPUTS - check coverage report" if actual and choose_cases(data, order) else "BENCHMARK PREVIEW - MODEL INFERENCE NOT RUN FOR THESE CASES",
               "cases": cases, "models": list(MODELS)}
    template = (ROOT / "demo/template.html").read_text()
    json_text = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.replace("/*PAYLOAD*/null", json_text)
    (ROOT / "demo/index.html").write_text(html)
    write_json(ROOT / "analysis/cases.json", selected)
    return {"cases": len(cases), "status": payload["status"]}
