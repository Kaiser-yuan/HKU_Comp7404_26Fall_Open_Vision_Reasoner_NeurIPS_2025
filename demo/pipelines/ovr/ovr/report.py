import itertools
import random
import statistics
from collections import defaultdict
from pathlib import Path
from .common import ROOT, MODELS, digest, read_json, read_jsonl, unique, wilson, write_csv, write_json


def load_scored(stage="full"):
    data = {}
    for model in MODELS:
        path = ROOT / f"results/scored/{stage}/{model}.jsonl"
        if not path.exists():
            data[model] = {}
            continue
        rows = read_jsonl(path)
        if any(r["model"] != model or type(r["correct"]) is not bool for r in rows):
            raise ValueError("Scored file has invalid model/correct fields")
        provenance = read_json(str(path) + ".provenance.json")
        if provenance["scored_sha256"] != digest(path):
            raise ValueError("Scored file changed since official evaluation")
        raw_path = ROOT / f"results/raw/{stage}/{model}.jsonl"
        if not raw_path.exists() or provenance["raw_sha256"] != digest(raw_path):
            raise ValueError("Raw source absent or changed since scoring; score again")
        data[model] = unique(rows)
    return data


def group_summary(model, group, label, rows, expected):
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    low, high = wilson(k, n)
    lengths = [r["output_tokens"] for r in rows]
    return {"model": model, "group": group, "label": label, "n": n, "expected": expected,
            "coverage": n / expected if expected else 0, "complete": n == expected,
            "accuracy_observed": k / n if n else None,
            "accuracy_full": k / n if n and n == expected else None,
            "ci95_low": low, "ci95_high": high,
            "mean_output_tokens": statistics.mean(lengths) if lengths else None,
            "median_output_tokens": statistics.median(lengths) if lengths else None,
            "truncation_rate": sum(r["finish_reason"] == "length" for r in rows) / n if n else None}


def paired_transition(left, right, a, b, expected):
    ids = sorted(a.keys() & b.keys())
    transitions = {"wrong_to_right": 0, "right_to_wrong": 0, "both_right": 0, "both_wrong": 0}
    diffs = []
    for qid in ids:
        x, y = a[qid]["correct"], b[qid]["correct"]
        key = "both_right" if x and y else "both_wrong" if not x and not y else "wrong_to_right" if y else "right_to_wrong"
        transitions[key] += 1
        diffs.append(int(y) - int(x))
    n = len(ids)
    low = high = None
    if n:
        rng = random.Random(42)
        boots = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(1000))
        low, high = boots[24], boots[974]
    return {"from": left, "to": right, "paired_n": n, "expected": expected, "complete": n == expected,
            **transitions, "accuracy_delta": statistics.mean(diffs) if diffs else None,
            "paired_bootstrap_ci95_low": low, "paired_bootstrap_ci95_high": high}


def build_report(stage="full"):
    questions = unique(read_jsonl(ROOT / "data/questions.jsonl"))
    expected_ids = set(read_json(ROOT / f"data/splits/{'pilot' if stage == 'pilot' else 'full'}.json"))
    data = load_scored(stage)
    summary = []
    for model, indexed in data.items():
        if set(indexed) - expected_ids:
            raise ValueError("Unexpected question ID in scored file")
        summary.append(group_summary(model, "overall", "all", list(indexed.values()), len(expected_ids)))
        for group in ["level", "subject"]:
            labels = sorted({str(questions[qid][group]) for qid in expected_ids})
            for label in labels:
                group_ids = [qid for qid in expected_ids if str(questions[qid][group]) == label]
                summary.append(group_summary(model, group, label, [indexed[qid] for qid in group_ids if qid in indexed], len(group_ids)))
    transitions = [paired_transition(a, b, data[a], data[b], len(expected_ids)) for a, b in itertools.combinations(MODELS, 2)]
    complete = all(set(data[m]) == expected_ids for m in MODELS)
    out = ROOT / f"results/reports/{stage}"
    write_csv(out / "summary.csv", summary)
    write_csv(out / "transitions.csv", transitions)
    status = {"stage": stage, "complete": complete,
              "status": "COMPLETE" if complete else "INCOMPLETE - do not claim full reproduction",
              "expected_per_model": len(expected_ids), "counts": {m: len(data[m]) for m in MODELS},
              "missing_ids": {m: sorted(expected_ids - data[m].keys()) for m in MODELS}}
    write_json(out / "coverage.json", status)
    return status


def freeze():
    status = build_report("pilot")
    if not status["complete"]:
        raise ValueError("Need all 30 pilot responses and official scores for all three models")
    cfg = read_json(ROOT / "configs/protocol.json")
    from .common import fingerprint
    # Check the raw run identity as well: old pilot results cannot freeze a changed protocol.
    for model in MODELS:
        raw = read_jsonl(ROOT / f"results/raw/pilot/{model}.jsonl")
        sessions = list((ROOT / "results/raw/pilot").glob(f"{model}.session-*.json"))
        valid = [read_json(p) for p in sessions if read_json(p)["identity"]["protocol"] == cfg]
        if not valid or any(r["run_fingerprint"] not in {s["run_fingerprint"] for s in valid} for r in raw):
            raise ValueError("Pilot protocol does not match candidate protocol")
    value = {"protocol_sha256": fingerprint(cfg), "questions_sha256": digest(ROOT / "data/questions.jsonl"),
             "pilot_coverage_sha256": digest(ROOT / "results/reports/pilot/coverage.json"),
             "note": "Frozen after pilot validation; independent reevaluation, not exact paper protocol"}
    write_json(ROOT / "configs/freeze.json", value)
    return value
