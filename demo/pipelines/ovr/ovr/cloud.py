"""Two-checkpoint shared-pilot CLI. Does not provision or shut down cloud instances."""
import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
import re
from collections import Counter
import statistics
import time
import zipfile
from contextlib import contextmanager, ExitStack
from pathlib import Path
from .common import ROOT, digest, fingerprint, read_json, read_jsonl, unique, wilson, write_csv, write_json

CONFIG = Path(os.environ.get("OVR_CLOUD_CONFIG", str(ROOT / "configs/cloud_pilot20.json")))


def namespace(config):
    name = config.get("results_namespace", "cloud_pilot20")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid results namespace")
    return name


def inputs(config, root=ROOT):
    shared = root / config["shared_dir"]
    for filename, key in [("inputs.jsonl", "inputs_sha256"), ("references.jsonl", "references_sha256")]:
        if digest(shared / filename) != config[key]:
            raise ValueError("Shared input/reference hash changed: " + filename)
    rows, refs = read_jsonl(shared / "inputs.jsonl"), read_jsonl(shared / "references.jsonl")
    ids = read_json(shared / "question_ids.json")
    if len(rows) != config["expected_pilot_questions"] or list(unique(rows)) != ids or set(unique(refs)) != set(ids):
        raise ValueError("Pilot IDs differ, are duplicated or do not match references")
    expected_levels = config.get("level_counts", {str(n): 4 for n in range(1, 6)})
    if dict(Counter(str(r["level"]) for r in rows)) != expected_levels:
        raise ValueError("Difficulty distribution differs from the frozen question manifest")
    from PIL import Image
    for r in rows:
        if any(k in r for k in ("answer", "solution", "correct", "extracted_answer")):
            raise ValueError("Reference data must never be included in model inputs")
        image = (shared / r["image"]).resolve()
        if not image.is_relative_to(shared.resolve()) or digest(image) != r["image_sha256"]:
            raise ValueError("Invalid image path or hash mismatch")
        with Image.open(image) as im:
            if config.get("max_image_side", 1024) is not None and max(im.size) > config.get("max_image_side", 1024):
                raise ValueError("Shared image exceeds the agreed maximum size")
            im.verify()
    return rows, shared


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process holds this model/phase lock") from exc
        try:
            yield
        finally:
            # Kernel releases this lock even after a killed process. Keep the
            # inode in place so concurrent open() calls cannot create two locks.
            fcntl.flock(f, fcntl.LOCK_UN)


def raw_path(root, phase, model, config=None):
    config = config or {}
    return root / f"results/raw/{namespace(config)}/{phase}/{model}.jsonl"


def execute(config, model, phase, identity, factory, root=ROOT, session_minutes=60):
    """Testable orchestration; only factory() loads a GPU model."""
    rows, shared = inputs(config, root)
    if phase == "canary":
        rows = rows[:config["canary_questions"]]
    ids = [r["question_id"] for r in rows]
    raw = raw_path(root, phase, model, config)
    fp = fingerprint({"identity": identity, "model": model, "phase": phase, "ids": ids, "config": config})
    sessions = root / f"results/runs/{namespace(config)}/{phase}/{model}"
    session_file = sessions / f"session-{time.time_ns()}.json"
    events = sessions / "events.jsonl"
    session = {"model": model, "phase": phase, "run_fingerprint": fp, "identity": identity,
               "protocol": config, "selected_ids": ids, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "expected": len(ids), "state": "starting", "runtime_validated_on_gpu": False}
    started = time.monotonic()
    if session_minutes <= 0:
        raise ValueError("Session minutes must be positive")
    with run_lock(raw.with_suffix(".lock")):
        existing = unique(read_jsonl(raw))
        if set(existing) - set(ids) or any(r.get("model") != model or r.get("run_fingerprint") != fp for r in existing.values()):
            raise ValueError("Resume refused: model, protocol, input, code or runtime differs")
        if all(qid in existing for qid in ids):
            return {"state": "already_complete", "completed": len(existing), "run_fingerprint": fp}
        write_json(session_file, session)
        qid = None
        try:
            backend = factory()
            session.update(backend=backend.metadata, runtime_validated_on_gpu=True)
            write_json(session_file, session)
            for item in rows:
                qid = item["question_id"]
                if qid in existing:
                    continue
                remaining = session_minutes * 60 - (time.monotonic() - started)
                # Do not change per-question deadline just because a session is
                # ending: resume pending questions later with the same protocol.
                if remaining < config["per_question_max_s"]:
                    session["state"] = "session_time_stop"
                    break
                result = backend.infer(item["prompt"], shared / item["image"], config["per_question_max_s"])
                row = {**result, "model": model, "question_id": qid, "run_fingerprint": fp,
                       "protocol_id": config["protocol_id"], "extracted_answer": None, "correct": None,
                       "image_sha256": item["image_sha256"], "status": "generated"}
                append(raw, row)
                existing[qid] = row
                session.update(state="running", completed=len(existing))
                write_json(session_file, session)
                print(f"{phase}/{model}: {len(existing)}/{len(ids)}, tokens={row['output_tokens']}, finish={row['finish_reason']}", flush=True)
            else:
                session["state"] = "complete"
            return session
        except BaseException as exc:
            session.update(state="error", error_type=type(exc).__name__, failed_question_id=qid)
            append(events, {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "question_id": qid,
                            "error_type": type(exc).__name__, "run_fingerprint": fp,
                            "note": "No successful row written for this attempt; resume pending IDs. Empty generated responses remain valid saved attempts."})
            raise
        finally:
            session["completed"] = len(existing)
            session["elapsed_s"] = time.monotonic() - started
            session["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_json(session_file, session)


def canary_ok(config, model, root=ROOT, identity=None):
    rows, _ = inputs(config, root)
    expected = [r["question_id"] for r in rows[:config["canary_questions"]]]
    raw = raw_path(root, "canary", model, config)
    actual = read_jsonl(raw)
    sessions = sorted((root / f"results/runs/{namespace(config)}/canary/{model}").glob("session-*.json"))
    valid = (len(actual) == len(expected) and list(unique(actual)) == expected and bool(sessions)
             and read_json(sessions[-1])["protocol"] == config)
    if identity is not None:
        fp = fingerprint({"identity": identity, "model": model, "phase": "canary", "ids": expected, "config": config})
        valid = valid and all(r.get("run_fingerprint") == fp for r in actual)
    if not valid or any(not r.get("response", "").strip() or not r.get("image_tensor_checked")
                        or r.get("finish_reason") != "stop" for r in actual):
        raise ValueError("Canary incomplete, empty, truncated or image tensor unverified; inspect before pilot")
    return True


def run(config, model, phase, minutes):
    from .cloud_backend import checkpoint, gpu_check, TransformersBackend
    rows, _ = inputs(config)
    hardware = gpu_check(config)
    load_dir, manifest = checkpoint(config, model)
    identity = {"hardware": hardware, "checkpoint": manifest,
                "code": {p.name: digest(p) for p in (Path(__file__), ROOT / "ovr/cloud_backend.py")}}
    if phase == "pilot":
        canary_ok(config, model, identity=identity)
    return execute(config, model, phase, identity, lambda: TransformersBackend(config, load_dir), session_minutes=minutes)


def _report(config, phase="pilot", root=ROOT):
    from .scoring import score
    rows, shared = inputs(config, root)
    if phase == "canary":
        rows = rows[:config["canary_questions"]]
    by_id = unique(rows)
    out = root / f"results/reports/{namespace(config)}/{phase}"
    out.mkdir(parents=True, exist_ok=True)
    coverage, summary, all_scored = {}, [], {}
    for model in config["models"]:
        raw = raw_path(root, phase, model, config)
        generated = read_jsonl(raw)
        indexed = unique(generated)
        if set(indexed) - set(by_id):
            raise ValueError("Unexpected question in raw file")
        if any(r.get("model") != model or r.get("protocol_id") != config["protocol_id"] for r in generated):
            raise ValueError("Raw model or protocol differs from this report")
        sessions = sorted((root / f"results/runs/{namespace(config)}/{phase}/{model}").glob("session-*.json"))
        if generated:
            if not sessions:
                raise ValueError("Raw outputs have no session provenance")
            session = read_json(sessions[-1])
            expected_fp = fingerprint({"identity": session["identity"], "model": model, "phase": phase,
                                       "ids": list(by_id), "config": config})
            if session["protocol"] != config or any(r.get("run_fingerprint") != expected_fp for r in generated):
                raise ValueError("Raw outputs disagree with their recorded protocol/session")
        coverage[model] = {"expected": len(rows), "generated": len(indexed),
                           "missing_ids": [qid for qid in by_id if qid not in indexed],
                           "events": read_jsonl(root / f"results/runs/{namespace(config)}/{phase}/{model}/events.jsonl")}
        scored_path = root / f"results/scored/{namespace(config)}/{phase}/{model}.jsonl"
        if generated:
            score(raw, shared / "references.jsonl", scored_path)
            scored = read_jsonl(scored_path)
        else:
            if scored_path.exists():
                raise ValueError("Stale scored file exists without corresponding raw outputs")
            scored = []
        all_scored[model] = unique(scored)
        groups = [("overall", "all", scored)]
        groups += [("difficulty", str(level), [r for r in scored if int(by_id[r["question_id"]]["level"]) == level]) for level in range(1, 6)]
        groups += [("subject", subject, [r for r in scored if by_id[r["question_id"]]["subject"] == subject]) for subject in sorted({r["subject"] for r in rows})]
        for kind, label, selected in groups:
            n, k = len(selected), sum(bool(r["correct"]) for r in selected)
            ci_low, ci_high = wilson(k, n)
            summary.append({"model": model, "group": kind, "value": label, "n": n, "correct": k,
                            "model_coverage_complete": not coverage[model]["missing_ids"],
                            "accuracy_observed": k / n if n else None, "ci95_low": ci_low, "ci95_high": ci_high,
                            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in selected) if n else None,
                            "mean_latency_s": statistics.mean(r["latency_s"] for r in selected) if n else None,
                            "length_or_time_limit_count": sum(r["finish_reason"] in {"length", "time_limit"} for r in selected),
                            "truncation_rate_observed": sum(r["finish_reason"] in {"length", "time_limit"} for r in selected) / n if n else None,
                            "peak_gpu_allocated_bytes": max((r.get("peak_gpu_allocated_bytes", 0) for r in selected), default=None),
                            "empty_count": sum(not r["response"].strip() for r in selected)})
    paired = []
    for qid in by_id:
        if all(qid in all_scored[m] for m in ("coldstart", "rl")):
            a, b = all_scored["coldstart"][qid], all_scored["rl"][qid]
            paired.append({"question_id": qid, "coldstart_correct": a["correct"], "rl_correct": b["correct"],
                           "transition": f"{int(a['correct'])}->{int(b['correct'])}",
                           "coldstart_tokens": a["output_tokens"], "rl_tokens": b["output_tokens"]})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired.csv", paired, ["question_id", "coldstart_correct", "rl_correct", "transition", "coldstart_tokens", "rl_tokens"])
    complete = all(not r["missing_ids"] for r in coverage.values())
    result = {"protocol_id": config["protocol_id"], "phase": phase, "complete": complete,
              "scope": f"{config['expected_pilot_questions']}-question subset, not a full benchmark reproduction", "models": coverage,
              "paired_n": len(paired), "transitions": {key: sum(r["transition"] == key for r in paired) for key in ["0->0", "0->1", "1->0", "1->1"]}}
    write_json(out / "coverage.json", result)
    write_json(out / "protocol.json", config)
    return result


def report(config, phase="pilot", root=ROOT):
    with ExitStack() as stack:
        for model in config["models"]:
            stack.enter_context(run_lock(raw_path(root, phase, model, config).with_suffix(".lock")))
        return _report(config, phase, root)


def _export(config, root=ROOT):
    _report(config, root=root)
    paths = [root / config["shared_dir"], root / config.get("config_source", "configs/cloud_pilot20.json")]
    paths += [root / f"results/{kind}/{namespace(config)}" for kind in ["raw", "scored", "runs", "reports"]]
    out = root / f"artifacts/OVR_{namespace(config).upper()}_RESULTS.zip"
    out.parent.mkdir(exist_ok=True)
    temporary = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        manifest = {}
        for path in paths:
            for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if file.is_file() and file.suffix not in {".lock", ".tmp"}:
                    key = file.relative_to(root).as_posix()
                    z.write(file, key)
                    manifest[key] = digest(file)
        z.writestr("SHA256SUMS.json", json.dumps(manifest, indent=2))
        z.writestr("README.txt", "Engineering pilot. Read coverage.json before interpreting any accuracy.\nCanary and pilot runs are separate. Contains no model weights or authentication files.\n")
    with zipfile.ZipFile(temporary) as z:
        if z.testzip():
            raise ValueError("Export CRC failed")
    temporary.replace(out)
    return {"file": str(out), "sha256": digest(out), "files": len(manifest)}


def export(config, root=ROOT):
    with ExitStack() as stack:
        for phase in ("canary", "pilot"):
            for model in config["models"]:
                stack.enter_context(run_lock(raw_path(root, phase, model, config).with_suffix(".lock")))
        return _export(config, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--access", action="store_true", help="HEAD config and all weights for both models; no weight downloads")
    p.add_argument("--gpu", action="store_true")
    sub.add_parser("download")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=["coldstart", "rl"])
    p.add_argument("--phase", required=True, choices=["canary", "pilot"])
    p.add_argument("--session-minutes", type=float, default=60)
    p = sub.add_parser("report")
    p.add_argument("--phase", choices=["canary", "pilot"], default="pilot")
    sub.add_parser("export")
    args = parser.parse_args()
    config = read_json(CONFIG)
    if args.command == "check":
        rows, _ = inputs(config)
        result = {"inputs_valid": True, "questions": len(rows), "inference_status": "NOT RUN BY CHECK"}
        from .cloud_backend import access, gpu_check
        if args.access:
            result["access"] = access(config)
        if args.gpu:
            result["gpu"] = gpu_check(config)
        write_json(ROOT / f"results/runs/{namespace(config)}/preflight.json", result)
        print(json.dumps(result, indent=2))
        return 2 if args.access and not result["access"]["complete"] else 0
    if args.command == "download":
        inputs(config)
        from .cloud_backend import access, download
        checked = access(config)
        write_json(ROOT / f"results/runs/{namespace(config)}/access.json", checked)
        download(config, checked)
        result = {"state": "downloaded_and_verified", "inference_status": "NOT RUN BY DOWNLOAD"}
    elif args.command == "run":
        result = run(config, args.model, args.phase, args.session_minutes)
        if result["state"] not in {"complete", "already_complete"}:
            print(json.dumps(result, indent=2))
            return 3
        if args.phase == "canary":
            canary_ok(config, args.model)
    elif args.command == "report":
        result = report(config, args.phase)
    elif args.command == "export":
        result = export(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
