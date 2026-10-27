"""Paid GPU runner. No model load before an explicit valid budget approval."""
import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from .common import ROOT, digest, fingerprint, read_json, read_jsonl, unique, write_json


def approval_check(approval, cfg, stage, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if not approval.get("approved") or approval.get("stage") != stage:
        raise ValueError("Explicit approval for this stage is required")
    if approval.get("protocol_sha256") != fingerprint(cfg):
        raise ValueError("Approval belongs to a different protocol")
    expiry = dt.datetime.fromisoformat(approval["expires_at"])
    if expiry.tzinfo is None or expiry <= now:
        raise ValueError("Approval expired or timezone absent")
    for key in ["hourly_rate_cny", "total_budget_cny", "session_max_minutes"]:
        if approval.get(key, 0) <= 0:
            raise ValueError(f"Positive {key} is required")
    available = approval["total_budget_cny"] - approval.get("spent_before_session_cny", 0) - approval.get("storage_and_setup_reserve_cny", 0)
    if available <= 0:
        raise ValueError("No remaining approved budget")
    seconds = min(approval["session_max_minutes"] * 60, available / approval["hourly_rate_cny"] * 3600,
                  (expiry - now).total_seconds())
    return seconds


def validate_existing(raw_path, expected_ids, model, run_fp):
    rows = read_jsonl(raw_path)
    indexed = unique(rows)
    if set(indexed) - set(expected_ids):
        raise ValueError("Existing run contains IDs outside the selected split")
    if any(r["model"] != model or r.get("run_fingerprint") != run_fp for r in rows):
        raise ValueError("Resume rejected: model/protocol/data/runtime mismatch")
    return indexed


def question_prompt(question, cfg):
    options = question.get("options") or []
    body = re.sub(r"<image\d+>", "", question["question"]).strip()
    if options:
        body += "\nOptions:\n" + "\n".join(f"({chr(65+i)}) {v}" for i, v in enumerate(options))
    return body + "\n\n" + cfg["prompt_suffix"]


def run(model, stage, approval_path):
    cfg = read_json(ROOT / "configs/protocol.json")
    allowance = approval_check(read_json(approval_path), cfg, stage)
    if stage == "full":
        frozen = read_json(ROOT / "configs/freeze.json")
        if frozen["protocol_sha256"] != fingerprint(cfg) or frozen["questions_sha256"] != digest(ROOT / "data/questions.jsonl"):
            raise ValueError("Protocol/data must be frozen after all three pilots")
    # Validate benchmark before importing/loading the expensive GPU runtime.
    from .data import validate
    validate()
    spec = cfg["models"][model]
    ids = read_json(ROOT / f"data/splits/{'pilot' if stage == 'pilot' else 'full'}.json")
    questions = unique(read_jsonl(ROOT / "data/questions.jsonl"))
    runtime = {k: importlib.metadata.version(k) for k in ["vllm", "transformers", "torch"]}
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    code_hashes = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = {"protocol": cfg, "code_sha256": fingerprint(code_hashes), "questions_sha256": digest(ROOT / "data/questions.jsonl"),
                "selected_ids": ids, "model": model, "runtime": runtime, "gpu": gpu}
    run_fp = fingerprint(identity)
    output = ROOT / f"results/raw/{stage}/{model}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    # A lock is never automatically removed after a crash. Inspect the process,
    # then remove a stale lock manually; this prevents concurrent append corruption.
    lock = output.with_suffix(".lock")
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    started = time.monotonic()
    session = output.parent / f"{model}.session-{time.time_ns()}.json"
    record = {"identity": identity, "run_fingerprint": run_fp, "stage": stage,
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "python": platform.python_version(), "expected": len(ids), "state": "starting"}
    write_json(session, record)
    try:
        existing = validate_existing(output, ids, model, run_fp)
        pending = [qid for qid in ids if qid not in existing]
        if not pending:
            record["state"] = "already_complete"
            return record
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        from PIL import Image
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        processor = AutoProcessor.from_pretrained(spec["repo"], revision=spec["revision"], **cfg["image"])
        engine = LLM(model=spec["repo"], revision=spec["revision"], tokenizer_revision=spec["revision"],
                     seed=cfg["seed"], mm_processor_kwargs=cfg["image"], **cfg["engine"])
        params = SamplingParams(**cfg["generation"])
        worst_batch_s = 0
        completed = 0
        for start in range(0, len(pending), cfg["batch_size"]):
            # Reserve at least a worst-case token-generation window before a batch.
            # This is a stop-request bound, not a provider billing/shutdown guarantee.
            reserve = max(worst_batch_s * 1.5, 180)
            if time.monotonic() - started + reserve >= allowance:
                record["state"] = "budget_stop"
                break
            batch_ids = pending[start:start + cfg["batch_size"]]
            inputs = []
            for qid in batch_ids:
                q = questions[qid]
                messages = [{"role": "user", "content": [{"type": "image", "image": str(ROOT / q["image"])},
                                                              {"type": "text", "text": question_prompt(q, cfg)}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                with Image.open(ROOT / q["image"]) as im:
                    image = im.convert("RGB")
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            tick = time.monotonic()
            outputs = engine.generate(inputs, params, use_tqdm=False)
            elapsed = time.monotonic() - tick
            worst_batch_s = max(worst_batch_s, elapsed)
            if len(outputs) != len(batch_ids):
                raise RuntimeError("Engine returned a different number of outputs")
            with output.open("a", encoding="utf-8") as f:
                for qid, result in zip(batch_ids, outputs):
                    answer = result.outputs[0]
                    row = {"model": model, "question_id": qid, "response": answer.text,
                           "extracted_answer": None, "correct": None,
                           "output_tokens": len(answer.token_ids), "finish_reason": answer.finish_reason,
                           "latency_s": elapsed, "latency_scope": "shared batch wall time",
                           "run_fingerprint": run_fp}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            completed += len(batch_ids)
            record.update(state="running", completed_this_session=completed,
                          total_completed=len(existing) + completed,
                          elapsed_s=time.monotonic() - started)
            write_json(session, record)
            print(f"{model}: {record['total_completed']}/{len(ids)}", flush=True)
        else:
            record["state"] = "complete"
        return record
    except BaseException as e:
        record.update(state="error", error_type=type(e).__name__)
        raise
    finally:
        record["elapsed_s"] = time.monotonic() - started
        record["estimated_session_cny"] = record["elapsed_s"] / 3600 * read_json(approval_path)["hourly_rate_cny"]
        record["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(session, record)
        lock.unlink(missing_ok=True)
