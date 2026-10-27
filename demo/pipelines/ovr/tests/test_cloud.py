import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ovr.cloud import inputs, execute, canary_ok, report, export, raw_path, run_lock
from ovr.cloud_backend import build_load_view
from ovr.common import ROOT, digest, read_json, read_jsonl, write_json


class FakeBackend:
    metadata = {"test_double": True}

    def __init__(self, answers, fail_after=None, reason="stop", response=None):
        self.answers, self.fail_after, self.calls = answers, fail_after, []
        self.reason, self.response = reason, response

    def infer(self, prompt, image_path, max_time_s):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise ConnectionError("test infrastructure interruption")
        self.calls.append(image_path.name)
        response = self.response if self.response is not None else "\\boxed{" + self.answers[image_path.name] + "}"
        return {"response": response, "output_tokens": 12, "image_tensor_checked": True,
                "finish_reason": self.reason, "latency_s": .1}


class CloudPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = read_json(ROOT / "configs/cloud_pilot20.json")
        shared = self.root / self.config["shared_dir"]
        shutil.copytree(ROOT / self.config["shared_dir"], shared)
        refs = {r["question_id"]: r for r in read_jsonl(shared / "references.jsonl")}
        self.answers = {Path(r["image"]).name: str(refs[r["question_id"]]["answer"]) for r in read_jsonl(shared / "inputs.jsonl")}
        (self.root / "configs").mkdir()
        write_json(self.root / "configs/cloud_pilot20.json", self.config)
        self.identity = {"test_only": "not a real GPU result"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_fake(self, model="coldstart", phase="canary", backend=None, identity=None):
        backend = backend or FakeBackend(self.answers)
        return execute(self.config, model, phase, identity or self.identity, lambda: backend, self.root)

    def test_inputs_validate_images_and_refuse_answer_leakage(self):
        rows, shared = inputs(self.config, self.root)
        self.assertEqual(len(rows), 20)
        path = shared / "inputs.jsonl"
        rows[0]["answer"] = "DO_NOT_SEND"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.config["inputs_sha256"] = digest(path)
        with self.assertRaisesRegex(ValueError, "Reference"):
            inputs(self.config, self.root)

    def test_interruption_resume_and_complete_does_not_reload(self):
        with self.assertRaises(ConnectionError):
            self.run_fake(backend=FakeBackend(self.answers, fail_after=1))
        raw = raw_path(self.root, "canary", "coldstart")
        first_bytes = raw.read_bytes()
        backend = FakeBackend(self.answers)
        self.run_fake(backend=backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(raw.read_bytes().startswith(first_bytes))
        self.assertEqual(len(read_jsonl(raw)), 2)
        self.assertTrue(canary_ok(self.config, "coldstart", self.root, self.identity))
        with patch.object(FakeBackend, "infer", side_effect=AssertionError("must not generate again")):
            self.assertEqual(self.run_fake()["state"], "already_complete")
        events = read_jsonl(self.root / "results/runs/cloud_pilot20/canary/coldstart/events.jsonl")
        self.assertEqual(events[0]["error_type"], "ConnectionError")

    def test_resume_and_canary_refuse_changed_runtime(self):
        self.run_fake()
        with self.assertRaisesRegex(ValueError, "Resume"):
            self.run_fake(identity={"different": True})
        with self.assertRaisesRegex(ValueError, "Canary"):
            canary_ok(self.config, "coldstart", self.root, {"different": True})

    def test_canary_requires_untruncated_nonempty_image_response(self):
        self.run_fake(backend=FakeBackend(self.answers, reason="length"))
        with self.assertRaises(ValueError):
            canary_ok(self.config, "coldstart", self.root)
        self.run_fake(model="rl", backend=FakeBackend(self.answers, response=""))
        with self.assertRaises(ValueError):
            canary_ok(self.config, "rl", self.root)

    def test_lock_prevents_concurrent_writes(self):
        raw = raw_path(self.root, "canary", "coldstart")
        with run_lock(raw.with_suffix(".lock")):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                self.run_fake()
        self.run_fake()

    def test_reporting_refuses_a_concurrent_writer(self):
        with run_lock(raw_path(self.root, "pilot", "coldstart").with_suffix(".lock")):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                report(self.config, root=self.root)

    def test_malformed_tail_is_not_silently_deleted(self):
        self.run_fake()
        raw = raw_path(self.root, "canary", "coldstart")
        with raw.open("a") as f:
            f.write('{"question_id":')
        before = raw.read_bytes()
        with self.assertRaisesRegex(ValueError, "Malformed"):
            self.run_fake()
        self.assertEqual(raw.read_bytes(), before)

    def test_real_scoring_partial_coverage_then_full_export(self):
        self.run_fake(phase="pilot")
        before = raw_path(self.root, "pilot", "coldstart").read_bytes()
        partial = report(self.config, root=self.root)
        self.assertFalse(partial["complete"])
        self.assertEqual(len(partial["models"]["rl"]["missing_ids"]), 20)
        self.run_fake(model="rl", phase="pilot")
        result = report(self.config, root=self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["paired_n"], 20)
        self.assertEqual(result["transitions"]["1->1"], 20)
        self.assertEqual(raw_path(self.root, "pilot", "coldstart").read_bytes(), before)
        archive = export(self.config, self.root)
        import zipfile
        with zipfile.ZipFile(archive["file"]) as z:
            self.assertIsNone(z.testzip())
            checks = json.loads(z.read("SHA256SUMS.json"))
            self.assertTrue(any(k.endswith("coldstart.jsonl") for k in checks))
            self.assertFalse(any("models/" in k or ".token" in k for k in checks))

    def test_report_rejects_protocol_drift(self):
        self.run_fake(phase="pilot")
        self.config["max_new_tokens"] = 512
        with self.assertRaisesRegex(ValueError, "protocol/session"):
            report(self.config, root=self.root)


class CheckpointViewTests(unittest.TestCase):
    def test_merged_alias_preserves_original_and_index(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d) / "source", Path(d) / "load"
            source.mkdir()
            (source / "model-merged.safetensors").write_bytes(b"fixture, not model weights")
            write_json(source / "model.safetensors.index.json", {"weight_map": {"x": "model-merged.safetensors"}})
            write_json(source / "config.json", {})
            before = digest(source / "model-merged.safetensors")
            build_load_view(source, target)
            self.assertTrue((target / "model.safetensors").is_symlink())
            self.assertFalse((target / "model.safetensors.index.json").exists())
            self.assertEqual(digest(target / "model.safetensors"), before)
            build_load_view(source, target)

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source"
            source.mkdir()
            for name in ["one.safetensors", "two.safetensors"]:
                (source / name).write_bytes(b"fixture")
            write_json(source / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
            with self.assertRaisesRegex(ValueError, "index"):
                build_load_view(source, Path(d) / "load")
