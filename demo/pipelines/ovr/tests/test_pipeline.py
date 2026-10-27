import datetime as dt
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ovr.common import ROOT, digest, fingerprint, read_json, read_jsonl, stratified_ids, unique, wilson, write_json, write_jsonl
from ovr.run import approval_check, question_prompt, validate_existing
from ovr.scoring import score
from ovr.report import paired_transition
from ovr.analysis import choose_cases


class IntegrityTests(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            unique([{"question_id": "1"}, {"question_id": "1"}])

    def test_sampling_deterministic_and_balanced(self):
        rows = [{"question_id": f"{lv}-{i}", "level": lv} for lv in range(1, 6) for i in range(15)]
        a = stratified_ids(rows, 6)
        self.assertEqual(a, stratified_ids(rows, 6))
        self.assertEqual(len(a), 30)
        self.assertTrue(all(sum(x.startswith(f"{lv}-") for x in a) == 6 for lv in range(1, 6)))

    def test_wilson_extremes(self):
        self.assertEqual(wilson(0, 0), (None, None))
        low, high = wilson(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0)
        self.assertLess(wilson(10, 10)[0], 1)

    def test_no_answer_leakage(self):
        text = question_prompt({"question": "Q?", "options": ["one", "two"], "answer": "SECRET"}, {"prompt_suffix": "reason"})
        self.assertNotIn("SECRET", text)
        self.assertIn("(B) two", text)

    def test_resume_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.jsonl"
            write_jsonl(p, [{"question_id": "1", "model": "base", "run_fingerprint": "a"}])
            self.assertEqual(len(validate_existing(p, ["1", "2"], "base", "a")), 1)
            for ids, model, fp in [(["2"], "base", "a"), (["1"], "rl", "a"), (["1"], "base", "b")]:
                with self.assertRaises(ValueError):
                    validate_existing(p, ids, model, fp)

    def test_corrupt_tail_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.jsonl"
            p.write_text('{"question_id":"1"}\n{"question_id":')
            with self.assertRaises(ValueError):
                read_jsonl(p)

    def test_budget_approval(self):
        cfg = {"value": 1}
        now = dt.datetime(2026, 10, 21, tzinfo=dt.timezone.utc)
        good = {"approved": True, "stage": "pilot", "protocol_sha256": fingerprint(cfg),
                "expires_at": "2026-10-22T00:00:00+00:00", "hourly_rate_cny": 10,
                "total_budget_cny": 100, "session_max_minutes": 30}
        self.assertEqual(approval_check(good, cfg, "pilot", now), 1800)
        for bad in [{"approved": False}, {"stage": "full"}, {"total_budget_cny": 0},
                    {"protocol_sha256": "wrong"}, {"expires_at": "2026-10-20T00:00:00+00:00"},
                    {"spent_before_session_cny": 101}]:
            with self.assertRaises(ValueError):
                approval_check(dict(good, **bad), cfg, "pilot", now)

    def test_transition_direction_and_partial_denominator(self):
        a = {"1": {"correct": False}, "2": {"correct": True}}
        b = {"1": {"correct": True}, "2": {"correct": False}}
        result = paired_transition("coldstart", "rl", a, b, 3040)
        self.assertEqual(result["wrong_to_right"], 1)
        self.assertEqual(result["right_to_wrong"], 1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["accuracy_delta"], 0)

    def test_case_categories_not_cherry_picked(self):
        data = {m: {} for m in ["base", "coldstart", "rl"]}
        for qid, flags in [("1", (True, False, True)), ("2", (False, True, False)), ("3", (False, False, False))]:
            for m, flag in zip(data, flags):
                data[m][qid] = {"correct": flag}
        cases = choose_cases(data, ["3", "2", "1"])
        self.assertEqual([c["question_id"] for c in cases], ["1", "2", "3"])


class OfficialScoringTests(unittest.TestCase):
    def test_real_official_evaluator_and_raw_immutability(self):
        # These synthetic answers exercise scoring only. Never write them under project results.
        examples = [
            ("mc", "B", ["2", "4"], "The final answer is \\boxed{B}.", True),
            ("num", "2", [], "The answer is 2.", True),
            ("formula", "1/2", [], "\\boxed{\\frac{1}{2}}", True),
            ("empty", "3", [], "", False),
            ("nested", "3/4", [], "\\boxed{\\frac{3}{4}}", True),
            ("last", "7", [], "\\boxed{4}. Correction: \\boxed{7}", True),
            ("bad", "4", [], "\\boxed{5}", False),
        ]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            q = [{"question_id": a, "answer": b, "options": c, "level": 1, "subject": "test"} for a, b, c, _, _ in examples]
            raw = [{"question_id": a, "model": "base", "response": text, "output_tokens": 1, "finish_reason": "stop"} for a, _, _, text, _ in examples]
            write_jsonl(d / "q.jsonl", q)
            write_jsonl(d / "raw.jsonl", raw)
            before = digest(d / "raw.jsonl")
            score(d / "raw.jsonl", d / "q.jsonl", d / "scored.jsonl")
            self.assertEqual(before, digest(d / "raw.jsonl"))
            scored = read_jsonl(d / "scored.jsonl")
            self.assertEqual([r["correct"] for r in scored], [x[-1] for x in examples])
            self.assertEqual(scored[2]["extracted_answer"], r"\frac{1}{2}")
            with self.assertRaises(ValueError):
                score(d / "raw.jsonl", d / "q.jsonl", d / "raw.jsonl")


class ResumeIntegrationTests(unittest.TestCase):
    def test_interrupt_then_resume_uses_missing_questions_only(self):
        from PIL import Image
        from ovr.run import run
        cfg = read_json(ROOT / "configs/protocol.json")
        cfg["batch_size"] = 1
        calls = []
        should_crash = [True]

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def apply_chat_template(self, messages, **kwargs): return messages[0]["content"][1]["text"]

        class FakeEngine:
            def __init__(self, **kwargs): pass
            def generate(self, inputs, params, **kwargs):
                calls.append(inputs[0]["prompt"])
                if should_crash[0] and len(calls) == 2:
                    raise RuntimeError("SIMULATED INFRASTRUCTURE INTERRUPTION")
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="TEST FIXTURE", token_ids=[1], finish_reason="stop")])]

        modules = {"vllm": types.SimpleNamespace(LLM=FakeEngine, SamplingParams=lambda **kw: kw),
                   "transformers": types.SimpleNamespace(AutoProcessor=FakeProcessor)}
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_json(tmp / "configs/protocol.json", cfg)
            (tmp / "data").mkdir(exist_ok=True)
            Image.new("RGB", (8, 8), "white").save(tmp / "data/image.png")
            write_jsonl(tmp / "data/questions.jsonl", [{"question_id": str(i), "question": f"Q{i}", "image": "data/image.png"} for i in [1, 2]])
            write_json(tmp / "data/splits/pilot.json", ["1", "2"])
            write_json(tmp / "approval.json", {"approved": True, "stage": "pilot", "protocol_sha256": fingerprint(cfg),
                       "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).isoformat(),
                       "hourly_rate_cny": 1, "total_budget_cny": 20, "session_max_minutes": 60})
            with patch("ovr.run.ROOT", tmp), patch("ovr.data.validate"), patch("ovr.run.importlib.metadata.version", return_value="TEST"), patch("ovr.run.subprocess.check_output", return_value="TEST GPU"), patch.dict("sys.modules", modules):
                with self.assertRaises(RuntimeError): run("base", "pilot", tmp / "approval.json")
                self.assertEqual(len(read_jsonl(tmp / "results/raw/pilot/base.jsonl")), 1)
                should_crash[0] = False
                run("base", "pilot", tmp / "approval.json")
                self.assertEqual(len(read_jsonl(tmp / "results/raw/pilot/base.jsonl")), 2)
                self.assertEqual(sum("Q1" in c for c in calls), 1)
                self.assertEqual(sum("Q2" in c for c in calls), 2)


if __name__ == "__main__":
    unittest.main()
