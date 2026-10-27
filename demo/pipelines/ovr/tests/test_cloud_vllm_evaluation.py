import copy
import math
import tempfile
import unittest
from pathlib import Path

from ovr.common import ROOT, digest, read_json, write_json, write_jsonl
from ovr.cloud import raw_path
from ovr.probability_continuation import probability_review
from ovr.vllm_evaluation import preflight_ready, pending_attempts, PREFLIGHT


class VllmEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = read_json(ROOT / 'configs/cloud_vllm_shared1008192.json')
        self.ids = read_json(ROOT / 'configs/cloud_vllm_preflight8192.json')['preflight_ids']
        report = self.root / 'results/reports' / PREFLIGHT
        write_json(report / 'complete.json', {'five_level_answers_completed': True, 'all_servers_stopped': True,
                   'models': {m: {'count': 5} for m in ['coldstart', 'rl']}})
        write_json(report / 'probability_review.json', {'pixel_influence_supported_by_review': True,
                                                       'post_hoc_engineering_review': True})
        for model in ['coldstart', 'rl']:
            rows = [{'model': model, 'question_id': q, 'finish_reason': 'length', 'correct': False,
                     'response': '', 'output_tokens': 8192} for q in self.ids]
            raw = self.root / 'results/raw' / PREFLIGHT / (model + '.jsonl')
            scored = self.root / 'results/scored' / PREFLIGHT / (model + '.jsonl')
            write_jsonl(raw, rows)
            write_jsonl(scored, rows)
            write_json(str(scored) + '.provenance.json', {'raw_sha256': digest(raw), 'scored_sha256': digest(scored)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_readiness_checks_integrity_not_accuracy_or_natural_eos(self):
        self.assertTrue(preflight_ready(self.config, self.root)['ready_for_separate_budget_review'])

    def test_changed_decoding_and_tampered_outputs_block_execution(self):
        self.config['max_new_tokens'] = 4096
        with self.assertRaisesRegex(ValueError, 'protocol'):
            preflight_ready(self.config, self.root)
        self.config['max_new_tokens'] = 8192
        raw = self.root / 'results/raw' / PREFLIGHT / 'rl.jsonl'
        raw.write_text(raw.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            preflight_ready(self.config, self.root)

    def test_uncertain_http_attempt_is_not_automatically_resampled(self):
        event = self.root / 'results/runs' / self.config['results_namespace'] / 'pilot/rl/http-events-1.jsonl'
        write_jsonl(event, [{'event': 'submitted', 'question_id': '40'}])
        with self.assertRaisesRegex(ValueError, 'uncertain'):
            pending_attempts(self.config, 'rl', self.root)
        write_jsonl(raw_path(self.root, 'pilot', 'rl', self.config), [{'question_id': '40'}])
        pending_attempts(self.config, 'rl', self.root)

    def test_probability_review_uses_mass_not_rare_token_log_ratio(self):
        def probe(p, rare, tag, ids=None):
            return {'probe_condition': tag, 'api_response': {'prompt_token_ids': ids or [1, 2],
                    'choices': [{'token_ids': [3], 'logprobs': {'content': [{'top_logprobs': [
                        {'token': 'main', 'logprob': math.log(p)}, {'token': 'rare', 'logprob': math.log(rare)}]}]}}]}}
        a = probe(.9, 1e-9, 'original')
        b = probe(.9, 1e-12, 'same_image_repeat')
        c = probe(.85, 1e-9, 'white_same_size')
        result = probability_review([a, b, c])
        self.assertTrue(result['pixel_influence_supported_by_review'])
        self.assertFalse(result['original_gate_replaced_or_marked_passed'])
        self.assertFalse(probability_review([a, b, probe(.9, 1e-9, 'white_same_size')])['pixel_influence_supported_by_review'])
        self.assertFalse(probability_review([a, b, probe(.85, 1e-9, 'white_same_size', ids=[99])])['pixel_influence_supported_by_review'])


if __name__ == '__main__':
    unittest.main()
