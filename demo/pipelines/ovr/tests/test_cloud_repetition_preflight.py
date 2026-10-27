import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ovr.common import ROOT, read_json
from ovr.cloud import inputs
from ovr.repetition_preflight import choose_ids, inspect_headers, request, validate, visual_effect


def probe(token='Okay', logprob=-0.1, ids=None):
    return {'api_response': {'prompt_token_ids': [7, 8, 9] if ids is None else ids,
                            'choices': [{'token_ids': [1 if token == 'Okay' else 2],
                                         'logprobs': {'content': [{'top_logprobs': [
                                             {'token': 'Okay', 'logprob': logprob}]}]}}]}}


class RepetitionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / 'configs/cloud_vllm_preflight8192.json')
        self.rows, _ = inputs(self.config)

    def test_selection_is_deterministic_stratified_and_excludes_old_cases(self):
        ids = choose_ids(self.rows)
        self.assertEqual(len(set(ids)), 5)
        self.assertFalse({'40', '51'} & set(ids))
        by_id = {r['question_id']: r for r in self.rows}
        self.assertEqual([by_id[q]['level'] for q in ids], [1, 2, 3, 4, 5])
        self.assertEqual(ids, choose_ids(list(reversed(self.rows))))
        validate(self.config, self.rows)
        self.config['max_new_tokens'] = 32768
        with self.assertRaises(ValueError):
            validate(self.config, self.rows)

    def test_image_effect_requires_stable_same_image_control_and_equal_prompt_ids(self):
        self.assertTrue(visual_effect([probe(), probe(logprob=-0.10001), probe(logprob=-0.4)])['image_affects_output_distribution'])
        self.assertFalse(visual_effect([probe(), probe(), probe()])['image_affects_output_distribution'])
        self.assertFalse(visual_effect([probe(), probe(logprob=-0.2), probe(logprob=-0.4)])['image_affects_output_distribution'])
        self.assertFalse(visual_effect([probe(), probe(), probe(logprob=-0.4, ids=[77])])['image_affects_output_distribution'])

    def test_tensor_header_audit_requires_both_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.safetensors'
            def save(keys):
                blob = json.dumps(keys).encode()
                path.write_bytes(struct.pack('<Q', len(blob)) + blob)
            save({'model.embed_tokens.weight': {'shape': [10, 4], 'dtype': 'BF16'}})
            with self.assertRaises(ValueError):
                inspect_headers(tmp)
            save({'model.embed_tokens.weight': {'shape': [10, 4], 'dtype': 'BF16'},
                  'visual.patch_embed.proj.weight': {'shape': [4, 3, 2, 14, 14], 'dtype': 'BF16'}})
            self.assertEqual(inspect_headers(tmp)['vision_tensor_count'], 1)

    def test_sampling_request_is_explicit_and_cannot_silently_lose_topk(self):
        from ovr.diagnostic_vllm import sha_json
        response = {'usage': {'completion_tokens': 1}, 'prompt_token_ids': [1, 2],
                    'choices': [{'token_ids': [99], 'message': {'content': ''}, 'finish_reason': 'length'}]}
        expected = {'input_tokens': 2, 'prompt_token_ids_sha256': sha_json([1, 2])}
        with tempfile.TemporaryDirectory() as tmp, patch('ovr.repetition_preflight.expected_input', return_value=expected), \
                patch('ovr.repetition_preflight.http', return_value=response) as http:
            row = request(self.config, 99, 'rl', self.rows[0], b'fixture', Path(tmp), Path(tmp) / 'events.jsonl')
            body = http.call_args.args[2]
            self.assertEqual((body['temperature'], body['top_p'], body['top_k'], body['max_tokens']), (0.6, 0.95, 40, 8192))
            self.assertEqual(row['response'], '')  # Empty/truncated model attempts are retained, never silently retried.
            response['prompt_token_ids'] = [5]
            with self.assertRaisesRegex(ValueError, 'Prompt tokens'):
                request(self.config, 99, 'rl', self.rows[0], b'fixture', Path(tmp), Path(tmp) / 'events.jsonl')


if __name__ == '__main__':
    unittest.main()
