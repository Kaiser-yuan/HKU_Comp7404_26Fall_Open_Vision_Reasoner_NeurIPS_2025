"""CPU-only tests: no model generation, downloads, SSH or CUDA imports."""
import copy
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ovr.common import ROOT, digest, fingerprint, read_json, read_jsonl, write_json, write_jsonl
from ovr.diagnostic_vllm import sha_json
from ovr.budget_compare import (BASE_CONFIG, CONFIG, WORK, NS, Stream, audit_attempts,
    check, choose, consume_sse, export, has_box, prepare, raw_file, report,
    request, request_body, run, run_dir, server_command, view)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'configs').mkdir()
        shutil.copy(ROOT / BASE_CONFIG, self.root / BASE_CONFIG)
        source = ROOT / 'workstreams/shared_test100'
        shutil.copytree(source, self.root / 'workstreams/shared_test100')
        self.ready = prepare(self.root)
        self.config = read_json(self.root / CONFIG)

    def tearDown(self):
        self.tmp.cleanup()

    def test_selection_has_four_per_level_and_ignores_previous_outcomes(self):
        self.assertEqual(self.ready['difficulty_counts'], {str(i): 4 for i in range(1, 6)})
        rows = read_jsonl(self.root / 'workstreams/shared_test100/inputs.jsonl')
        ids = choose(rows)
        for r in rows:
            r.update(correct=True, response='imaginary correct answer')
        self.assertEqual(ids, choose(rows))
        self.assertFalse(set(ids) & set(self.config['excluded_development_ids']))
        self.assertEqual(prepare(self.root), self.ready)

    def test_config_and_selected_references_cannot_drift(self):
        path = self.root / CONFIG
        cfg = read_json(path)
        cfg['generation']['temperature'] = 0
        write_json(path, cfg)
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            check(self.root)
        write_json(path, self.config)
        path = self.root / WORK / 'references.jsonl'
        rows = read_jsonl(path)
        rows[0]['answer'] = 'tampered'
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, 'references differ'):
            check(self.root)

    def test_request_keeps_official_prompt_and_sampling_without_reference_leak(self):
        item = read_jsonl(self.root / 'workstreams/shared_test100/inputs.jsonl')[0]
        body = request_body(self.config, 'rl', item, b'image')
        self.assertEqual(body['messages'][0]['content'][1]['text'], item['prompt'])
        self.assertEqual((body['max_tokens'], body['temperature'], body['top_k']), (16384, .6, 40))
        self.assertFalse(body['ignore_eos'])
        self.assertNotIn('answer', body)
        command = server_command(self.config, 'rl', '/local/model', 1234)
        self.assertEqual(command[command.index('--max-model-len') + 1], '24576')
        with self.assertRaisesRegex(ValueError, 'runtime limit'):
            run(float('nan'), self.root)

    def test_delta_stream_utf8_token_counts_and_usage(self):
        packets = [
            {'id':'r', 'prompt_token_ids':[9], 'choices':[{'index':0,'delta':{'role':'assistant','content':''}}]},
            {'id':'r', 'choices':[{'index':0,'delta':{'content':'图'},'token_ids':[1,2]}]},
            {'id':'r', 'choices':[{'index':0,'delta':{'content':'答'},'token_ids':[3], 'finish_reason':'stop'}]},
            {'id':'r', 'choices':[], 'usage':{'prompt_tokens':1,'completion_tokens':3}}]
        lines = [b': keepalive\n', b'\n']
        for p in packets:
            lines += [('data: ' + json.dumps(p, ensure_ascii=False) + '\n').encode(), b'\n']
        lines += [b'data: [DONE]\n', b'\n']
        stream = Stream()
        stream.done = consume_sse(lines, lambda p: stream.add(p, .5))
        stream.validate(8, sha_json([9]))
        self.assertEqual(stream.text, '图答')
        self.assertEqual(stream.ids, [1,2,3])
        stream.done = False
        with self.assertRaisesRegex(ValueError, 'Incomplete stream'):
            stream.validate(8, sha_json([9]))
        stream.done = True
        stream.finish = 'length'
        with self.assertRaisesRegex(ValueError, 'Usage/termination'):
            stream.validate(8, sha_json([9]))
        stream.finish = 'stop'
        with self.assertRaisesRegex(ValueError, 'prompt IDs'):
            stream.validate(8, sha_json([8]))

    def test_prefix_crosses_chunk_boundary_and_short_eos_reuses_text(self):
        row = {'model':'rl','question_id':'x','token_ids':list(range(10)), 'response':'abcdefghij',
               'token_timings':[[3,.3],[7,.7],[10,1.0]], 'latency_s':1.1,
               'finish_reason':'stop','stop_reason':None, 'image_sha256':'sha'}
        a = view(row, 4, lambda ids: ''.join('abcdefghij'[i] for i in ids))
        b = view(row, 16)
        self.assertEqual((a['response'], a['finish_reason'], a['output_tokens']), ('abcd','length',4))
        self.assertEqual(a['token_boundary_observed_s'], .7)
        self.assertIsNone(a['latency_s'])
        self.assertEqual((b['response'], b['finish_reason'], b['latency_s']), ('abcdefghij','stop',1.1))
        # EOS exactly on the smaller boundary remains a natural stop.
        self.assertEqual(view(row,10)['finish_reason'], 'stop')

    def test_early_disconnect_does_not_look_complete(self):
        self.assertFalse(consume_sse([b'data: {"choices":[]}\n', b'\n'], lambda _: None))
        with self.assertRaisesRegex(ValueError, 'error'):
            Stream().add({'error': {'message':'failure'}}, 0)

    def test_uncertain_or_duplicate_attempt_is_never_retried(self):
        ids = self.ready['question_ids']
        events = run_dir(self.root) / 'rl/http-events.jsonl'
        write_jsonl(events, [{'event':'submitted','question_id':ids[0]}])
        with self.assertRaisesRegex(ValueError, 'Uncertain'):
            audit_attempts(self.root, 'rl', ids, 'fp')
        write_jsonl(events, [{'event':'submitted','question_id':ids[0]}] * 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit_attempts(self.root, 'rl', ids, 'fp')

    def test_full_stream_request_and_prefix_persistence_without_gpu(self):
        item = next(r for r in read_jsonl(self.root / 'workstreams/shared_test100/inputs.jsonl')
                    if r['question_id'] == self.ready['question_ids'][0])
        packets = [
            {'id':'r','prompt_token_ids':[9],'choices':[{'index':0,'delta':{'content':''}}]},
            {'id':'r','choices':[{'index':0,'delta':{'content':'a'*4096},'token_ids':[97]*4096}]},
            {'id':'r','choices':[{'index':0,'delta':{'content':'a'*4096+'b'*9},
                                 'token_ids':[97]*4096+[98]*9,'finish_reason':'stop'}]},
            {'id':'r','choices':[],'usage':{'prompt_tokens':1,'completion_tokens':8201}}]
        wire = ''.join('data: '+json.dumps(p)+'\n\n' for p in packets)+'data: [DONE]\n\n'
        expected = {'input_tokens':1,'prompt_token_ids_sha256':sha_json([9])}
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ''.join(chr(n) for n in ids)
        with patch('ovr.diagnostic_vllm.expected_input', return_value=expected), \
             patch('urllib.request.urlopen', return_value=io.BytesIO(wire.encode())):
            row = request(self.config,'rl',item,'/unused',1234,Tokenizer(),'fp',self.root)
        self.assertEqual(row['prefix_8192_text'],'a'*8192)
        self.assertEqual(view(row,8192)['finish_reason'],'length')
        self.assertEqual(view(row,16384)['response'],'a'*8192+'b'*9)
        self.assertEqual(len(audit_attempts(self.root,'rl',self.ready['question_ids'],'fp')),1)
        # No JSON stream completion marker: partial journal retained, no completed row.
        second = next(r for r in read_jsonl(self.root / 'workstreams/shared_test100/inputs.jsonl')
                      if r['question_id'] == self.ready['question_ids'][1])
        wire = 'data: '+json.dumps(packets[0])+'\n\n'
        with patch('ovr.diagnostic_vllm.expected_input', return_value=expected), \
             patch('urllib.request.urlopen', return_value=io.BytesIO(wire.encode())):
            with self.assertRaisesRegex(ValueError,'Incomplete stream'):
                request(self.config,'rl',second,'/unused',1234,Tokenizer(),'fp',self.root)
        self.assertEqual(len(read_jsonl(raw_file(self.root,'rl'))),1)
        with self.assertRaisesRegex(ValueError,'Uncertain'):
            audit_attempts(self.root,'rl',self.ready['question_ids'],'fp')

    def fixture(self, model, count):
        rows, events = [], []
        refs = read_jsonl(self.root / WORK / 'references.jsonl')
        for i, ref in enumerate(refs[:count]):
            qid = ref['question_id']
            journal = run_dir(self.root) / model / ('stream-' + qid + '.jsonl')
            write_jsonl(journal, [{'fixture':True, 'done':True}])
            answer = str(ref['answer']) if i % 2 == 0 else ''
            text = '\\boxed{' + answer + '}' if answer else ''
            row = {'model':model,'question_id':qid,'token_ids':[1] if text else [],
                   'response':text, 'prefix_8192_text':text,'token_timings':[[1,.1]] if text else [],
                   'latency_s':.2,'finish_reason':'stop','stop_reason':None,'image_sha256':'fixture',
                   'run_fingerprint':'fixture', 'journal_path':journal.relative_to(self.root).as_posix(),
                   'journal_sha256':digest(journal)}
            rows.append(row)
            events += [{'event':'submitted','question_id':qid},
                       {'event':'received_and_saved','question_id':qid,'trajectory_sha256':fingerprint(row)}]
        write_jsonl(raw_file(self.root, model), rows)
        write_jsonl(run_dir(self.root) / model / 'http-events.jsonl', events)

    def test_official_scoring_roundtrip_complete_and_partial_without_model_weights(self):
        # Scorer source/manifest are copied unchanged. Fixture data never goes to real results.
        for path in ['vendor/mathvision', 'sources']:
            shutil.copytree(ROOT / path, self.root / path)
        shutil.copy(ROOT / 'configs/protocol.json', self.root / 'configs/protocol.json')
        self.fixture('coldstart',20)
        self.fixture('rl',19)
        before = {m:digest(raw_file(self.root,m)) for m in ('coldstart','rl')}
        result = report(self.root)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['summaries']),4)
        self.assertEqual({m:digest(raw_file(self.root,m)) for m in before},before)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            export(self.root)
        self.fixture('rl',20)
        result = report(self.root)
        self.assertTrue(result['complete'])
        self.assertEqual(result['transitions']['rl'],{'observed_n':20,'improved':0,'regressed':0})
        self.assertTrue(all(s['empty_n']==10 for s in result['summaries']))
        # Export includes verified member SHA and no model/credential assets.
        (self.root / WORK / 'README.md').write_text('fixture')
        (self.root / 'ovr').mkdir(exist_ok=True)
        for name in ['budget_compare.py','common.py','scoring.py']:
            shutil.copy(ROOT / 'ovr' / name, self.root / 'ovr' / name)
        receipt = export(self.root)
        self.assertTrue(Path(receipt['file']).exists())
        self.assertGreater(receipt['members_verified'],40)
        self.assertTrue(has_box('\\boxed{\\frac{1}{2}}'))
        self.assertFalse(has_box('\\boxed{'))
        # Mutation of a saved response must be detected by its durable receipt.
        path = raw_file(self.root,'rl')
        rows = read_jsonl(path)
        rows[0]['response'] = 'altered'
        write_jsonl(path,rows)
        with self.assertRaisesRegex(ValueError,'receipt differs'):
            report(self.root)


if __name__ == '__main__':
    unittest.main()
