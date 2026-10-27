import contextlib
import copy
import io
import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from ovr.cloud import inputs, execute, report, export, raw_path, namespace
from ovr.common import ROOT, read_json, read_jsonl, write_json, digest
from ovr.shared100 import build_question
from ovr.cloud_backend import gpu_check


class Shared100Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = read_json(ROOT / 'configs/cloud_shared100.json')
        shutil.copytree(ROOT / self.config['shared_dir'], self.root / self.config['shared_dir'])
        write_json(self.root / self.config['config_source'], self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_ids_original_images_and_exact_teammate_prompts(self):
        rows, shared = inputs(self.config, self.root)
        source = {str(r['id']):r for r in read_jsonl(shared / 'source/questions.jsonl')}
        validation = read_json(shared / 'validation.json')
        self.assertEqual(len(rows),100)
        self.assertEqual(validation['official_metadata_matches'],100)
        self.assertTrue(validation['seeded_selection_reproduced'])
        for row in rows:
            self.assertEqual(row['prompt'],build_question(source[row['question_id']]))
            self.assertNotIn('<image',row['prompt'])
            self.assertNotIn('answer',row)
        self.assertNotEqual(raw_path(self.root,'pilot','coldstart'),raw_path(self.root,'pilot','coldstart',self.config))

    def test_changed_original_image_and_solution_leak_are_rejected(self):
        rows, shared = inputs(self.config,self.root)
        image=shared/rows[0]['image'];original=image.read_bytes();image.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'hash'): inputs(self.config,self.root)
        image.write_bytes(original)
        rows[0]['solution']='NEVER_SEND'
        path=shared/'inputs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        self.config['inputs_sha256']=digest(path)
        with self.assertRaisesRegex(ValueError,'Reference'):inputs(self.config,self.root)

    def test_results_namespace_and_counts_cannot_be_silently_changed(self):
        with self.assertRaisesRegex(ValueError,'namespace'): namespace({'results_namespace':'../outside'})
        self.config['level_counts']['1']+=1
        with self.assertRaisesRegex(ValueError,'distribution'):inputs(self.config,self.root)

    def test_full100_resume_scoring_and_export_use_separate_paths(self):
        references=read_jsonl(self.root/self.config['shared_dir']/'references.jsonl')
        answers={r['question_id']:r['answer'] for r in references}
        class FixtureBackend:
            metadata={'unit_test_fixture':True,'not_real_model_output':True}
            calls=0
            def infer(self,prompt,image_path,max_time_s):
                self.calls+=1
                return {'response':'\\boxed{'+answers[image_path.stem]+'}', 'output_tokens':1,
                        'finish_reason':'stop','latency_s':.01,'image_tensor_checked':True}
        with contextlib.redirect_stdout(io.StringIO()):
            for model in self.config['models']:
                fake=FixtureBackend()
                done=execute(self.config,model,'pilot',{'test_fixture':True},lambda:fake,self.root)
                self.assertEqual(done['completed'],100)
                again=execute(self.config,model,'pilot',{'test_fixture':True},lambda:fake,self.root)
                self.assertEqual(again['state'],'already_complete')
                self.assertEqual(fake.calls,100)
            result=report(self.config,root=self.root)
            self.assertTrue(result['complete']);self.assertEqual(result['paired_n'],100)
            packed=export(self.config,self.root)
        self.assertIn('SHARED100',packed['file'])
        self.assertFalse((self.root/'results/raw/cloud_pilot20').exists())
        import zipfile
        with zipfile.ZipFile(packed['file']) as z:
            self.assertIsNone(z.testzip())
            self.assertIn('configs/cloud_shared100.json',z.namelist())
            self.assertFalse(any('.token' in p or '/models/' in p for p in z.namelist()))


class RuntimeProtocolTests(unittest.TestCase):
    def fake_runtime(self,config):
        fake=types.SimpleNamespace(version=types.SimpleNamespace(cuda='12.8'),cuda=types.SimpleNamespace(
            is_available=lambda:True,is_bf16_supported=lambda:True,
            get_device_properties=lambda n:types.SimpleNamespace(total_memory=48*2**30,name='fixture'),
            get_device_capability=lambda n:(8,9)))
        versions={'torch':'2.8.0+cu128','torchvision':'0.23.0+cu128','transformers':'4.57.1','accelerate':'1.11.0',
                  'huggingface-hub':'0.35.3','safetensors':'0.6.2','tokenizers':'0.22.1','Pillow':'11.3.0'}
        with patch.dict('sys.modules',{'torch':fake}),patch('platform.system',return_value='Linux'),\
             patch('platform.python_version_tuple',return_value=('3','12','3')),patch('platform.python_version',return_value='3.12.3'),\
             patch('ovr.cloud_backend.versions',return_value=versions),patch('subprocess.check_output',return_value='580.105.08\n'):
            return gpu_check(config)

    def test_image_runtime_is_explicit_and_cuda_build_is_checked(self):
        self.assertEqual(self.fake_runtime({'runtime':{'python':'3.12','torch_cuda':'12.8'}})['torch_cuda'],'12.8')
        with self.assertRaisesRegex(RuntimeError,'Python'):self.fake_runtime({})
        with self.assertRaisesRegex(RuntimeError,'CUDA build'):self.fake_runtime({'runtime':{'python':'3.12','torch_cuda':'12.6'}})
