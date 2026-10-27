import unittest,random,json
from pathlib import Path
from ovr.trace_lengths import extract,describe,validate_teacher,ROOT

class TraceLengthTests(unittest.TestCase):
 def test_explicit_boundary_excludes_final_answer(self):
  self.assertEqual(extract('<think>check the shape</think>Final 7'),('check the shape','Final 7','explicit_closed',False))
 def test_template_prefill_is_student_only(self):
  self.assertEqual(extract('checking</think>Final',True),('checking','Final','prefilled_closed',False))
  self.assertIsNone(extract('checking</think>Final',False)[0])
 def test_capped_student_is_lower_bound(self):
  r=extract('still checking',True,True);self.assertEqual(r[0],'still checking');self.assertTrue(r[3])
  self.assertIsNone(extract('still checking',False,False)[0])
 def test_ambiguous_tags_are_not_silently_truncated(self):
  self.assertIsNone(extract('<think>first</think><think>second</think>')[0])
  self.assertIsNone(extract('</think>wrong order<think>')[0])
 def test_empty_unknown_is_not_zero_reasoning(self):
  self.assertIsNone(extract('Answer only')[0]);self.assertEqual(describe([])['mean'],None)
 def test_descriptive_quantiles(self):
  d=describe([1,3,5,7]);self.assertEqual(d['mean'],4);self.assertEqual(d['median'],4);self.assertEqual(d['p25'],2.5)
 def test_sample_is_global_uniform_draw_not_equal_shard_sampling(self):
  p=json.loads((ROOT/'workstreams/teacher_lengths/protocol.json').read_text());n=p['population_rows_from_parquet_metadata']
  expected=sorted(random.Random(p['seed']).sample(range(n),p['sample_size']))
  actual=sorted(s['global_offset']+i for s in p['files'] for i in s['sample_row_indices'])
  self.assertEqual(actual,expected);self.assertEqual(len(set(actual)),1000);self.assertEqual(n,1637924)

 def teacher_fixture(self):
     protocol={'revision':'fixed','sample_size':2,'files':[{'file':'a.parquet','global_offset':0,'sample_row_indices':[1,9]}]}
     api={'sha':'fixed','siblings':[{'rfilename':'a.parquet','lfs':{'sha256':'file-sha'}}]}
     rows=[{'global_row_index':i,'file':'a.parquet','row_index':i,'teacher_dataset_revision':'fixed','teacher_file_sha256':'file-sha','response_text_tokens':10,'question_sha256':'a'*64,'response_sha256':'b'*64} for i in [1,9]]
     return rows,protocol,api
 def test_teacher_receipt_rejects_duplicate_or_replaced_sample(self):
     rows,p,a=self.teacher_fixture();self.assertTrue(validate_teacher(rows,p,a))
     rows[1]=rows[0].copy()
     with self.assertRaises(AssertionError):validate_teacher(rows,p,a)
 def test_teacher_receipt_rejects_different_version_or_file(self):
     rows,p,a=self.teacher_fixture();rows[0]['teacher_dataset_revision']='different'
     with self.assertRaises(AssertionError):validate_teacher(rows,p,a)
     rows,p,a=self.teacher_fixture();rows[0]['teacher_file_sha256']='different'
     with self.assertRaises(AssertionError):validate_teacher(rows,p,a)


if __name__=='__main__':unittest.main()
