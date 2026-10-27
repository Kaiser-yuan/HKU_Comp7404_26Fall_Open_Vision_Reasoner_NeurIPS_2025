"""Run the unchanged original CPU pipeline and add an English result summary."""
import argparse,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def summary(data):
 rows=data['official_four_model_metrics']
 table='\n'.join('|{model}|{correct}/100|{truncated}|{output_budget}|{decoding}|'.format(**r) for r in rows)
 return ('# Verified offline reproduction results\n\n'
 'Saved responses were re-scored; no new inference or training was performed. Distributed source files remain unchanged.\n\n'
 '| Model | Official correct | Limit hits | Output cap | Decoding |\n|---|---:|---:|---:|---|\n'+table+
 '\n\nThe original 100-question runs share inputs and scoring, but differ in generation budget, decoding and backend. This is not a controlled capability ranking.\n\n'
 '## Separate output-budget follow-up\n\n'
 '| Model | Correct at 8k | Correct at 16k | Limit hits at 8k | Limit hits at 16k |\n'
 '|---|---:|---:|---:|---:|\n| ColdStart | 4/20 | 10/20 | 11/20 | 0/20 |\n| RL | 6/20 | 8/20 | 15/20 | 4/20 |\n\n'
 'Each 8k view is the token prefix of the same 16k trajectory. The exploratory follow-up does not replace the original 100-question scores. No additional same-20-question Qwen comparison is included.\n\n'
 'Observed scores depend on output budget and answer finalization/extraction. More tokens helped some responses finish, but four RL responses still hit the 16k cap; longer generation did not guarantee correctness. These observations do not establish that RL intrinsically reduces reasoning ability.\n\n'
 '## Descriptive reasoning lengths\n\n'
 'Mean reasoning lengths are 4444.473 tokens for 1000 teacher traces, 8672.9 for ColdStart, and at least 10968.75 for RL on the 20 student questions. Four RL traces are right-censored; teacher stopping status is unknown. Task and modality differences preclude capability-transfer conclusions from length alone.\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 spec=importlib.util.spec_from_file_location('original_reproduction',ROOT/'scripts/reproduce_all.py')
 base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 if a.verify_only:
  base.verify();base.validate_inputs();print('INPUT_AND_FILE_CHECKS_PASSED');return
 before=set((ROOT/'outputs').glob('*/VERIFICATION.json'))
 base.reproduce()
 added=set((ROOT/'outputs').glob('*/VERIFICATION.json'))-before
 if len(added)!=1:raise RuntimeError('Expected exactly one new verification record')
 record=next(iter(added));data=json.loads(record.read_text())
 assert data['status']=='ALL_OFFLINE_REPRODUCTION_VERIFIED'
 (record.parent/'RESULTS.md').write_text(summary(data),encoding='utf-8')
 print('English results: '+str(record.parent/'RESULTS.md'))

if __name__=='__main__':main()
