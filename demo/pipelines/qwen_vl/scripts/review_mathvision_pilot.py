"""Re-score saved outputs without running a model or overwriting raw scores.
Not the official MathVision evaluator. Report answer matching, not reasoning quality.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path


def load_lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('pilot', Path(__file__).with_name('run_mathvision_test.py'))
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    qs = load_lines(args.dataset/'questions.jsonl')
    labels = {r['id']: r for r in load_lines(args.dataset/'answers.jsonl')}
    pred = {r['id']: r for r in load_lines(args.run/'predictions.jsonl') if r['status']=='ok'}
    original = {r['id']: r for r in csv.DictReader((args.run/'scores.csv').open(encoding='utf-8-sig'))}
    digest = hashlib.sha256((args.dataset/'questions.jsonl').read_bytes())
    for q in qs:
        digest.update((args.dataset/q['image']).read_bytes())
    assert digest.hexdigest() == json.loads((args.run/'run_config.json').read_text())['dataset_sha256']
    reviewed=[]
    for q in qs:
        p=pred.get(q['id'],{})
        gold=labels[q['id']]['answer']
        answer=pilot.extract_answer(p.get('raw_output',''))
        method='boxed'
        # Only use the whole response as a fallback if it is a simple unambiguous answer.
        if answer is None and not p.get('truncated'):
            raw=p.get('raw_output','').strip()
            import re
            if re.fullmatch(r'[A-E]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?',raw):
                answer=raw;method='entire_response_single_answer'
        mapped=None
        if q['options'] and answer is not None:
            matches=[chr(65+i) for i,value in enumerate(q['options'])
                     if pilot.conservative_match(answer,value)]
            if len(matches)==1:
                mapped=matches[0];method+=';unique_option_content_match'
        checked=pilot.conservative_match(mapped or answer,gold,bool(q['options']))
        if p.get('truncated'):
            note='Truncated; inspect raw output for repetition or incomplete reasoning.'
        elif checked and mapped:
            note=f'Final answer uniquely matches option {mapped} by content; reasoning not certified.'
        elif checked:
            note='Final answer matches reference; this does not certify the reasoning.'
        elif method == 'entire_response_single_answer':
            note='Whole-response answer recovered without a box; differs from reference.'
        elif answer is None:
            note='No answer extracted; manual review required.'
        else:
            note='Explicit final answer differs from reference; preserved as incorrect for this pilot.'
        reviewed.append({'id':q['id'],'reference_answer':gold,
            'original_extracted_answer':p.get('extracted_answer'),
            'reviewed_answer':answer,'mapped_option':mapped,
            'original_match':original[q['id']]['conservative_match'],
            'reviewed_answer_match':checked,'extraction_method':method if answer is not None else 'none',
            'truncated':p.get('truncated'), 'generated_tokens':p.get('generated_tokens'),
            'inference_seconds':p.get('inference_seconds'),
            'subject':labels[q['id']]['subject'],'level':labels[q['id']]['level'],
            'review_note':note})
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'reviewed_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(reviewed[0]));writer.writeheader();writer.writerows(reviewed)
    summary={
        'total':len(qs),'successful':len(pred),
        'original_conservative_matches':sum(r['original_match']=='True' for r in reviewed),
        'reviewed_final_answer_matches':sum(r['reviewed_answer_match'] for r in reviewed),
        'reviewed_final_answer_match_rate':sum(r['reviewed_answer_match'] for r in reviewed)/len(qs),
        'correct_ids':[r['id'] for r in reviewed if r['reviewed_answer_match']],
        'truncated_ids':[r['id'] for r in reviewed if r['truncated']],
        'original_extraction_failure_ids':[r['id'] for r in reviewed if r['original_extracted_answer'] is None],
        'unresolved_answer_ids':[r['id'] for r in reviewed if r['reviewed_answer'] is None],
        'sum_inference_seconds':sum(r['inference_seconds'] for r in reviewed),
        'median_inference_seconds':statistics.median(r['inference_seconds'] for r in reviewed),
        'total_generated_tokens':sum(r['generated_tokens'] for r in reviewed),
        'scope':f'{len(qs)}-question pilot; final-answer equivalence review, NOT official scoring or reasoning validation',
        'protocol':'boxed extraction; whole-response simple answer fallback; unique option-content mapping; exact/rational comparison',
        'source_predictions_sha256':hashlib.sha256((args.run/'predictions.jsonl').read_bytes()).hexdigest(),
    }
    (args.output/'review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
