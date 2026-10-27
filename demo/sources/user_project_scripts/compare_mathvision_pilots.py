"""Compare saved pilot outputs with a common review protocol; no GPU required."""
import argparse,csv,json,statistics,shutil
from pathlib import Path

def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def csvrows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser()
    for name in ['qwen-run','vl-run','qwen-review','vl-review','dataset','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    qs=jsonl(a.dataset/'questions.jsonl');labels={r['id']:r for r in jsonl(a.dataset/'answers.jsonl')}
    qr={r['id']:r for r in csvrows(a.qwen_review/'reviewed_scores.csv')}
    vr={r['id']:r for r in csvrows(a.vl_review/'reviewed_scores.csv')}
    qp={r['id']:r for r in jsonl(a.qwen_run/'predictions.jsonl') if r['status']=='ok'}
    vp={r['id']:r for r in jsonl(a.vl_run/'predictions.jsonl') if r['status']=='ok'}
    ids={r['id'] for r in qs}
    assert ids==set(qr)==set(vr)==set(qp)==set(vp)
    qc=json.loads((a.qwen_run/'run_config.json').read_text());vc=json.loads((a.vl_run/'run_config.json').read_text())
    for k in ['dataset_sha256','max_new_tokens','max_pixels','min_pixels','dtype','attention','do_sample','prompt_version','torch','transformers']:
        assert qc[k]==vc[k], f'Configuration mismatch: {k}'
    paired=[]
    for q in qs:
        i=q['id'];qok=qr[i]['reviewed_answer_match']=='True';vok=vr[i]['reviewed_answer_match']=='True'
        category='both_correct' if qok and vok else 'vl_improved' if vok else 'vl_regressed' if qok else 'both_incorrect'
        paired.append({'id':i,'subject':labels[i]['subject'],'level':labels[i]['level'],
            'reference_answer':labels[i]['answer'],
            'qwen_answer':qr[i]['reviewed_answer'],'vl_answer':vr[i]['reviewed_answer'],
            'qwen_mapped_option':qr[i]['mapped_option'],'vl_mapped_option':vr[i]['mapped_option'],
            'qwen_answer_match':qok,'vl_answer_match':vok,'category':category,
            'qwen_tokens':qp[i]['generated_tokens'],'vl_tokens':vp[i]['generated_tokens'],
            'qwen_seconds':qp[i]['inference_seconds'],'vl_seconds':vp[i]['inference_seconds'],
            'qwen_truncated':qp[i]['truncated'],'vl_truncated':vp[i]['truncated']})
    def writecsv(path,rows):
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    writecsv(a.output/'paired_results.csv',paired)
    metrics=[]
    for name,review,raw in [('Qwen2.5-VL-7B-Instruct',qr,qp),('VL-Rethinker-7B',vr,vp)]:
        metrics.append({'model':name,'n':len(qs),
            'original_matches':sum(r['original_match']=='True' for r in review.values()),
            'reviewed_answer_matches':sum(r['reviewed_answer_match']=='True' for r in review.values()),
            'truncated':sum(r['truncated'] for r in raw.values()),
            'original_extraction_failures':sum(r['extracted_answer'] is None for r in raw.values()),
            'total_generated_tokens':sum(r['generated_tokens'] for r in raw.values()),
            'sum_inference_seconds':sum(r['inference_seconds'] for r in raw.values()),
            'median_inference_seconds':statistics.median(r['inference_seconds'] for r in raw.values())})
    writecsv(a.output/'metrics.csv',metrics)
    categories={k:sum(r['category']==k for r in paired) for k in ['both_correct','vl_improved','vl_regressed','both_incorrect']}
    summary={'n':len(qs),'metrics':metrics,'paired_categories':categories,
        'scope':'Fixed test subset; final-answer equivalence rules; not official benchmark or validated reasoning.',
        'cache_note':'Qwen did not explicitly record use_cache; VL explicitly enabled it. Timing is descriptive, not a controlled speed benchmark.'}
    (a.output/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    report='''# Qwen and VL-Rethinker on 100 fixed MathVision questions

## Main result

Both models completed all 100 identical questions without execution errors. Under one
post-generation answer extraction and equivalence protocol, Qwen matches 28 reference
answers and VL-Rethinker matches 32. This is a 4-percentage-point difference on this
particular subset, not evidence of a general advantage or an official benchmark score.
The original conservative scorer remains unchanged at 26 and 30 respectively.

| Measure | Qwen | VL-Rethinker |
|---|---:|---:|
| Questions completed | 100 | 100 |
| Original conservative matches | 26 | 30 |
| Unified final-answer matches | 28 | 32 |
| Truncated responses | 2 | 0 |
| Original answer extraction failures | 4 | 2 |
| Total output tokens | 57,334 | 49,720 |
| Sum of measured question inference time | 853.56 s | 741.92 s |

## Paired outcomes

- Both final answers match: 16 questions.
- Qwen incorrect, VL-Rethinker correct: 16 questions.
- Qwen correct, VL-Rethinker incorrect: 12 questions.
- Neither final answer matches: 56 questions.

A higher aggregate score does not mean improvement on every question. Preserve and
discuss both improvements and regressions. See paired_results.csv for all IDs.

## Scoring and interpretation

The same saved-output review script is applied to both models. It extracts the last
complete boxed answer, accepts a whole response only if it is a simple unambiguous
answer, and maps an answer to an option only when it uniquely matches that option's
content. Exact rational equivalence is supported; general symbolic equivalence is not.
It is not the official MathVision evaluator. Remaining mismatches should be inspected
by the team before a final poster claim. No labels or solutions enter inference prompts.

Qwen gains matches at IDs 1324 and 2725 by option-content equivalence. VL-Rethinker gains
matches at IDs 51 and 272 by recovering complete single-number responses without boxes.
Correct final answers do not certify correct reasoning; e.g., Qwen ID1324 assumes an
equilateral triangle without justification despite arriving at the reference expression.

## Shared settings and limitations

All compared IDs, prompts and images were checked. Recorded settings match for BF16,
SDPA, greedy decoding, pixel limits, Transformers 4.57.1, PyTorch 2.8.0+cu128 and a
4096-token output cap. Each model uses its own downloaded chat template/processor;
identical pixel limits do not prove that all preprocessing details are identical.
VL-Rethinker explicitly enables KV caching; Qwen's original run did not record that
flag explicitly. Output lengths also differ, so total time does not establish a
controlled model-speed advantage. The cap can constrain a reasoning model and this
is a shared-budget comparison, not each author's complete published evaluation.

The dataset is a two-stage fixed random sample (20 plus 80 from the remaining pool),
not official testmini. Results must not be tuned to match the paper's reported score.
No claim about OVR training or causal behavior transfer follows from these two models.
OVR-ColdStart/RL results are to be provided by teammates under aligned protocols.

## Reproducibility

Raw predictions, initial scores, configs, exact executed scripts and environment
logs are preserved in the demo. Derived reviews and this comparison are separate.
See comparison_summary.json, metrics.csv and CASEBOOK.md. Run the commands in
scripts/06_reproduce_analysis.sh from a demo folder to reproduce the offline analysis
without GPU access or model weights.
'''
    (a.output/'REPORT.md').write_text(report)
    # Deliberately include improvement, regression, shared correct and shared failure cases.
    chosen=['51','674','1414','2760','1324','1542']
    (a.output/'images').mkdir(exist_ok=True)
    for i in chosen:
        shutil.copy2(a.dataset/'images'/f'{i}.jpg', a.output/'images'/f'{i}.jpg')
    lookup={r['id']:r for r in qs};pair={r['id']:r for r in paired}
    book=['# Paired casebook','These cases illustrate outcomes; they were selected after observing results and are not a second evaluation sample. Full predictions remain in the raw JSONL files. Answer correctness is distinct from reasoning quality.']
    for i in chosen:
        q=lookup[i];r=pair[i]
        book += [f'## ID {i}  {r["category"]}',q['question'],
            f'![Original image]({q["image"]})',
            'Options: '+json.dumps(q['options'],ensure_ascii=False),
            f'Reference: {r["reference_answer"]}. Qwen answer: {r["qwen_answer"] or "not extracted"}. VL answer: {r["vl_answer"] or "not extracted"}.',
            '### Qwen output excerpt', '```text\n'+qp[i]['raw_output'][:1500]+'\n```',
            '### VL-Rethinker output excerpt','```text\n'+vp[i]['raw_output'][:1500]+'\n```']
        if i=='1542':book.append('Qwen truncates after producing an artificial numeric grid; VL terminates but its final answer 1 is still incorrect (reference 3). Finishing without truncation does not imply correctness.')
        if i=='1324':book.append('Both final answers match the reference option, but inspect the reasoning separately; Qwen incorrectly assumes an equilateral triangle.')
    (a.output/'CASEBOOK.md').write_text('\n\n'.join(book)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
