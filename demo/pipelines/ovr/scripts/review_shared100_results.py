"""CPU-only, post-hoc review of the teammate ZIP against frozen OVR shared100."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ovr.common import digest,read_json,read_jsonl,unique,wilson,write_json,write_jsonl
from ovr.scoring import score


def has_nonempty_box(text):
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{',text):
        depth=1;end=match.end()
        while end<len(text) and depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        if not depth and text[match.end():end-1].strip():return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teammate-zip',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    shared=ROOT/'workstreams/shared_test100';inputs=unique(read_jsonl(shared/'inputs.jsonl'))
    ids=list(inputs);assert len(ids)==100
    models={};sources={};raw_hashes={}
    for model in ['coldstart','rl']:
        raw=ROOT/f'results/raw/cloud_vllm_shared1008192_v1/pilot/{model}.jsonl'
        scored=ROOT/f'results/scored/cloud_vllm_shared1008192_v1/pilot/{model}.jsonl'
        raw_hashes[raw]=digest(raw)
        provenance=read_json(str(scored)+'.provenance.json')
        assert provenance['raw_sha256']==digest(raw) and provenance['scored_sha256']==digest(scored)
        models[model]=unique(read_jsonl(scored));assert set(models[model])==set(ids)
        sources[model]={'raw_sha256':digest(raw),'scored_sha256':digest(scored),'official_count':100,'official_correct':sum(x['correct'] for x in models[model].values())}
    zip_sha=digest(args.teammate_zip)
    with zipfile.ZipFile(args.teammate_zip) as z,tempfile.TemporaryDirectory(prefix='ovr-peer-score-') as tmp:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for model,folder,review_folder in [('qwen','Qwen_100题合并结果','Qwen_统一规则复核'),('vl_rethinker','VL_Rethinker_100题原始结果','VL_Rethinker_统一规则复核')]:
            prefix='100题结果共享包/'+folder+'/'
            content=z.read(prefix+'predictions.jsonl')
            rows=[json.loads(x) for x in content.decode('utf-8-sig').splitlines() if x.strip()]
            by_id=unique(rows,'id');assert set(by_id)==set(ids) and len(rows)==100
            assert all(x['status']=='ok' and x['prompt']==inputs[str(x['id'])]['prompt'] for x in rows)
            raw=Path(tmp)/f'{model}.jsonl';scored=Path(tmp)/f'{model}.scored.jsonl'
            # "base" is solely the existing scoring adapter's allowed schema label.
            # File and report keys retain the actual bundle label; no identity claim.
            canonical=[{'model':'base','question_id':str(x['id']),'response':x['raw_output'],
                'output_tokens':x['generated_tokens'],'finish_reason':'length' if x['truncated'] else 'stop',
                'latency_s':x['inference_seconds']} for x in rows]
            write_jsonl(raw,canonical);result=score(raw,shared/'references.jsonl',scored)
            models[model]=unique(read_jsonl(scored))
            review=read_json_from_zip(z,'100题结果共享包/'+review_folder+'/review_summary.json')
            official_correct={q for q,x in models[model].items() if x['correct']}
            reviewed_correct=set(review['correct_ids'])
            sources[model]={'member':prefix+'predictions.jsonl','raw_sha256':hashlib.sha256(content).hexdigest(),
                'official_count':result['count'],'official_correct':result['correct'],
                'agrees_with_peer_review_correct_ids':official_correct==reviewed_correct,
                'official_only_correct_ids':sorted(official_correct-reviewed_correct),
                'peer_review_only_correct_ids':sorted(reviewed_correct-official_correct),
                'run_config_as_provided':read_json_from_zip(z,prefix+'run_config.json'),
                'identity_status':'Bundle label only: immutable HF revision and weight hashes absent',
                'scoring_adapter_model_label':'base; schema compatibility only, not actual model identity'}
        for q in inputs.values():
            image=z.read('100题结果共享包/共同100题及图片/images/'+q['question_id']+'.jpg')
            assert hashlib.sha256(image).hexdigest()==q['image_sha256']
    summary={}
    for model,rows in models.items():
        vals=list(rows.values());n=len(vals);correct=sum(bool(x['correct']) for x in vals)
        lo,hi=wilson(correct,n)
        summary[model]={'n':n,'correct':correct,'accuracy':correct/n,'wilson_ci95':[lo,hi],
            'mean_output_tokens':statistics.mean(x['output_tokens'] for x in vals),
            'median_output_tokens':statistics.median(x['output_tokens'] for x in vals),
            'truncated':sum(x['finish_reason']=='length' for x in vals),
            'balanced_nonempty_boxed_anywhere':sum(has_nonempty_box(x['response']) for x in vals),
            'think_closed':sum('</think>' in x['response'] for x in vals),
            'by_finish':{f:{'n':sum(x['finish_reason']==f for x in vals),'correct':sum(bool(x['correct']) for x in vals if x['finish_reason']==f)} for f in sorted({x['finish_reason'] for x in vals})},
            'by_level':{str(l):{'n':sum(int(inputs[q]['level'])==l for q in rows),'correct':sum(bool(x['correct']) for q,x in rows.items() if int(inputs[q]['level'])==l)} for l in range(1,6)}}
    paired=[]
    for q in ids:
        a,b=models['coldstart'][q],models['rl'][q]
        paired.append({'question_id':q,'level':inputs[q]['level'],'subject':inputs[q]['subject'],
            'coldstart_correct':a['correct'],'rl_correct':b['correct'],
            'coldstart_truncated':a['finish_reason']=='length','rl_truncated':b['finish_reason']=='length',
            'coldstart_nonempty_boxed':has_nonempty_box(a['response']),'rl_nonempty_boxed':has_nonempty_box(b['response']),
            'coldstart_think_closed':'</think>' in a['response'],'rl_think_closed':'</think>' in b['response'],
            'qwen_correct':models['qwen'][q]['correct'],'vl_rethinker_correct':models['vl_rethinker'][q]['correct']})
    transition={}
    for a,b in [(0,0),(0,1),(1,0),(1,1)]:
        selected=[x for x in paired if x['coldstart_correct']==bool(a) and x['rl_correct']==bool(b)]
        transition[f'{a}->{b}']={'n':len(selected),'ids':[x['question_id'] for x in selected],
            'coldstart_truncated':sum(x['coldstart_truncated'] for x in selected),'rl_truncated':sum(x['rl_truncated'] for x in selected),
            'rl_no_nonempty_boxed':sum(not x['rl_nonempty_boxed'] for x in selected),
            'rl_think_unclosed':sum(not x['rl_think_closed'] for x in selected)}
    finish_groups=[]
    for ac in [False,True]:
        for bc in [False,True]:
            g=[x for x in paired if x['coldstart_truncated']==ac and x['rl_truncated']==bc]
            finish_groups.append({'coldstart_truncated':ac,'rl_truncated':bc,'n':len(g),
                'coldstart_correct':sum(x['coldstart_correct'] for x in g),'rl_correct':sum(x['rl_correct'] for x in g)})
    loss,gain=transition['1->0']['n'],transition['0->1']['n'];n=loss+gain
    pvalue=min(1,2*sum(math.comb(n,k) for k in range(min(loss,gain)+1))/2**n) if n else 1.0
    result={'status':'CPU_REVIEW_COMPLETE_NO_NEW_INFERENCE','teammate_archive_sha256':zip_sha,
        'same_100_ids_prompts_and_image_hashes':True,'four_models':summary,'source_provenance':sources,
        'upstream_evaluator_commit':read_json(ROOT/'configs/protocol.json')['upstream_mathvision_commit'],
        'ovr_transitions':transition,'finish_conditioned_descriptive_groups':finish_groups,
        'exploratory_exact_mcnemar':{'discordant_n':n,'rl_improvements':gain,'rl_regressions':loss,'two_sided_p':pvalue},
        'feature_definitions':{'balanced_nonempty_boxed_anywhere':'Brace-balanced nonempty boxed/fbox span anywhere; not necessarily a finalized answer',
            'think_closed':'Literal closing think marker in saved output; not a cognitive mechanism',
            'truncated':'OVR finish_reason length; peer original truncated boolean',
            'statistical_scope':'Post-hoc exploratory; same 100 questions, single generation per model; completion-conditioned rates are selection-biased'},
        'comparison_limit':'Two teammate models greedy/4096/Transformers; OVR sampled/8192/vLLM. Common scoring alone does not create a controlled four-model comparison.',
        'human_behavior_annotation_status':'Not performed by this script; AI-assisted case review is separate and awaits B review'}
    for raw,h in raw_hashes.items():assert digest(raw)==h
    assert digest(args.teammate_zip)==zip_sha
    write_json(out/'four_model_review.json',result)
    write_json(out/'paired_features.json',paired)
    print(json.dumps({'status':result['status'],'official_correct':{m:s['correct'] for m,s in summary.items()},'regressions':transition['1->0'],'mcnemar_p':pvalue},ensure_ascii=False))


def read_json_from_zip(z,name):return json.loads(z.read(name).decode('utf-8-sig'))


if __name__=='__main__':main()
