"""Reproduce saved evidence on CPU, without generating new model responses."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
QV = ROOT / 'pipelines/qwen_vl'
OVR = ROOT / 'pipelines/ovr'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def jsonl(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in value), encoding='utf-8')

def csvwrite(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(value[0]))
        w.writeheader()
        w.writerows(value)

def unique(data, key):
    result = {}
    for item in data:
        k = str(item[key])
        if k in result:
            raise ValueError('Duplicate ID: '+k)
        result[k] = item
    return result

def run(args, cwd, log):
    with log.open('w', encoding='utf-8') as f:
        process = subprocess.run([str(x) for x in args], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('Command failed; see '+str(log))

def verify():
    expected = read(ROOT / 'provenance/source_inventory.json')['source_files_sha256']
    manifest = ROOT / 'MANIFEST.sha256.json'
    if manifest.exists():
        expected = read(manifest)
    for name, digest in expected.items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p) != digest:
            raise ValueError('Missing or changed distributed file: '+name)
    print('Verified distributed file hashes:', len(expected), flush=True)
    return expected

def validate_inputs():
    qs = rows(QV / 'data/dataset_test100/questions.jsonl')
    labels = unique(rows(QV / 'data/dataset_test100/answers.jsonl'), 'id')
    shared = rows(OVR / 'workstreams/shared_test100/inputs.jsonl')
    refs = unique(rows(OVR / 'workstreams/shared_test100/references.jsonl'), 'question_id')
    qmap, smap = unique(qs, 'id'), unique(shared, 'question_id')
    assert len(qmap) == len(smap) == 100
    assert list(qmap) == list(smap), 'Question order differs'
    sys.path.insert(0, str(QV / 'scripts'))
    from run_mathvision_test_v3 import build_question
    for k, q in qmap.items():
        s = smap[k]
        assert build_question(q) == s['prompt'], 'Prompt differs: '+k
        assert sha(QV / 'data/dataset_test100' / q['image']) == sha(OVR / 'workstreams/shared_test100' / s['image'])
        assert str(labels[k]['answer']) == str(refs[k]['answer'])
        assert q['options'] == refs[k]['options']
        assert int(labels[k]['level']) == int(s['level'])
    split20 = rows(QV / 'results/qwen_mathvision20_4096/predictions.jsonl')
    split80 = rows(QV / 'results/qwen_mathvision_extra80_4096/predictions.jsonl')
    assert len(unique(split20, 'id')) == 20 and len(unique(split80, 'id')) == 80
    assert not (set(unique(split20, 'id')) & set(unique(split80, 'id')))
    assert set(qmap) == set(unique(split20 + split80, 'id'))
    return qs, labels

def official(raw, references, dest, label):
    """Run the archived official evaluator without rewriting source responses."""
    with tempfile.TemporaryDirectory(prefix='mathvision-official-') as tmp:
        tmp = Path(tmp)
        jsonl(tmp / 'data/test.jsonl', references)
        jsonl(tmp / 'outputs/predictions.jsonl', raw)
        run([sys.executable, OVR / 'vendor/mathvision/evaluate.py'], tmp, dest / (label+'-official.log'))
        scored = rows(tmp / 'outputs/predictions.jsonl')
    assert len(unique(scored, 'id')) == 100
    jsonl(dest / (label+'-official.jsonl'), scored)
    return unique(scored, 'id')

def reproduce():
    if os.name == 'nt':
        raise SystemExit('Use WSL2. The preserved OVR modules use POSIX file locks.')
    initial = verify()
    qs, labels = validate_inputs()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    dest = ROOT / 'outputs' / stamp
    dest.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    write(dest / 'data_validation.json', {'questions': 100, 'same_order_images_prompts_labels': True,
          'qwen_split20_plus80_disjoint': True, 'no_new_generation': True})
    qout = dest / 'qwen_vl'
    qout.mkdir()
    print('1/4 Merge Qwen 20+80 and replay the original Qwen/VL analysis', flush=True)
    run([sys.executable, QV/'scripts/merge_mathvision_runs.py', '--runs',
         QV/'results/qwen_mathvision20_4096', QV/'results/qwen_mathvision_extra80_4096',
         '--dataset', QV/'data/dataset_test100', '--output', qout/'merged'], QV, dest/'qwen_merge.log')
    for name, raw in [('qwen', qout/'merged'), ('vl', QV/'results/vl_rethinker_mathvision100_4096')]:
        run([sys.executable, QV/'scripts/review_mathvision_pilot.py', '--run', raw,
             '--dataset', QV/'data/dataset_test100', '--output', qout/(name+'_review')], QV, dest/(name+'_review.log'))
    run([sys.executable, QV/'scripts/compare_mathvision_pilots.py', '--qwen-run', qout/'merged',
         '--vl-run', QV/'results/vl_rethinker_mathvision100_4096', '--qwen-review', qout/'qwen_review',
         '--vl-review', qout/'vl_review', '--dataset', QV/'data/dataset_test100', '--output', qout/'comparison'],
        QV, dest/'qwen_vl_comparison.log')
    history = read(qout/'comparison/comparison_summary.json')
    assert [x['reviewed_answer_matches'] for x in history['metrics']] == [28, 32]
    assert history['paired_categories'] == {'both_correct':16,'vl_improved':16,'vl_regressed':12,'both_incorrect':56}
    print('2/4 Re-score OVR 100 + budget20 and recount 1,000 teacher traces', flush=True)
    run([sys.executable, OVR/'scripts/reproduce_technical_demo.py'], OVR, dest/'ovr_reproduction.log')
    ovr_work = sorted((OVR/'results/reproduced').glob('*/REPRODUCTION_VERIFICATION.json'), key=lambda p:p.stat().st_mtime)[-1].parent
    ovr_check = read(ovr_work/'REPRODUCTION_VERIFICATION.json')
    assert ovr_check['status'] == 'OFFLINE_REPRODUCTION_VERIFIED'
    print('3/4 Apply the SAME archived official scorer to all four saved model runs', flush=True)
    out = dest/'official_four_models'
    out.mkdir()
    sources = {
        'qwen': rows(qout/'merged/predictions.jsonl'),
        'vl_rethinker': rows(QV/'results/vl_rethinker_mathvision100_4096/predictions.jsonl'),
        'coldstart': rows(OVR/'results/raw/cloud_vllm_shared1008192_v1/pilot/coldstart.jsonl'),
        'rl': rows(OVR/'results/raw/cloud_vllm_shared1008192_v1/pilot/rl.jsonl'),
    }
    references = [dict(labels[str(q['id'])], id=str(q['id']), options=q['options']) for q in qs]
    metrics, grades = [], {}
    for name, source in sources.items():
        index = unique(source, 'id' if name in ('qwen','vl_rethinker') else 'question_id')
        assert set(index) == set(labels) and len(index) == 100
        raw = [{'id':str(q['id']), 'response':index[str(q['id'])]['raw_output' if name in ('qwen','vl_rethinker') else 'response']} for q in qs]
        grades[name] = official(raw, references, out, name)
        if name in ('qwen','vl_rethinker'):
            tokens = [x['generated_tokens'] for x in source]
            truncated = sum(bool(x['truncated']) for x in source)
        else:
            tokens = [x['output_tokens'] for x in source]
            truncated = sum(x['finish_reason']=='length' for x in source)
            archived = unique(rows(OVR/f'results/scored/cloud_vllm_shared1008192_v1/pilot/{name}.jsonl'), 'question_id')
            assert all(grades[name][i]['correct'] == archived[i]['correct'] for i in grades[name])
        k = sum(bool(x['correct']) for x in grades[name].values())
        metrics.append({'model':name,'n':100,'correct':k,'accuracy':k/100,'truncated':truncated,
            'mean_output_tokens':statistics.mean(tokens),'output_budget':4096 if name in ('qwen','vl_rethinker') else 8192,
            'decoding':'greedy' if name in ('qwen','vl_rethinker') else 'sampling T0.6 p0.95 k40',
            'backend':'Transformers SDPA' if name in ('qwen','vl_rethinker') else 'vLLM 0.11.0',
            'controlled_four_way_comparison':False})
    csvwrite(out/'metrics.csv', metrics)
    paired = [{'id':str(q['id']), 'level':labels[str(q['id'])]['level'],
               **{name+'_correct':grades[name][str(q['id'])]['correct'] for name in grades}} for q in qs]
    csvwrite(out/'paired_correctness.csv', paired)
    print('4/4 Check source immutability and export a compact summary', flush=True)
    assert all(sha(ROOT/name) == h for name,h in initial.items()), 'Distributed evidence changed'
    budget = read(ovr_work/'results/reports/cloud_vllm_budget20_prefix_v1/comparison.json')
    length = read(ovr_work/'results/reports/teacher_student_lengths_v1/comparison.json')
    result = {'status':'ALL_OFFLINE_REPRODUCTION_VERIFIED','new_generations':0,'gpu_used':False,
        'distributed_files_unchanged':True,'inputs_same_ids_images_prompts_labels':True,
        'original_qwen_vl_history':history,'official_four_model_metrics':metrics,
        'ovr_reproduction':str(ovr_work.relative_to(ROOT)), 'budget':budget,'length':length,
        'interpretation':'Performance is sensitive to output budget and answer extraction/finalization. These observations do not establish that RL intrinsically decreases reasoning ability.',
        'limits':['All four models share data but have different generation budgets/backends/decoding.',
                  'Budget20 consists of 40 trajectories with 80 scored prefix views, not 80 independent generations.',
                  'This is checkpoint evaluation and offline re-analysis, not training reproduction.']}
    write(dest/'VERIFICATION.json', result)
    table = '\n'.join('|{model}|{correct}/100|{truncated}|{output_budget}|{decoding}|'.format(**m) for m in metrics)
    (dest/'RESULTS.md').write_text('# 本次离线复算结果\n\n原始回答未改写；没有新增模型生成。\n\n'
        '|模型|统一官方评分|截断数|输出上限|解码|\n|---|---:|---:|---:|---|\n'+table+
        '\n\n同评分器不等于同推理协议；四模型表仅作描述，不作严格能力排名。\n\n'
        '预算实验：ColdStart 4/20 → 10/20，截断 11 → 0；RL 6/20 → 8/20，截断 15 → 4。'
        '8k 是同一条16k回答的token前缀。原100题34/100与31/100不替换。\n', encoding='utf-8')
    print(json.dumps({'status':result['status'],'output':str(dest),'official_metrics':metrics},ensure_ascii=False,indent=2),flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.verify_only:
        verify()
        validate_inputs()
        print('INPUT_AND_FILE_CHECKS_PASSED')
    else:
        reproduce()

if __name__ == '__main__':
    main()
