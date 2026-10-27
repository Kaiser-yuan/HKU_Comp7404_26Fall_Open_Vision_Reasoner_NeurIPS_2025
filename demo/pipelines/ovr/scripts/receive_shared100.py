"""Verify and receive the completed shared100 archive; CPU-only, no inference."""
import argparse
import base64
from collections import Counter
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ovr.common import digest, fingerprint, read_json, read_jsonl, unique, wilson, write_csv, write_json
from ovr.scoring import score

NS = 'cloud_vllm_shared1008192_v1'
EXPECTED_SHA = '64e507bc52c7caffa367cee4655931d0531572d097a12c449f8e1d9a7da22786'
DEV_IDS = {'40', '51', '689', '2968', '465', '1542', '2016'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    assert digest(args.archive) == EXPECTED_SHA, 'Whole-archive SHA mismatch'
    out = ROOT / f'results/reports/{NS}/pilot'
    with tempfile.TemporaryDirectory(prefix='ovr-shared100-receipt-') as tmp:
        received = Path(tmp)
        with zipfile.ZipFile(args.archive) as z:
            names = z.namelist()
            assert len(names) == len(set(names)), 'Duplicate archive members'
            assert z.testzip() is None, 'Archive CRC failure'
            manifest = json.loads(z.read('SHA256SUMS.json'))
            assert set(names) == set(manifest) | {'SHA256SUMS.json', 'README.txt'}
            assert len(manifest) == 129
            for name, sha in manifest.items():
                p = PurePosixPath(name)
                assert not p.is_absolute() and '..' not in p.parts
                assert name.startswith(('workstreams/shared_test100/', f'results/raw/{NS}/',
                    f'results/scored/{NS}/', f'results/runs/{NS}/', f'results/reports/{NS}/')) or name == 'configs/cloud_vllm_shared1008192.json'
                data = z.read(name)
                assert hashlib.sha256(data).hexdigest() == sha, f'Member SHA: {name}'
                target = received / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                existing = ROOT / name
                if existing.exists():
                    assert digest(existing) == sha, f'Existing local file differs; refusing overwrite: {name}'
        shared = received / 'workstreams/shared_test100'
        inputs = read_jsonl(shared / 'inputs.jsonl')
        refs = unique(read_jsonl(shared / 'references.jsonl'))
        ids = [str(x['question_id']) for x in inputs]
        assert len(ids) == len(set(ids)) == 100 and set(ids) == set(refs)
        assert read_json(shared / 'question_ids.json') == ids
        from PIL import Image
        for q in inputs:
            image = shared / q['image']
            assert digest(image) == q['image_sha256']
            with Image.open(image) as im:
                im.verify()
        config = read_json(received / 'configs/cloud_vllm_shared1008192.json')
        assert digest(shared / 'inputs.jsonl') == config['inputs_sha256']
        assert digest(shared / 'references.jsonl') == config['references_sha256']
        summary_path = received / f'results/reports/{NS}/pilot/summary.csv'
        summary = list(csv.DictReader(io.StringIO(summary_path.read_text())))
        data, evidence, overall = {}, {}, {}
        for model in ['coldstart', 'rl']:
            raw_path = received / f'results/raw/{NS}/pilot/{model}.jsonl'
            cloud_path = received / f'results/scored/{NS}/pilot/{model}.jsonl'
            before = digest(raw_path)
            raw, cloud = read_jsonl(raw_path), unique(read_jsonl(cloud_path))
            assert len(raw) == 100 and [str(x['question_id']) for x in raw] == ids
            assert len(unique(raw)) == 100 and set(cloud) == set(ids)
            assert all(x['model'] == model and x['protocol_id'] == config['protocol_id'] and x['status'] == 'generated' for x in raw)
            for row, q in zip(raw, inputs):
                assert row['image_sha256'] == q['image_sha256']
                decoding = row['request_decoding']
                for key, value in {'seed':42, 'max_tokens':8192, 'temperature':0.6, 'top_p':0.95, 'top_k':40, 'repetition_penalty':1.0}.items():
                    assert decoding[key] == value
            runs = received / f'results/runs/{NS}/pilot/{model}'
            sessions = list(runs.glob('session-*.json'))
            assert len(sessions) == 1
            session = read_json(sessions[0])
            assert session['state'] == 'complete' and session['selected_ids'] == ids and session['protocol'] == config
            expected_fp = fingerprint({'identity':session['identity'],'model':model,'phase':'pilot','ids':ids,'config':config})
            assert session['run_fingerprint']==expected_fp and all(row['run_fingerprint']==expected_fp for row in raw)
            servers = list(runs.glob('backend-*/*-server.json'))
            assert len(servers) == 1
            events = [e for p in runs.glob('http-events-*.jsonl') for e in read_jsonl(p)]
            assert Counter((e['event'], str(e['question_id'])) for e in events) == Counter((kind,qid) for qid in ids for kind in ['submitted','received'])
            local_scored = received / f'independent/{model}.jsonl'
            result = score(raw_path, shared / 'references.jsonl', local_scored)
            independent = unique(read_jsonl(local_scored))
            assert all((independent[q]['correct'], independent[q]['extracted_answer']) == (cloud[q]['correct'], cloud[q]['extracted_answer']) for q in ids)
            assert digest(raw_path) == before
            data[model] = independent
            evidence[model] = {**result, 'unique_ids':100, 'missing_ids':[], 'duplicate_ids':[], 'raw_sha256':before,
                'cloud_scored_sha256':digest(cloud_path), 'mac_scored_sha256':digest(local_scored),
                'official_scores_match':True, 'raw_unchanged':True, 'http_submitted':100, 'http_received':100,
                'session_started_utc':session['started_utc'], 'session_ended_utc':session['ended_utc']}
            for item in [s for s in summary if s['model'] == model]:
                selected = [x for x in independent.values() if item['group']=='overall' or str(refs[x['question_id']]['level' if item['group']=='difficulty' else 'subject']) == item['value']]
                n = len(selected); k = sum(bool(x['correct']) for x in selected)
                trunc = sum(x['finish_reason'] in {'length','time_limit'} for x in selected)
                assert int(item['n']) == n and int(item['correct']) == k
                assert int(item['length_or_time_limit_count']) == trunc
                lo, hi = wilson(k,n)
                values = {'accuracy_observed':k/n, 'ci95_low':lo, 'ci95_high':hi, 'mean_output_tokens':statistics.mean(x['output_tokens'] for x in selected), 'mean_latency_s':statistics.mean(x['latency_s'] for x in selected), 'truncation_rate_observed':trunc/n}
                assert all(math.isclose(float(item[key]),value,rel_tol=1e-10,abs_tol=1e-10) for key,value in values.items())
                if item['group'] == 'overall': overall[model] = {**values,'n':n,'correct':k,'truncated':trunc,'natural_stop':n-trunc,'empty':sum(not x['response'].strip() for x in selected)}
        transitions = Counter(f"{int(data['coldstart'][q]['correct'])}->{int(data['rl'][q]['correct'])}" for q in ids)
        coverage = read_json(received / f'results/reports/{NS}/pilot/coverage.json')
        assert coverage['complete'] and coverage['paired_n']==100 and dict(transitions)==coverage['transitions']
        paired = list(csv.DictReader(io.StringIO((received/f'results/reports/{NS}/pilot/paired.csv').read_text())))
        assert [p['question_id'] for p in paired]==ids
        for p in paired:
            q=p['question_id'];a,b=data['coldstart'][q],data['rl'][q]
            assert p['transition']==f"{int(a['correct'])}->{int(b['correct'])}"
            assert int(p['coldstart_tokens'])==a['output_tokens'] and int(p['rl_tokens'])==b['output_tokens']
        # Copy only verified missing files; never rewrite inputs or cloud records.
        for name, sha in manifest.items():
            dest = ROOT/name
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(received/name,dest)
            assert digest(dest)==sha
        archive_dest=ROOT/'artifacts'/args.archive.name
        archive_dest.parent.mkdir(exist_ok=True)
        if archive_dest.exists(): assert digest(archive_dest)==EXPECTED_SHA
        else: shutil.copyfile(args.archive,archive_dest)
        verification={'status':'VERIFIED_COMPLETE_200_FIRST_ATTEMPTS','verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'scope':'Fixed teammate 100-question subset; two OVR models; not full MathVision reproduction',
            'archive_sha256':EXPECTED_SHA,'archive_bytes':args.archive.stat().st_size,'member_sha_verified':len(manifest),
            'images_readable':100,'same_order_and_ids':True,'models':evidence,'overall':overall,'transitions':dict(transitions),
            'development_overlap_ids':sorted(DEV_IDS,key=int),'upstream_evaluator_commit':read_json(ROOT/'configs/protocol.json')['upstream_mathvision_commit']}
        write_json(out/'mac_verification.json',verification)
        cases=[]
        for transition,category in [('0->1','RL improvement'),('1->0','RL regression'),('0->0','Both OVR incorrect')]:
            chosen=[q for q in ids if f"{int(data['coldstart'][q]['correct'])}->{int(data['rl'][q]['correct'])}"==transition][:2]
            for q in chosen:
                cases.append({'question_id':q,'category':category,'development_overlap':q in DEV_IDS,'human_review_status':'NOT_REVIEWED'})
        write_csv(out/'case_candidates.csv',cases)
        payload=[]
        questions=unique(inputs)
        for case in cases:
            qid=case['question_id'];q=questions[qid]
            options=q['options']
            if options==[chr(65+i) for i in range(len(options))]: options=[]
            payload.append({**case,'question':re.sub(r'<image\d+>','',q['question']).strip(),'options':options,'answer':refs[qid]['answer'],
                'level':q['level'],'subject':q['subject'],'image':'data:image/jpeg;base64,'+base64.b64encode((shared/q['image']).read_bytes()).decode(),
                'models':{m:{k:data[m][qid][k] for k in ['response','correct','output_tokens','finish_reason']} for m in data}})
        demo={'status':'REAL SHARED100 OUTPUTS · 6 rule-selected candidates · HUMAN REVIEW PENDING · ColdStart 34/100; RL 31/100', 'cases':payload,'models':['coldstart','rl']}
        encoded=json.dumps(demo,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
        html=(ROOT/'demo/template.html').read_text().replace('/*PAYLOAD*/null',encoded).replace('repeat(3,1fr)','repeat(2,1fr)')
        assert 'fetch(' not in html and '<script src=' not in html
        (ROOT/'artifacts/OVR_SHARED100_CASES.html').write_text(html)
        print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
