"""Combine disjoint pilot runs only when inference settings match."""
import argparse,csv,hashlib,json
from pathlib import Path

def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 qs=lines(a.dataset/'questions.jsonl'); ids=[str(x['id']) for x in qs]
 predictions={};scores={};configs=[];sources=[]
 for run in a.runs:
  config=json.loads((run/'run_config.json').read_text());configs.append(config)
  sources.append({'path':str(run),'predictions_sha256':hashlib.sha256((run/'predictions.jsonl').read_bytes()).hexdigest(),'run_config':config})
  current={str(r['id']):r for r in lines(run/'predictions.jsonl') if r['status']=='ok'}
  assert not predictions.keys() & current.keys(),'Overlapping question IDs'
  predictions.update(current)
  with (run/'scores.csv').open(encoding='utf-8-sig') as f:
   for row in csv.DictReader(f):scores[str(row['id'])]=row
 canonical=[{k:v for k,v in c.items() if k!='dataset_sha256'} for c in configs]
 assert all(c==canonical[0] for c in canonical),'Inference configurations differ'
 assert set(ids)==set(predictions)==set(scores),'Missing or unexpected IDs'
 digest=hashlib.sha256((a.dataset/'questions.jsonl').read_bytes())
 for q in qs:digest.update((a.dataset/q['image']).read_bytes())
 a.output.mkdir(parents=True,exist_ok=True)
 if any(a.output.iterdir()):raise RuntimeError('Output must be empty')
 config=dict(configs[0]);config['dataset_sha256']=digest.hexdigest()
 (a.output/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
 (a.output/'predictions.jsonl').write_text(''.join(json.dumps(predictions[i],ensure_ascii=False)+'\n' for i in ids))
 with (a.output/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(scores[ids[0]]));w.writeheader();w.writerows(scores[i] for i in ids)
 summary={'total':len(ids),'successful':len(ids),'errors':0,'conservative_matches':sum(scores[i]['conservative_match']=='True' for i in ids),'truncated':sum(bool(predictions[i]['truncated']) for i in ids),'extraction_failures':sum(predictions[i]['extracted_answer'] is None for i in ids),'warning':'Merged disjoint pilot runs; not official scoring.'}
 summary['conservative_accuracy']=summary['conservative_matches']/len(ids)
 (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 (a.output/'merge_provenance.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
