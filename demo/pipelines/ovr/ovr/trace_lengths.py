"""CPU-only descriptive trace-length analysis. No model inference or training."""
from pathlib import Path
import json,hashlib,statistics,random,csv,argparse,math
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
NS='teacher_student_lengths_v1'
TOKENIZER_SHA='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_lines(path,rows):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def extract(text,implicit_open=False,capped=False):
 opens=text.count('<think>');closes=text.count('</think>')
 if opens==closes==1 and text.index('<think>')<text.index('</think>'):
  a=text.index('<think>')+len('<think>');b=text.index('</think>');return text[a:b],text[b+len('</think>'):],'explicit_closed',False
 if implicit_open and opens==0 and closes==1:
  b=text.index('</think>');return text[:b],text[b+len('</think>'):],'prefilled_closed',False
 if implicit_open and opens==closes==0 and capped:return text,'','prefilled_right_censored',True
 if opens==1 and closes==0 and capped:return text.split('<think>',1)[1],'','explicit_right_censored',True
 return None,None,'unsegmented_or_malformed',None

def measure(text,tok,implicit_open=False,capped=False):
 reasoning,answer,boundary,censored=extract(text,implicit_open,capped)
 count=lambda s:len(tok.encode(s,add_special_tokens=False).ids)
 return {'response_text_tokens':count(text),'reasoning_tokens_observed':None if reasoning is None else count(reasoning),'final_answer_tokens':None if answer is None else count(answer),'boundary':boundary,'reasoning_right_censored':censored,'response_sha256':hashlib.sha256(text.encode()).hexdigest()}

def describe(values):
 if not values:return {'n':0,'mean':None,'median':None,'min':None,'max':None,'p25':None,'p75':None}
 v=sorted(values)
 def q(p):
  i=(len(v)-1)*p;lo=math.floor(i);hi=math.ceil(i);return v[lo]+(v[hi]-v[lo])*(i-lo)
 return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'min':v[0],'max':v[-1],'p25':q(.25),'p75':q(.75)}

def summary(name,rows):
 return {'group':name,'sample_n':len(rows),'response_text':describe([x['response_text_tokens'] for x in rows]),'reasoning_observed':describe([x['reasoning_tokens_observed'] for x in rows if x['reasoning_tokens_observed'] is not None]),'reasoning_completed':describe([x['reasoning_tokens_observed'] for x in rows if x['reasoning_right_censored'] is False]),'capped_n':None if name=='teacher' else sum(bool(x['capped']) for x in rows),'reasoning_right_censored_n':sum(x['reasoning_right_censored'] is True for x in rows),'unsegmented_n':sum(x['reasoning_tokens_observed'] is None for x in rows)}

def students(root,tok):
 out=[];source_ns='cloud_vllm_budget20_prefix_v1';ids=read(root/'workstreams/budget20/selection.json')['question_ids']
 for m in ['coldstart','rl']:
  path=root/'results/raw'/source_ns/(m+'.jsonl');source_sha=sha(path);raw=lines(path)
  assert [x['question_id'] for x in raw]==ids and len(raw)==20
  checkpoint=read(root/'results/runs'/source_ns/m/'checkpoint.json')
  assert next(x['sha256'] for x in checkpoint['files'] if x['name']=='tokenizer.json')==TOKENIZER_SHA
  scored={x['question_id']:x for x in lines(root/'results/scored'/source_ns/m/'16384-scored.jsonl')}
  for r in raw:
   assert r['prompt_token_ids_verified'] and r['run_fingerprint']=='37fbd6443419566fe02ed67b2fed8277c015e2d82e9fb3b67a2d45fcbaba5229'
   capped=r['finish_reason']=='length';out.append({'group':m,'question_id':r['question_id'],'source_sha256':source_sha,'generated_tokens':r['output_tokens'],'finish_reason':r['finish_reason'],'capped':capped,'correct':scored[r['question_id']]['correct'],**measure(r['response'],tok,True,capped)})
  assert sha(path)==source_sha
 return out

def collect_teacher(root,protocol,tok):
 import pyarrow.parquet as pq
 api=read(root/'workstreams/teacher_lengths/dataset_api.json');assert api['sha']==protocol['revision'];chosen=protocol['files'];out=[];sample_raw=[]
 for spec in chosen:
  name=spec['file'];path=root/'data/cache/teacher_length_v1'/name;entry=next(x for x in api['siblings'] if x['rfilename']==name)
  assert path.stat().st_size==entry['size'] and sha(path)==entry['lfs']['sha256'],name
  selected=set(spec['sample_row_indices']);offset=0;count=0
  pf=pq.ParquetFile(path,memory_map=True);assert pf.metadata.num_rows==spec['population_rows']
  for batch in pf.iter_batches(batch_size=64,columns=['question','answer']):
   wanted=[i-offset for i in selected if offset<=i<offset+batch.num_rows]
   if wanted:
    items=batch.to_pylist()
    for j in sorted(wanted):
     item=items[j];text=item['answer'];assert isinstance(text,str) and text.strip()
     sample_raw.append({'file':name,'row_index':offset+j,'global_row_index':spec['global_offset']+offset+j,'dataset_revision':api['sha'],'question':item['question'],'answer':text})
     out.append({'group':'teacher','teacher_dataset_revision':api['sha'],'file':name,'row_index':offset+j,'global_row_index':spec['global_offset']+offset+j,'capped':None,'teacher_stop_status':'not_provided_by_dataset','question_sha256':hashlib.sha256(item['question'].encode()).hexdigest(),'teacher_file_sha256':entry['lfs']['sha256'],**measure(text,tok)})
     count+=1
   offset+=batch.num_rows
  assert offset==spec['population_rows'] and count==len(selected)
  print(json.dumps({'file':name,'sample_n':count,'state':'sample_measured'}),flush=True)
 assert len(out)==protocol['sample_size']
 write_lines(root/'results/raw'/NS/'teacher_sample.jsonl',sorted(sample_raw,key=lambda x:x['global_row_index']))
 return sorted(out,key=lambda x:x['global_row_index'])

def measure_saved_teacher_sample(root,protocol,tok):
 api=read(root/'workstreams/teacher_lengths/dataset_api.json');assert api['sha']==protocol['revision']
 raw=lines(root/'results/raw'/NS/'teacher_sample.jsonl')
 hashes={x['rfilename']:x['lfs']['sha256'] for x in api['siblings'] if x['rfilename'].endswith('.parquet')}
 result=[]
 for item in raw:
  assert item['dataset_revision']==protocol['revision']
  text=item['answer'];assert isinstance(text,str) and text.strip()
  result.append({'group':'teacher','teacher_dataset_revision':api['sha'],'file':item['file'],'row_index':item['row_index'],'global_row_index':item['global_row_index'],'capped':None,'teacher_stop_status':'not_provided_by_dataset','question_sha256':hashlib.sha256(item['question'].encode()).hexdigest(),'teacher_file_sha256':hashes[item['file']],**measure(text,tok)})
 validate_teacher(result,protocol,api)
 return result


def validate_teacher(rows,protocol,api):
 expected={spec['global_offset']+i:(spec['file'],i) for spec in protocol['files'] for i in spec['sample_row_indices']}
 assert api['sha']==protocol['revision']
 assert len(rows)==len(expected)==protocol['sample_size']
 assert [x['global_row_index'] for x in rows]==sorted(expected), 'Teacher sample IDs differ or repeat'
 hashes={x['rfilename']:x['lfs']['sha256'] for x in api['siblings'] if x['rfilename'].endswith('.parquet')}
 for row in rows:
  assert (row['file'],row['row_index'])==expected[row['global_row_index']]
  assert row['teacher_dataset_revision']==protocol['revision'] and row['teacher_file_sha256']==hashes[row['file']]
  assert row['response_text_tokens']>0 and len(row['question_sha256'])==len(row['response_sha256'])==64
 return True


def analyze(root=ROOT,teacher=False,from_sample=False):
 tokfile=root/'.build/contract-assets/tokenizer.json';assert sha(tokfile)==TOKENIZER_SHA;tok=Tokenizer.from_file(str(tokfile));p=read(root/'workstreams/teacher_lengths/protocol.json')
 out=root/'results/reports'/NS;rows=students(root,tok);write_lines(out/'student_lengths.jsonl',rows)
 teacher_path=out/'teacher_lengths.jsonl'
 assert not (teacher and from_sample)
 if teacher:write_lines(teacher_path,collect_teacher(root,p,tok))
 if from_sample:write_lines(teacher_path,measure_saved_teacher_sample(root,p,tok))
 teachers=lines(teacher_path) if teacher_path.exists() else []
 if teachers:validate_teacher(teachers,p,read(root/'workstreams/teacher_lengths/dataset_api.json'))
 groups=[summary(m,[x for x in rows if x['group']==m]) for m in ['coldstart','rl']]
 if teachers:groups.insert(0,summary('teacher',teachers))
 a={x['question_id']:x for x in rows if x['group']=='coldstart'};b={x['question_id']:x for x in rows if x['group']=='rl'}
 shared=[q for q in a if a[q]['reasoning_right_censored'] is False and b[q]['reasoning_right_censored'] is False]
 paired={'both_completed_n':len(shared),'question_ids':shared,'coldstart_reasoning':describe([a[q]['reasoning_tokens_observed'] for q in shared]),'rl_reasoning':describe([b[q]['reasoning_tokens_observed'] for q in shared]),'paired_reasoning_delta_rl_minus_coldstart':describe([b[q]['reasoning_tokens_observed']-a[q]['reasoning_tokens_observed'] for q in shared]),'limitation':'Completion-selected subset; excludes four RL-censored questions, not an unbiased replacement for all 20.'}
 result={'complete':len(teachers)==p['sample_size'],'teacher_expected_n':p['sample_size'],'teacher_observed_n':len(teachers),'teacher_unique_question_n':len({x['question_sha256'] for x in teachers}),'teacher_unique_response_n':len({x['response_sha256'] for x in teachers}),'tokenizer_sha256':TOKENIZER_SHA,'counting':'Canonical re-tokenization of decoded text with the same Qwen tokenizer; add_special_tokens=False; reasoning excludes think delimiters and final answer. Student opening think delimiter was supplied by the official template. Observed censored lengths are lower bounds. Teacher unsegmented answers retain response length but have reasoning length=null.','groups':groups,'paired_completed':paired,'scope':p['scope'],'teacher_training_coverage':'Published revision only; not assumed identical to all original training examples; task and modality differ from MathVision.','limitations':['Longer text is not evidence of stronger reasoning, behavior transfer or actual visual attention.','Teacher end-of-generation status is unavailable; absence of a length finish reason does not establish uncensored teacher generation.','No resampling, new generation, training, GPU, or paid model judge.']}
 write(out/'comparison.json',result)
 csvrows=[]
 for g in groups:
  csvrows.append({'group':g['group'],'n':g['sample_n'],'mean_response_text_tokens':g['response_text']['mean'],'median_response_text_tokens':g['response_text']['median'],'segmented_reasoning_n':g['reasoning_observed']['n'],'mean_observed_reasoning_tokens':g['reasoning_observed']['mean'],'median_observed_reasoning_tokens':g['reasoning_observed']['median'],'student_length_limit_n':g['capped_n'] if g['group']!='teacher' else None,'reasoning_right_censored_n':g['reasoning_right_censored_n'],'unsegmented_n':g['unsegmented_n']})
 with (out/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group();g.add_argument('--teacher',action='store_true');g.add_argument('--from-sample',action='store_true');args=ap.parse_args();analyze(teacher=args.teacher,from_sample=args.from_sample)
