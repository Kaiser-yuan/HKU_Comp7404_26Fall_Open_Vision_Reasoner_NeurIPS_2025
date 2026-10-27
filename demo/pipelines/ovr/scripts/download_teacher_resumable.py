"""Pinned public dataset downloader; bounded memory, resumable Range chunks, whole-file SHA."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,subprocess,time,threading
ROOT=Path(__file__).resolve().parents[1]
REV='e377de8ec93f9d1ff48744dcdc8c51c4801c6da2'

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--chunk-mib',type=int,default=8);args=ap.parse_args()
 assert 1<=args.workers<=12 and 1<=args.chunk_mib<=64
 api=json.loads((ROOT/'workstreams/teacher_lengths/dataset_api.json').read_text());assert api['sha']==REV
 base=ROOT/'data/cache/teacher_length_v1';base.mkdir(parents=True,exist_ok=True);tmpbase=base/'.range-v2';tmpbase.mkdir(exist_ok=True)
 run=ROOT/'results/runs/teacher_download_v2';run.mkdir(parents=True,exist_ok=True)
 import fcntl
 lockfile=(run/'controller.lock').open('a');fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 state_path=run/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'revision':REV,'chunk_bytes':args.chunk_mib*2**20,'done':{},'verified_files':{},'failed':[]}
 assert state['revision']==REV and state['chunk_bytes']==args.chunk_mib*2**20
 chunk=state['chunk_bytes'];mu=threading.Lock();started=time.monotonic();files=[x for x in api['siblings'] if x['rfilename'].endswith('.parquet')];tasks=[];fds={}
 state['state']='running';state['pid']=os.getpid();state['started_or_resumed_unix']=time.time();state['failed']=[];atomic_json(state_path,state)
 for item in files:
  name=item['rfilename'];p=base/name
  if p.exists():
   assert p.stat().st_size==item['size'] and digest(p)==item['lfs']['sha256'];state['verified_files'][name]=item['lfs']['sha256'];continue
  target=base/(name+'.range-download');fd=os.open(target,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,item['size']);fds[name]=fd
  for start in range(0,item['size'],chunk):
   end=min(start+chunk,item['size'])-1;key=f'{name}:{start}-{end}'
   if key in state['done']:
    data=os.pread(fd,end-start+1,start)
    if hashlib.sha256(data).hexdigest()==state['done'][key]['sha256']:continue
    del state['done'][key]
   tasks.append((item,start,end,key))
 atomic_json(state_path,state)
 total=sum(x['size'] for x in files)
 bytes_at_resume=sum(x['bytes'] for x in state['done'].values())
 def worker(task):
  item,start,end,key=task;name=item['rfilename'];size=end-start+1;part=tmpbase/(key.replace(':','_')+'.part');headers=part.with_suffix('.headers');status=part.with_suffix('.status')
  for attempt in range(1,11):
   host='huggingface.co' if attempt%2 else 'hf-mirror.com'
   url=f'https://{host}/datasets/Kangheng/OVR-Dataset-ColdStart/resolve/{REV}/{name}?download=true'
   cmd=['/usr/bin/curl','-L','--fail','--silent','--connect-timeout','20','--max-time','180','--range',f'{start}-{end}','--dump-header',str(headers),'--output',str(part),'--write-out','%{http_code}',url]
   try:
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=190)
    h=headers.read_text(errors='replace') if headers.exists() else ''
    ranges=[line.split(':',1)[1].strip() for line in h.splitlines() if line.lower().startswith('content-range:')]
    ok=proc.returncode==0 and proc.stdout.strip()=='206' and part.exists() and part.stat().st_size==size and ranges and ranges[-1]==f'bytes {start}-{end}/{item["size"]}'
    if ok:
     data=part.read_bytes();sha=hashlib.sha256(data).hexdigest();written=0
     while written<size:written+=os.pwrite(fds[name],data[written:],start+written)
     os.fsync(fds[name])
     with mu:
      state['done'][key]={'sha256':sha,'bytes':size,'host':host,'attempt':attempt};state['downloaded_verified_range_bytes']=sum(x['bytes'] for x in state['done'].values());state['updated_unix']=time.time();atomic_json(state_path,state)
     part.unlink(missing_ok=True);headers.unlink(missing_ok=True)
     return True
    code=proc.returncode
   except (OSError,subprocess.SubprocessError):code='transport_error'
   part.unlink(missing_ok=True);headers.unlink(missing_ok=True)
   if attempt<10:time.sleep(min(2**attempt,30))
  with mu:state['failed'].append({'chunk':key,'error_code':code,'attempts':10});atomic_json(state_path,state)
  return False
 try:
  print(json.dumps({'state':'running','chunks_pending':len(tasks),'workers':args.workers,'total_bytes':total}),flush=True)
  last=time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
   futures=[ex.submit(worker,t) for t in tasks]
   for f in concurrent.futures.as_completed(futures):
    f.result()
    if time.monotonic()-last>=20:
     with mu:
      n=sum(x['bytes'] for x in state['done'].values());elapsed=time.monotonic()-started
      print(json.dumps({'state':'downloading','range_bytes':n,'total_bytes':total,'completed_chunks':len(state['done']),'failed_chunks':len(state['failed']),'elapsed_s':round(elapsed),'MB_s_since_resume':round((n-bytes_at_resume)/elapsed/1e6,2)}),flush=True)
     last=time.monotonic()
  for fd in fds.values():os.close(fd)
  fds={}
  for item in files:
   name=item['rfilename'];p=base/name
   if p.exists():continue
   keys=[f'{name}:{s}-{min(s+chunk,item["size"])-1}' for s in range(0,item['size'],chunk)]
   if not all(k in state['done'] for k in keys):continue
   partial=base/(name+'.range-download');actual=digest(partial)
   if actual!=item['lfs']['sha256']:raise ValueError('Whole-file SHA mismatch: '+name)
   partial.replace(p);state['verified_files'][name]=actual;atomic_json(state_path,state);print(json.dumps({'state':'file_verified','file':name,'sha256':actual}),flush=True)
  state['state']='complete_verified' if len(state['verified_files'])==len(files) else 'incomplete_ranges_retained'
  state['ended_unix']=time.time();atomic_json(state_path,state);print(json.dumps({'state':state['state'],'verified_files':list(state['verified_files'])}),flush=True)
  return 0 if state['state']=='complete_verified' else 2
 except BaseException as exc:
  state['state']='stopped_ranges_retained';state['error_type']=type(exc).__name__;atomic_json(state_path,state);raise
 finally:
  for fd in fds.values():os.close(fd)
if __name__=='__main__':raise SystemExit(main())
