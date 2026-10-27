"""Download public teacher data without secrets, model calls or cloud actions."""
from pathlib import Path
import os,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 api=json.loads((ROOT/'workstreams/teacher_lengths/dataset_api.json').read_text())
 dest=ROOT/'data/cache/teacher_length_v1';dest.mkdir(parents=True,exist_ok=True)
 os.environ['HF_XET_CACHE']=str(dest/'.xet');os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS']='4'
 from huggingface_hub import hf_hub_download
 for item in api['siblings']:
  name=item['rfilename']
  if not name.endswith('.parquet'):continue
  print(json.dumps({'file':name,'state':'starting'}),flush=True)
  try:
   p=Path(hf_hub_download(repo_id=api['id'],repo_type='dataset',filename=name,revision=api['sha'],local_dir=dest,token=False))
   with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
   assert p.stat().st_size==item['size'] and h==item['lfs']['sha256']
  except Exception as exc:
   # Errors can contain signed URLs; never echo exception text or traceback.
   print(json.dumps({'file':name,'state':'failed','error_type':type(exc).__name__,'retry_automatically':False}),flush=True)
   return 2
  print(json.dumps({'file':name,'state':'verified','sha256':h}),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
