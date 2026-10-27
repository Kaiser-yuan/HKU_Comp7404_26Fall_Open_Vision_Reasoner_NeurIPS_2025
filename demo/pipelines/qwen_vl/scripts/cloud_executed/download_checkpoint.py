"""Download an exact resolved model revision and record provenance. Never saves tokens."""
import argparse,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
p=argparse.ArgumentParser()
p.add_argument('--repo',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
revision=HfApi().model_info(a.repo).sha
print('Repository:',a.repo,'Revision:',revision,flush=True)
snapshot_download(repo_id=a.repo,revision=revision,local_dir=a.output,max_workers=1)
(a.output/'download_metadata.json').write_text(json.dumps({'repo_id':a.repo,'revision':revision},indent=2)+'\n')
print('Download completed:',a.output,flush=True)
