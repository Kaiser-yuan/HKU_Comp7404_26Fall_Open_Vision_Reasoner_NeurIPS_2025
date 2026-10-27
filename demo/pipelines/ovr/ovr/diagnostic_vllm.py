"""Bounded official vLLM serving audit: at most six HTTP generations, no pilot."""
import base64,datetime,hashlib,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from .common import ROOT,read_json,write_json,digest
from .cloud import inputs,append
from .cloud_backend import checkpoint,gpu_check

OUT=ROOT/'results/runs/vllm_deployment_audit'
RAW=ROOT/'results/raw/vllm_deployment_audit'
PYTHON='/root/ovr-course-env/vllm011/bin/python'

def sha_json(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()

def http(port,path,body=None,timeout=30):
    data=None if body is None else json.dumps(body).encode()
    request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        content=response.read()
        return json.loads(content) if content else {}

def expected_input(config,model,item,load):
    from transformers import AutoProcessor
    from PIL import Image
    proc=AutoProcessor.from_pretrained(load,local_files_only=True,trust_remote_code=False,use_fast=True,**config['image_processor'])
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':item['prompt']}]}]
    text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    with Image.open(ROOT/config['shared_dir']/item['image']) as im:
        batch=proc(text=[text],images=[im.convert('RGB')],return_tensors='pt',padding=True)
    ids=batch['input_ids'][0].tolist()
    return {'formatted_prompt':text,'prompt_token_ids_sha256':sha_json(ids),'input_tokens':len(ids),'image_grid_thw':batch['image_grid_thw'].tolist(),'processor_class':type(proc.image_processor).__name__}

def serve_cases(config,model,conditions,override=False):
    from PIL import Image
    load,manifest=checkpoint(config,model)
    rows,_=inputs(config);rows=rows[:2]
    s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
    name=f'{model}-'+('processor-template' if override else 'official-default-template')
    command=[PYTHON,'-m','vllm.entrypoints.openai.api_server','--model',str(load),'--served-model-name','ovr-'+model,'--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1','--gpu-memory-utilization','0.6','--dtype','bfloat16','--max-model-len','8192','--max-num-seqs','1','--max-num-batched-tokens','8192','--generation-config','auto','--seed','42','--enforce-eager','--limit-mm-per-prompt',json.dumps({'image':1,'video':0}),'--mm-processor-kwargs',json.dumps({**config['image_processor'],'use_fast':True}),'--disable-log-requests']
    if override:
        from transformers import AutoProcessor
        proc=AutoProcessor.from_pretrained(load,local_files_only=True,use_fast=True)
        template=ROOT/'.build/ovr-processor-template.jinja';template.write_text(proc.chat_template)
        command += ['--chat-template',str(template),'--chat-template-content-format','openai']
    session_path=OUT/(name+'-'+str(time.time_ns())+'.json')
    session={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':model,'conditions':conditions,'processor_template_override':override,'command':command,'checkpoint':manifest,'diagnostic_only':True,'max_http_generations':len(conditions)*2,'source_sha256':digest(Path(__file__))}
    write_json(session_path,session)
    log=Path('/root/ovr-private-logs')/('vllm-server-'+name+'.log')
    env={**os.environ,'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    with log.open('a') as f:
        server=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=f,env=env,start_new_session=True)
    session['server_pid']=server.pid;write_json(session_path,session)
    collected=[]
    try:
        deadline=time.monotonic()+360
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError(f'vLLM server exited with {server.returncode}; inspect private log')
            try:
                http(port,'/health',timeout=2);break
            except Exception:time.sleep(2)
        else:raise TimeoutError('Bounded vLLM startup timeout')
        session['served_models']=http(port,'/v1/models')
        assert str(load) in {m.get('root') for m in session['served_models']['data']}, 'Served checkpoint path differs from the requested pinned model'
        write_json(session_path,session)
        for condition in conditions:
            path=RAW/f'{model}_{condition}_{"processor" if override else "official"}.jsonl'
            if path.exists():raise RuntimeError('Audit output already exists; no automatic regeneration')
            for item in rows:
                image=ROOT/config['shared_dir']/item['image']
                data_url='data:image/jpeg;base64,'+base64.b64encode(image.read_bytes()).decode()
                body={'model':'ovr-'+model,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':data_url}},{'type':'text','text':item['prompt']}]}],'max_tokens':4096,'seed':42,'return_token_ids':True}
                if condition=='greedy':body.update(temperature=0.0,top_p=1.0,top_k=-1,repetition_penalty=1.0)
                expected=expected_input(config,model,item,load)
                # No generation is requested by /tokenize. Preserve its view as a
                # second input check, even if its token expansion differs.
                tokenized=http(port,'/tokenize',{'model':body['model'],'messages':body['messages'],'add_generation_prompt':True},timeout=45)
                started=time.monotonic()
                append(OUT/'request_events.jsonl',{'event':'submitted','model':model,'condition':condition,'question_id':item['question_id'],'processor_override':override,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                result=http(port,'/v1/chat/completions',body,timeout=600)
                append(OUT/'request_events.jsonl',{'event':'response_received','model':model,'condition':condition,'question_id':item['question_id'],'processor_override':override,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                choice=result['choices'][0];actual_ids=result.get('prompt_token_ids')
                actual_sha=sha_json(actual_ids) if actual_ids is not None else None
                row={'model':model,'question_id':item['question_id'],'condition':condition,'backend':'vllm-0.11.0','diagnostic_only':True,'processor_template_override':override,'response':choice['message'].get('content') or '', 'finish_reason':choice['finish_reason'],'stop_reason':choice.get('stop_reason'),'output_tokens':result['usage']['completion_tokens'],'input_tokens':result['usage']['prompt_tokens'],'latency_s':time.monotonic()-started,'image_sha256':item['image_sha256'],'expected_tf_input':expected,'actual_prompt_token_ids_sha256':actual_sha,'prompt_ids_match_tf':actual_sha==expected['prompt_token_ids_sha256'] if actual_sha else None,'tokenize_response':tokenized,'api_response':result,'request_decoding':{k:v for k,v in body.items() if k!='messages'},'source_sha256':digest(Path(__file__))}
                append(path,row);collected.append(row)
                print(json.dumps({'model':model,'condition':condition,'id':item['question_id'],'tokens':row['output_tokens'],'finish':row['finish_reason'],'prompt_ids_match_tf':row['prompt_ids_match_tf']}),flush=True)
        return collected
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
        session.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),server_exit_code=server.returncode,server_stopped=True,completed_responses=len(collected))
        write_json(session_path,session)
        time.sleep(3)

def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if list(RAW.glob('*.jsonl')) or (OUT/'request_events.jsonl').exists():raise RuntimeError('Existing vLLM audit requests or outputs; inspect before any further generations')
    def budget_timeout(*_):raise TimeoutError('Twenty-minute diagnostic wall-clock budget reached')
    signal.signal(signal.SIGALRM,budget_timeout);signal.alarm(1200)
    config=read_json(ROOT/'configs/cloud_shared100.json')
    hardware=gpu_check(config)
    runtime_code="import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['vllm','torch','torchvision','transformers','xformers']}))"
    runtime=json.loads(subprocess.check_output([PYTHON,'-c',runtime_code],text=True))
    assert runtime['vllm']=='0.11.0' and runtime['torch']=='2.8.0+cu128' and runtime['transformers']=='4.57.1'
    write_json(OUT/'runtime.json',{'hardware':hardware,'vllm_environment':runtime})
    plan={'scope':'At most six vLLM HTTP generations plus two separately recorded Transformers RL defaults controls. No 100-question run.','question_ids':['40','51'],'fixed_limits':{'max_new_tokens':4096,'max_context_tokens':8192,'seed':42},'stage1':'RL official server template, greedy and checkpoint defaults: four generations','stage2_rule':'If RL prompt token IDs differ from Transformers, run two RL defaults cases with processor template override; otherwise run two ColdStart defaults cases. Rule depends on input equality, not correctness.','source_sha256':digest(Path(__file__)),'formal_pilot_started':False}
    write_json(OUT/'plan.json',plan)
    rows=serve_cases(config,'rl',['greedy','defaults'])
    if any(r['prompt_ids_match_tf'] is not True for r in rows):
        rows+=serve_cases(config,'rl',['defaults'],override=True)
    else:rows+=serve_cases(config,'coldstart',['defaults'])
    assert len(rows)==6
    signal.alarm(0)
    write_json(OUT/'complete.json',{'completed_http_generations':len(rows),'formal_pilot_started':False,'all_servers_stopped':True,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__':main()
