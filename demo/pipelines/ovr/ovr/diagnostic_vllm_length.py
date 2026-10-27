"""One RL question-51 8192-token length diagnostic; no formal evaluation."""
import base64,datetime,hashlib,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from .common import ROOT,read_json,write_json,digest
from .cloud import inputs,append
from .cloud_backend import checkpoint,gpu_check

OUT=ROOT/'results/runs/vllm_length_audit'
RAW=ROOT/'results/raw/vllm_length_audit'
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
    rows,_=inputs(config);rows=[r for r in rows if r['question_id']=='51'];assert len(rows)==1
    s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
    name=f'{model}-'+('processor-template' if override else 'official-default-template')
    command=[PYTHON,'-m','vllm.entrypoints.openai.api_server','--model',str(load),'--served-model-name','ovr-'+model,'--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1','--gpu-memory-utilization','0.6','--dtype','bfloat16','--max-model-len','12288','--max-num-seqs','1','--max-num-batched-tokens','12288','--generation-config','auto','--seed','42','--enforce-eager','--limit-mm-per-prompt',json.dumps({'image':1,'video':0}),'--mm-processor-kwargs',json.dumps({**config['image_processor'],'use_fast':True}),'--disable-log-requests']
    if override:
        from transformers import AutoProcessor
        proc=AutoProcessor.from_pretrained(load,local_files_only=True,use_fast=True)
        template=ROOT/'.build/ovr-processor-template.jinja';template.write_text(proc.chat_template)
        command += ['--chat-template',str(template),'--chat-template-content-format','openai']
    session_path=OUT/(name+'-'+str(time.time_ns())+'.json')
    session={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':model,'conditions':conditions,'processor_template_override':override,'command':command,'checkpoint':manifest,'diagnostic_only':True,'max_http_generations':len(conditions),'source_sha256':digest(Path(__file__))}
    write_json(session_path,session)
    log=Path('/root/ovr-private-logs')/('vllm-length-server-'+name+'.log')
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
                body={'model':'ovr-'+model,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':data_url}},{'type':'text','text':item['prompt']}]}],'max_tokens':8192,'seed':42,'return_token_ids':True}
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
    if list(RAW.glob('*.jsonl')) or (OUT/'request_events.jsonl').exists():raise RuntimeError('Existing length-probe requests; no automatic rerun')
    base=read_json(ROOT/'results/runs/vllm_deployment_audit/complete.json')
    assert base['completed_http_generations']==6 and base['all_servers_stopped']
    baseline=[json.loads(x) for x in (ROOT/'results/raw/vllm_deployment_audit/rl_defaults_official.jsonl').read_text().splitlines()]
    baseline=next(x for x in baseline if x['question_id']=='51');assert baseline['finish_reason']=='length' and baseline['output_tokens']==4096
    def budget_timeout(*_):raise TimeoutError('Ten-minute length diagnostic budget reached')
    signal.signal(signal.SIGALRM,budget_timeout);signal.alarm(600)
    config=read_json(ROOT/'configs/cloud_shared100.json');hardware=gpu_check(config)
    plan={'scope':'One independent RL question-51 length diagnostic; not an answer-dependent replacement or formal result','reason':'Default vLLM sampling stopped at the 4096 output-token budget','changes':{'max_new_tokens':[4096,8192],'server_max_model_len':[8192,12288],'new_server_cold_cache':True},'decoding':'Same checkpoint defaults, seed42 per request','question_ids':['51'],'max_http_generations':1,'formal_pilot_started':False,'source_sha256':digest(Path(__file__)),'baseline_raw_sha256':digest(ROOT/'results/raw/vllm_deployment_audit/rl_defaults_official.jsonl'),'hardware':hardware}
    write_json(OUT/'plan.json',plan)
    rows=serve_cases(config,'rl',['defaults']);assert len(rows)==1
    a=baseline['api_response']['choices'][0]['token_ids'];b=rows[0]['api_response']['choices'][0]['token_ids'];n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    signal.alarm(0)
    write_json(OUT/'complete.json',{'completed_http_generations':1,'formal_pilot_started':False,'all_servers_stopped':True,'matching_prefix_tokens':n,'entire_4096_token_prefix_matches':b[:len(a)]==a,'same_prompt_token_ids':rows[0]['actual_prompt_token_ids_sha256']==baseline['actual_prompt_token_ids_sha256'],'output_tokens':rows[0]['output_tokens'],'finish_reason':rows[0]['finish_reason'],'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__':main()
