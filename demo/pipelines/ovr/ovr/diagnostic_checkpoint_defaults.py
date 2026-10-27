"""Two fixed RL control generations using the checkpoint's decoding defaults."""
from pathlib import Path
import copy,json
from .common import ROOT,read_json,write_json,digest
from .cloud import execute,inputs


def configuration(base):
    c=copy.deepcopy(base)
    c.update(protocol_id='ovr-rl-checkpoint-defaults-control-v1',results_namespace='cloud_diag_tf_checkpoint_defaults_20261022',config_source='.build/checkpoint_defaults_protocol.json',purpose='Two-question RL decoding control; never a formal evaluation',diagnostic_only=True,diagnostic_max_generations=2)
    c['generation']={'do_sample':True,'num_beams':1,'repetition_penalty':1.0,'temperature':0.6,'top_p':0.95,'top_k':40}
    return c


def main():
    from .cloud_backend import checkpoint,gpu_check,TransformersBackend
    from .scoring import score
    config=configuration(read_json(ROOT/'configs/cloud_shared100.json'))
    rows,_=inputs(config);assert config['canary_questions']==2 and [r['question_id'] for r in rows[:2]]==['40','51']
    path=ROOT/config['config_source']
    if path.exists():assert read_json(path)==config
    else:write_json(path,config)
    hardware=gpu_check(config);load_dir,manifest=checkpoint(config,'rl')
    defaults=read_json(load_dir/'generation_config.json')
    for k in ['do_sample','repetition_penalty','temperature','top_p','top_k']:assert defaults[k]==config['generation'][k]
    identity={'hardware':hardware,'checkpoint':manifest,'code':{p.name:digest(p) for p in [ROOT/'ovr/cloud.py',ROOT/'ovr/cloud_backend.py',Path(__file__)]},'diagnostic_only':True}
    def factory():
        b=TransformersBackend(config,load_dir)
        for k,v in config['generation'].items():setattr(b.generation,k,v)
        b.generation.validate();b.metadata['effective_generation']=b.generation.to_dict();b.metadata['diagnostic_only']=True
        return b
    result=execute(config,'rl','canary',identity,factory,session_minutes=25)
    if result['state'] not in ['complete','already_complete']:raise RuntimeError('Control incomplete; inspect before resuming')
    ns=config['results_namespace']
    score(ROOT/f'results/raw/{ns}/canary/rl.jsonl',ROOT/config['shared_dir']/'references.jsonl',ROOT/f'results/scored/{ns}/canary/rl.jsonl')
    print(json.dumps({'state':result['state'],'diagnostic_only':True,'formal_pilot_started':False},indent=2))


if __name__=='__main__':main()
