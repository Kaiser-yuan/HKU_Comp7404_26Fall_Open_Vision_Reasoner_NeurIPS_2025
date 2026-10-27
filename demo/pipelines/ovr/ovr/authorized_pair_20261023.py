"""User-authorized 200 first attempts after the existing preflight, no new tests."""
from pathlib import Path
import datetime as dt
import json,os,time,traceback,signal
from ovr.common import ROOT,read_json,read_jsonl,write_json,digest
from ovr.vllm_evaluation import run,preflight_ready,CONFIG
from ovr.cloud import raw_path,export,report
DEADLINE=dt.datetime.fromisoformat('2026-10-23T11:20:08+00:00')
STATE=ROOT/'.build/formal-pair-controller-state.json'
if STATE.exists():raise RuntimeError('This authorized pair controller already has a state file; inspect before any repeat launch')
state={'state':'waiting_for_existing_preflight','authorized_utc':'2026-10-23T01:20:08+00:00','deadline_utc':DEADLINE.isoformat(),'gpu_hourly_price_cny':2.78,'max_task_runtime_hours_from_authorization':10,'model_order':['coldstart','rl'],'planned_questions_per_model':100,'no_more_diagnostics':True,'instance_power_or_billing_changed':False,'pid':os.getpid()}
write_json(STATE,state)
try:
 evidence=ROOT/'results/reports/cloud_vllm_preflight8192_probability_review_v1/complete.json'
 preflight_deadline=dt.datetime.fromisoformat('2026-10-23T01:46:30+00:00')
 preflight_pid=read_json(ROOT/'.build/probability-continuation-controller.json')['pid']
 while not evidence.exists():
  proc=Path('/proc')/str(preflight_pid)/'stat'
  if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Existing preflight stopped before completion; inspect saved records without starting new diagnostics')
  if dt.datetime.now(dt.timezone.utc)>preflight_deadline:raise TimeoutError('Existing preflight did not finish within its original bounded window')
  time.sleep(10)
 config=read_json(CONFIG);preflight_ready(config)
 state.update(state='formal_running',formal_started_utc=dt.datetime.now(dt.timezone.utc).isoformat());write_json(STATE,state)
 for model in ['coldstart','rl']:
  state.update(current_model=model);write_json(STATE,state)
  while len(read_jsonl(raw_path(ROOT,'pilot',model,config)))<100:
   remaining=(DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds()/60
   if remaining<10:raise TimeoutError('User-visible ten-hour task limit reached; no automatic extension')
   result=run(model,min(300,remaining))
   state.setdefault('model_results',{})[model]={k:result[k] for k in ['state','completed','model_server_stopped']};write_json(STATE,state)
   if result['state'] not in ['complete','already_complete','session_time_stop']:raise RuntimeError('Unexpected model run state; inspect instead of retrying')
   if result['state'] in ['complete','already_complete']:break
 state['coverage']=report(config,'pilot')
 if not state['coverage']['complete']:raise RuntimeError('Formal paired coverage incomplete')
 state['archive']=export(config)
 state.update(state='complete',all_model_servers_stopped=True,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),formal_generations=200)
except BaseException as exc:
 state.update(state='stopped_error_or_limit',error_type=type(exc).__name__,error=str(exc)[:350],ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
 traceback.print_exc()
finally:
 write_json(STATE,state)
 print(json.dumps(state,indent=2),flush=True)
