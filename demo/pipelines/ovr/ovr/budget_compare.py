"""Frozen 20-question, shared-trajectory 8192/16384 comparison.

prepare/check/report/export are CPU-only. GPU work requires --run --minutes N.
One 16k streaming request per model/question yields two scored views; never
resample an uncertain attempt or transition automatically to 100 questions.
"""
import argparse
import base64
import contextlib
import datetime as dt
import importlib.metadata
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from .common import (ROOT, digest, fingerprint, read_json, read_jsonl, unique,
                     wilson, write_csv, write_json, write_jsonl)
from .cloud import append, inputs, run_lock

BASE_CONFIG = 'configs/cloud_vllm_shared1008192.json'
CONFIG = 'configs/cloud_vllm_budget20.json'
WORK = 'workstreams/budget20'
NS = 'cloud_vllm_budget20_prefix_v1'
MODELS = ('coldstart', 'rl')
DEV_IDS = ('40', '51', '465', '689', '1542', '2016', '2968')
PYTHON = '/root/ovr-course-env/vllm011/bin/python'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def choose(rows):
    """Use only IDs and difficulty, never previous answers or correctness."""
    rng = random.Random(20261025)
    selected = set()
    for level in range(1, 6):
        pool = sorted(str(r['question_id']) for r in rows
                      if int(r['level']) == level and str(r['question_id']) not in DEV_IDS)
        selected.update(rng.sample(pool, 4))
    return [str(r['question_id']) for r in rows if str(r['question_id']) in selected]


def make_config(base):
    keep = ['shared_dir', 'inputs_sha256', 'references_sha256', 'models', 'backend',
            'dtype', 'quantization', 'seed', 'image_processor', 'generation',
            'runtime', 'level_counts', 'image_processor_use_fast', 'prompt_version']
    config = {k: base[k] for k in keep}
    config.update(protocol_id='ovr-budget20-same-trajectory-8192-16384-v1',
                  status='PREPARED_NOT_RUN', results_namespace=NS,
                  expected_pilot_questions=100, max_image_side=None,
                  max_new_tokens=16384, max_context_tokens=24576,
                  per_question_max_s=900, batch_size=1,
                  budgets=[8192, 16384], selection_seed=20261025,
                  per_level=4, excluded_development_ids=list(DEV_IDS),
                  method='one_16k_request_two_token_prefix_views',
                  gpu_memory_utilization=0.6, server_startup_max_s=360,
                  source_protocol_sha256=None,
                  max_generation_requests=40,
                  scope='Post-hoc exploratory follow-up on an already evaluated subset; '
                        'not an independent holdout or original paper reproduction.')
    return config


def frozen_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen preparation differs; preserve existing file: ' + str(path))
    else:
        write_json(path, value)


def prepare(root=ROOT):
    base = read_json(root / BASE_CONFIG)
    rows, shared = inputs(base, root)
    ids = choose(rows)
    config = make_config(base)
    config['source_protocol_sha256'] = digest(root / BASE_CONFIG)
    selection = {'question_ids': ids, 'seed': config['selection_seed'],
                 'sampling': '4 per difficulty; sorted ID pools, original shared100 order',
                 'excluded_development_ids': list(DEV_IDS),
                 'selection_used_previous_outputs': False,
                 'source_inputs_sha256': digest(shared / 'inputs.jsonl'),
                 'items': [{k: r[k] for k in ('question_id', 'level', 'image_sha256')}
                           for r in rows if str(r['question_id']) in ids]}
    frozen_json(root / CONFIG, config)
    frozen_json(root / WORK / 'selection.json', selection)
    refs = [r for r in read_jsonl(shared / 'references.jsonl') if str(r['question_id']) in ids]
    target = root / WORK / 'references.jsonl'
    if target.exists():
        if read_jsonl(target) != refs:
            raise ValueError('Frozen selected references changed')
    else:
        write_jsonl(target, refs)
    return check(root)


def check(root=ROOT):
    base = read_json(root / BASE_CONFIG)
    config = read_json(root / CONFIG)
    expected = make_config(base)
    expected['source_protocol_sha256'] = digest(root / BASE_CONFIG)
    if config != expected:
        raise ValueError('Budget protocol differs from the prepared fixed comparison')
    rows, shared = inputs(config, root)
    selected = read_json(root / WORK / 'selection.json')
    ids = choose(rows)
    expected_selection = {
        'question_ids': ids, 'seed': config['selection_seed'],
        'sampling': '4 per difficulty; sorted ID pools, original shared100 order',
        'excluded_development_ids': list(DEV_IDS), 'selection_used_previous_outputs': False,
        'source_inputs_sha256': digest(shared / 'inputs.jsonl'),
        'items': [{k: r[k] for k in ('question_id', 'level', 'image_sha256')}
                  for r in rows if str(r['question_id']) in ids]}
    if selected != expected_selection or len(ids) != 20:
        raise ValueError('Selected IDs/metadata differ from outcome-independent sampling')
    refs = [r for r in read_jsonl(shared / 'references.jsonl') if str(r['question_id']) in ids]
    if read_jsonl(root / WORK / 'references.jsonl') != refs:
        raise ValueError('Selected references differ')
    return {'status': 'CPU_PREPARATION_VERIFIED_NOT_GPU_VALIDATED', 'question_ids': ids,
            'difficulty_counts': dict(Counter(str(r['level']) for r in refs)),
            'config_sha256': digest(root / CONFIG),
            'selection_sha256': digest(root / WORK / 'selection.json'),
            'generations_planned': 40, 'scored_views_planned': 80,
            'gpu_started_by_check': False,
            'comparison': 'New 8k prefixes versus 16k on the same trajectories; '
                          'original 100-question results are not replaced.'}


class Stream:
    """Assemble vLLM delta token IDs without counting characters as tokens."""
    def __init__(self):
        self.ids, self.text, self.prompt_ids = [], '', None
        self.finish, self.stop_reason, self.usage = None, None, None
        self.timings, self.done, self.request_id = [], False, None

    def add(self, payload, elapsed):
        if 'error' in payload:
            raise ValueError('Serving stream returned an error; preserve journal, do not retry')
        if self.request_id is None:
            self.request_id = payload.get('id')
        elif payload.get('id') and payload['id'] != self.request_id:
            raise ValueError('Stream request identity changed')
        if payload.get('prompt_token_ids') is not None:
            ids = payload['prompt_token_ids']
            if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
                raise ValueError('Invalid prompt token IDs')
            if self.prompt_ids is not None and self.prompt_ids != ids:
                raise ValueError('Prompt token IDs changed')
            self.prompt_ids = ids
        choices = payload.get('choices', [])
        if len(choices) > 1:
            raise ValueError('Only one sampled output is allowed')
        for c in choices:
            if c.get('index') != 0:
                raise ValueError('Unexpected choice index')
            delta = c.get('delta') or {}
            if delta.get('reasoning_content') or delta.get('tool_calls'):
                raise ValueError('Unexpected reasoning/tool parser; use plain official template')
            ids, text = c.get('token_ids') or [], delta.get('content') or ''
            if not isinstance(ids, list) or any(type(x) is not int for x in ids) or not isinstance(text, str):
                raise ValueError('Malformed token/text delta')
            if self.finish is not None:
                raise ValueError('Received a choice after the terminal choice')
            self.ids.extend(ids)
            self.text += text
            if ids:
                self.timings.append([len(self.ids), elapsed])
            if c.get('finish_reason') is not None:
                self.finish, self.stop_reason = c['finish_reason'], c.get('stop_reason')
        if payload.get('usage') is not None:
            self.usage = payload['usage']

    def validate(self, max_tokens, expected_prompt_ids_hash):
        if not self.done or self.finish not in {'stop', 'length'} or self.usage is None:
            raise ValueError('Incomplete stream; preserve partial response and do not resample')
        from .diagnostic_vllm import sha_json
        if sha_json(self.prompt_ids) != expected_prompt_ids_hash:
            raise ValueError('Served prompt IDs differ from the official processor')
        if (self.usage['completion_tokens'] != len(self.ids)
                or self.usage['prompt_tokens'] != len(self.prompt_ids)
                or len(self.ids) > max_tokens
                or (self.finish == 'length' and len(self.ids) != max_tokens)):
            raise ValueError('Usage/termination differs from the requested budget')


def consume_sse(lines, callback):
    data = []
    for line in lines:
        text = line.decode('utf-8').rstrip('\r\n') if isinstance(line, bytes) else line.rstrip('\r\n')
        if text == '':
            if data:
                payload = '\n'.join(data)
                data = []
                if payload == '[DONE]':
                    return True
                callback(json.loads(payload))
        elif text.startswith('data:'):
            data.append(text[5:].lstrip(' '))
        elif not text.startswith(':'):
            raise ValueError('Unexpected non-SSE server response')
    return False


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                            spaces_between_special_tokens=True)


def view(row, budget, decode_ids=None):
    count = min(len(row['token_ids']), budget)
    full = len(row['token_ids']) <= budget
    text = row['response'] if full else (decode_ids(row['token_ids'][:budget]) if decode_ids
                                        else row['prefix_8192_text'])
    if not full and decode_ids is None and budget != 8192:
        raise ValueError('Only the prepared 8192-token prefix is available')
    observed = next((t for n, t in row['token_timings'] if n >= count), row['latency_s'])
    return {'model': row['model'], 'question_id': row['question_id'],
            'response': text, 'output_tokens': count,
            'finish_reason': row['finish_reason'] if full else 'length',
            'stop_reason': row['stop_reason'] if full else None,
            'budget_tokens': budget, 'derived_from_same_16k_trajectory': True,
            'latency_s': row['latency_s'] if full else None,
            'token_boundary_observed_s': observed,
            'timing_scope': 'Client arrival of first chunk reaching this token count; '
                            'an upper bound with chunk granularity, not a separate 8k request latency',
            'source_trajectory_sha256': fingerprint(row),
            'image_sha256': row['image_sha256']}


def raw_file(root, model):
    return root / 'results/raw' / NS / (model + '.jsonl')


def run_dir(root):
    return root / 'results/runs' / NS


def report_dir(root):
    return root / 'results/reports' / NS


def audit_attempts(root, model, ids, run_fingerprint):
    rows = read_jsonl(raw_file(root, model))
    if list(unique(rows)) != ids[:len(rows)]:
        raise ValueError('Raw IDs must be a unique prefix of the selected order')
    events = read_jsonl(run_dir(root) / model / 'http-events.jsonl')
    submitted = Counter(str(e['question_id']) for e in events if e['event'] == 'submitted')
    if any(n != 1 or q not in ids for q, n in submitted.items()):
        raise ValueError('Duplicate or unknown submitted attempt; no automatic retry')
    for row in rows:
        if row['run_fingerprint'] != run_fingerprint or submitted[row['question_id']] != 1:
            raise ValueError('Run identity or submitted evidence differs')
        journal = root / row['journal_path']
        if not journal.is_relative_to(root / 'results/runs' / NS) or digest(journal) != row['journal_sha256']:
            raise ValueError('Immutable stream journal changed')
        saved = [e for e in events if e['event'] == 'received_and_saved' and e['question_id'] == row['question_id']]
        if len(saved) != 1 or saved[0]['trajectory_sha256'] != fingerprint(row):
            raise ValueError('Saved trajectory receipt differs; inspect without regeneration')
    pending = set(submitted) - set(unique(rows))
    if pending:
        raise ValueError('Uncertain previous HTTP attempt; inspect journal without resampling: ' + ','.join(sorted(pending)))
    return rows


def request_body(config, model, item, image_bytes):
    g = config['generation']
    return {'model': 'ovr-' + model, 'seed': config['seed'], 'n': 1,
            'max_tokens': config['max_new_tokens'], 'min_tokens': 0,
            'temperature': g['temperature'], 'top_p': g['top_p'], 'top_k': g['top_k'],
            'repetition_penalty': g['repetition_penalty'], 'ignore_eos': False,
            'stream': True, 'stream_options': {'include_usage': True}, 'return_token_ids': True,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode()}},
                {'type': 'text', 'text': item['prompt']}]}]}


@contextlib.contextmanager
def time_limit(seconds):
    """Nest request and whole-run limits; neither is a billing/instance limit."""
    if seconds <= 0:
        raise TimeoutError('Runtime limit reached')
    def expired(*_):
        raise TimeoutError('Bounded execution time reached; preserve evidence, do not retry')
    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    if old_timer[0] > 0:
        signal.setitimer(signal.ITIMER_REAL, min(seconds, old_timer[0]))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        remaining = old_timer[0] - (time.monotonic() - started)
        if old_timer[0] > 0 and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining)


def server_command(config, model, load, port):
    return [PYTHON, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(load),
            '--served-model-name', 'ovr-' + model, '--host', '127.0.0.1', '--port', str(port),
            '--tensor-parallel-size', '1', '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
            '--dtype', config['dtype'], '--max-model-len', str(config['max_context_tokens']),
            '--max-num-seqs', '1', '--max-num-batched-tokens', str(config['max_context_tokens']),
            '--generation-config', 'auto', '--seed', str(config['seed']), '--enforce-eager',
            '--limit-mm-per-prompt', '{"image":1,"video":0}',
            '--mm-processor-kwargs', json.dumps({**config['image_processor'], 'use_fast': True}),
            '--disable-log-requests']


@contextlib.contextmanager
def server(config, model, load, out):
    from .diagnostic_vllm import http
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = server_command(config, model, load, port)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / ('server-' + str(time.time_ns()) + '.json')
    log_path = state_path.with_suffix('.log')
    state = {'command': command, 'started_utc': utc(), 'stopped': False}
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_NO_USAGE_STATS': '1',
           'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    with log_path.open('x') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, start_new_session=True)
    try:
        state['pid'] = proc.pid
        write_json(state_path, state)
        deadline = time.monotonic() + config['server_startup_max_s']
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('vLLM startup failed; inspect saved server log')
            try:
                http(port, '/health', timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError('vLLM startup time limit reached')
        state['served_models'] = http(port, '/v1/models')
        if str(load) not in {x.get('root') for x in state['served_models']['data']}:
            raise ValueError('Served checkpoint differs from pinned local path')
        write_json(state_path, state)
        yield port
    finally:
        # Stop only this process group; never control the platform instance.
        prior_timer = signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_started = time.monotonic()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        state.update(stopped=True, exit_code=proc.returncode, ended_utc=utc())
        write_json(state_path, state)
        remaining = prior_timer[0] - (time.monotonic() - cleanup_started)
        if prior_timer[0] > 0 and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining)


def request(config, model, item, load, port, tokenizer, fp, root=ROOT):
    from .diagnostic_vllm import expected_input
    expected = expected_input(config, model, item, load)
    if expected['input_tokens'] + config['max_new_tokens'] > config['max_context_tokens']:
        raise ValueError('Input plus output exceeds context; never silently truncate input')
    image = root / config['shared_dir'] / item['image']
    body = request_body(config, model, item, image.read_bytes())
    events = run_dir(root) / model / 'http-events.jsonl'
    journal = run_dir(root) / model / ('stream-' + item['question_id'] + '.jsonl')
    journal.parent.mkdir(parents=True, exist_ok=True)
    state = Stream()
    started = time.monotonic()
    # Create journal exclusively and durably log submission BEFORE sending.
    with journal.open('x', encoding='utf-8') as log:
        append(events, {'event': 'submitted', 'question_id': item['question_id'], 'utc': utc(),
                        'run_fingerprint': fp, 'request_decoding': {k: v for k, v in body.items() if k != 'messages'}})
        def on_chunk(payload):
            elapsed = time.monotonic() - started
            log.write(json.dumps({'elapsed_s': elapsed, 'payload': payload}, ensure_ascii=False) + '\n')
            log.flush()
            state.add(payload, elapsed)
        http_request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                       data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with time_limit(config['per_question_max_s']):
            with urllib.request.urlopen(http_request, timeout=config['per_question_max_s']) as response:
                state.done = consume_sse(response, on_chunk)
        elapsed = time.monotonic() - started
        log.write(json.dumps({'elapsed_s': elapsed, 'done': state.done}) + '\n')
        log.flush()
        os.fsync(log.fileno())
    state.validate(config['max_new_tokens'], expected['prompt_token_ids_sha256'])
    if decode(tokenizer, state.ids) != state.text:
        raise ValueError('Full token decoding differs from server text; preserve journal, do not rerun')
    row = {'model': model, 'question_id': item['question_id'], 'response': state.text,
           'token_ids': state.ids, 'token_timings': state.timings,
           'finish_reason': state.finish, 'stop_reason': state.stop_reason,
           'output_tokens': len(state.ids), 'input_tokens': len(state.prompt_ids),
           'latency_s': elapsed, 'image_sha256': item['image_sha256'],
           'expected_tf_input': expected, 'prompt_token_ids_verified': True,
           'run_fingerprint': fp, 'journal_path': journal.relative_to(root).as_posix(),
           'journal_sha256': digest(journal), 'received_utc': utc()}
    row['prefix_8192_text'] = (state.text if len(state.ids) <= 8192
                               else decode(tokenizer, state.ids[:8192]))
    row['prefix_decode_method'] = 'Pinned checkpoint tokenizer; skip_special_tokens=True; cleanup=False'
    append(raw_file(root, model), row)
    append(events, {'event': 'received_and_saved', 'question_id': item['question_id'], 'utc': utc(),
                    'trajectory_sha256': fingerprint(row)})
    return row


def runtime_check():
    code = """import importlib.metadata as m,json,platform,torch
assert platform.system()=='Linux' and platform.python_version().startswith('3.12.')
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
p=torch.cuda.get_device_properties(0)
assert p.total_memory>=40*2**30
print(json.dumps({'versions':{k:m.version(k) for k in ['vllm','torch','transformers']},
 'gpu':p.name,'vram_bytes':p.total_memory,'cuda':torch.version.cuda}))
"""
    data = json.loads(subprocess.check_output([PYTHON, '-c', code], text=True, timeout=45))
    if data['versions'] != {'vllm': '0.11.0', 'torch': '2.8.0+cu128', 'transformers': '4.57.1'} or data['cuda'] != '12.8':
        raise ValueError('Serving runtime differs from the pinned existing environment')
    if importlib.metadata.version('transformers') != '4.57.1':
        raise ValueError('Client processor/tokenizer version differs')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=15).strip()
    if active:
        raise RuntimeError('GPU already has a compute process; inspect without killing unrelated work')
    return data


def run(minutes, root=ROOT):
    if not math.isfinite(minutes) or minutes < 1:
        raise ValueError('A finite positive --minutes runtime limit is required')
    ready = check(root)
    config = read_json(root / CONFIG)
    rows, _ = inputs(config, root)
    selected = [r for r in rows if r['question_id'] in ready['question_ids']]
    source_files = ['ovr/budget_compare.py', 'ovr/common.py', 'ovr/cloud.py',
                    'ovr/cloud_backend.py', 'ovr/diagnostic_vllm.py', 'ovr/scoring.py']
    fp = fingerprint({'config': config, 'selection': read_json(root / WORK / 'selection.json'),
                      'code': {p: digest(root / p) for p in source_files}})
    out = run_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out / 'controller.lock'):
        completed = {m: audit_attempts(root, m, ready['question_ids'], fp) for m in MODELS}
        if all(len(v) == 20 for v in completed.values()):
            return {'state': 'already_complete', 'new_generations': 0, 'next': 'Run --report on CPU'}
        state = {'state': 'starting', 'started_utc': utc(), 'minutes_limit': minutes,
                 'run_fingerprint': fp, 'instance_power_or_billing_changed': False,
                 'servers_stopped': True, 'planned_generations': 40}
        state_file = out / 'controller-state.json'
        session_file = out / ('controller-session-' + str(time.time_ns()) + '.json')
        def save_state():
            write_json(session_file, state)
            write_json(state_file, state)
        save_state()
        started = time.monotonic()
        try:
            with time_limit(minutes * 60):
                hardware = runtime_check()
                from .cloud_backend import checkpoint
                from transformers import AutoTokenizer
                state['hardware'] = hardware
                for model in MODELS:
                    if len(completed[model]) == 20:
                        continue
                    load, identity = checkpoint(config, model)
                    identity_file = out / model / 'checkpoint.json'
                    frozen_json(identity_file, identity)
                    tokenizer = AutoTokenizer.from_pretrained(load, local_files_only=True, trust_remote_code=False)
                    state.update(state='running', current_model=model)
                    save_state()
                    with server(config, model, load, out / model) as port:
                        state['servers_stopped'] = False
                        save_state()
                        for item in selected[len(completed[model]):]:
                            remaining = minutes * 60 - (time.monotonic() - started)
                            if remaining < config['per_question_max_s'] + 30:
                                state['state'] = 'incomplete_runtime_cap'
                                return state
                            row = request(config, model, item, load, port, tokenizer, fp, root)
                            completed[model].append(row)
                            state['completed'] = {m: len(v) for m, v in completed.items()}
                            save_state()
                            print(json.dumps({'model': model, 'id': item['question_id'],
                                              'tokens': row['output_tokens'], 'finish': row['finish_reason']}), flush=True)
                    state['servers_stopped'] = True
                state['state'] = 'generation_complete_pending_cpu_scoring'
        except BaseException as exc:
            state.update(state='stopped_error_or_limit', error_type=type(exc).__name__, error=str(exc)[:400])
            raise
        finally:
            server_states = [read_json(p) for p in out.glob('*/server-*.json')]
            state.update(ended_utc=utc(), task_runtime_s=time.monotonic() - started,
                         servers_stopped=all(s.get('stopped') for s in server_states),
                         instance_power_or_billing_changed=False)
            try:
                state['gpu_compute_pids_after'] = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                    text=True, timeout=15).strip().splitlines()
                state['gpu_compute_idle_after'] = not state['gpu_compute_pids_after']
            except (OSError, subprocess.SubprocessError):
                state['gpu_compute_idle_after'] = None
            save_state()
            print('Check controller-state.json for service shutdown. Instance billing is controlled by the user.', flush=True)
    return state


def has_box(text):
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        depth = 1
        for i in range(match.end(), len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                if text[match.end():i].strip():
                    return True
                break
    return False


def summarize(rows):
    n = len(rows)
    correct = sum(bool(r['correct']) for r in rows)
    return {'observed_n': n, 'expected_n': 20, 'correct': correct,
            'accuracy_over_observed': correct / n if n else None, 'wilson_ci95': wilson(correct, n),
            'natural_stop_n': sum(r['finish_reason'] == 'stop' for r in rows),
            'length_limit_n': sum(r['finish_reason'] == 'length' for r in rows),
            'empty_n': sum(not r['response'].strip() for r in rows),
            'nonempty_box_anywhere_n': sum(has_box(r['response']) for r in rows),
            'natural_stop_with_box_n': sum(r['finish_reason'] == 'stop' and has_box(r['response']) for r in rows),
            'think_closed_n': sum('</think>' in r['response'] for r in rows),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in rows) if n else None,
            'mean_token_boundary_observed_s': statistics.mean(r['token_boundary_observed_s'] for r in rows) if n else None}


def report(root=ROOT):
    from .scoring import score
    ready = check(root)
    config = read_json(root / CONFIG)
    refs = root / WORK / 'references.jsonl'
    output = report_dir(root)
    scored, summaries, paired = {}, [], []
    originals = {m: digest(raw_file(root, m)) for m in MODELS if raw_file(root, m).exists()}
    for model in MODELS:
        raw = read_jsonl(raw_file(root, model))
        if not raw:
            continue
        if list(unique(raw)) != ready['question_ids'][:len(raw)]:
            raise ValueError('Missing, reordered, duplicate, or unknown trajectory IDs')
        audit_attempts(root, model, ready['question_ids'], raw[0]['run_fingerprint'])
        for row in raw:
            if digest(root / row['journal_path']) != row['journal_sha256']:
                raise ValueError('Stream evidence changed')
        for budget in config['budgets']:
            # Prefix text is decoded and pinned in each original cloud receipt.
            # Mac rescoring therefore needs neither model weights nor transformers.
            rows = [view(r, budget) for r in raw]
            derived = root / 'results/scored' / NS / model / (str(budget) + '-derived.jsonl')
            target = derived.with_name(str(budget) + '-scored.jsonl')
            write_jsonl(derived, rows)
            score(derived, refs, target)
            scored[model, budget] = read_jsonl(target)
            summaries.append({'model': model, 'budget_tokens': budget, **summarize(scored[model, budget])})
        a, b = [scored[model, budget] for budget in config['budgets']]
        for x, y in zip(a, b):
            paired.append({'model': model, 'question_id': x['question_id'],
                           'correct_8192': x['correct'], 'correct_16384': y['correct'],
                           'finish_8192': x['finish_reason'], 'finish_16384': y['finish_reason'],
                           'tokens_8192': x['output_tokens'], 'tokens_16384': y['output_tokens'],
                           'extra_tokens': y['output_tokens'] - x['output_tokens'],
                           'boundary_s_8192': x['token_boundary_observed_s'],
                           'boundary_s_16384': y['token_boundary_observed_s'],
                           'extra_boundary_observed_s': y['token_boundary_observed_s'] - x['token_boundary_observed_s']})
    if any(digest(raw_file(root, m)) != sha for m, sha in originals.items()):
        raise ValueError('Original trajectories changed during scoring')
    complete = len(summaries) == 4 and all(s['observed_n'] == 20 for s in summaries)
    transitions = {}
    for model in MODELS:
        rows = [r for r in paired if r['model'] == model]
        transitions[model] = {'observed_n': len(rows),
                             'improved': sum(not r['correct_8192'] and r['correct_16384'] for r in rows),
                             'regressed': sum(r['correct_8192'] and not r['correct_16384'] for r in rows)}
    value = {'complete': complete, 'status': 'COMPLETE' if complete else 'INCOMPLETE',
             'method': config['method'], 'summaries': summaries, 'transitions': transitions,
             'original_trajectory_sha256': originals,
             'limits': ['Exploratory selected subset of already observed questions, not a holdout.',
                        '8k is a token-prefix view, not an independently executed 8k request.',
                        'Chunk-arrival timing is approximate; total charged runtime is separately recorded.',
                        'Nonempty boxes and closing think markers do not prove visual reflection or correct finalization.',
                        'Do not combine best answers across budgets or replace original 100-question results.',
                        'Accuracy over observed rows is not a complete 20-question score when complete=false.']}
    write_json(output / 'comparison.json', value)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'paired.csv', paired)
    return value


def export(root=ROOT):
    result = read_json(report_dir(root) / 'comparison.json')
    if not result['complete']:
        raise ValueError('Comparison incomplete; do not label an export as complete')
    config = read_json(root / CONFIG)
    rows, shared = inputs(config, root)
    ids = read_json(root / WORK / 'selection.json')['question_ids']
    fixed = [root / CONFIG, root / BASE_CONFIG, root / WORK / 'selection.json',
             root / WORK / 'references.jsonl', root / WORK / 'README.md', root / 'ovr/budget_compare.py']
    files = set(fixed)
    files.update(root / p for p in ['configs/protocol.json', 'sources/manifest.json',
                                   'ovr/common.py', 'ovr/scoring.py'])
    files.update((root / 'vendor/mathvision').glob('*.py'))
    # Exact prompts are included without reference labels, alongside scoring references.
    subset_inputs = report_dir(root) / 'selected_inputs.jsonl'
    write_jsonl(subset_inputs, [r for r in rows if r['question_id'] in ids])
    files.add(subset_inputs)
    for folder in [root / 'results/raw' / NS, root / 'results/scored' / NS, run_dir(root), report_dir(root)]:
        files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix not in {'.log', '.lock'})
    files.update(shared / r['image'] for r in rows if r['question_id'] in ids)
    out = root / 'artifacts/OVR_BUDGET20_8K_16K_RESULTS.zip'
    out.parent.mkdir(parents=True, exist_ok=True)
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)}
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(root).as_posix())
        z.writestr('SHA256SUMS.json', json.dumps(hashes, indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip() or any(__import__('hashlib').sha256(z.read(k)).hexdigest() != v for k, v in hashes.items()):
            raise ValueError('Export CRC/member SHA mismatch')
    return {'file': str(out), 'sha256': digest(out), 'members_verified': len(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ['prepare', 'check', 'run', 'report', 'export']:
        group.add_argument('--' + name, action='store_true')
    parser.add_argument('--minutes', type=float)
    args = parser.parse_args()
    if args.run:
        if args.minutes is None:
            parser.error('--run needs an explicit --minutes runtime limit; no implicit GPU launch')
        result = run(args.minutes)
    else:
        if args.minutes is not None:
            parser.error('--minutes applies only to --run')
        result = (prepare if args.prepare else check if args.check else report if args.report else export)()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
