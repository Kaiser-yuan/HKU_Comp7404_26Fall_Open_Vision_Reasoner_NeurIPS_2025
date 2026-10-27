"""One bounded vLLM diagnostic: 10 answers and 3 one-token image probes.

No formal 100-question entry point and no automatic transition to evaluation.
"""
import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import random
import signal
import socket
import struct
import subprocess
import time
from pathlib import Path

from .common import ROOT, digest, fingerprint, read_json, read_jsonl, write_json
from .cloud import append, inputs, run_lock
from .diagnostic_vllm import expected_input, http, sha_json

CONFIG = ROOT / 'configs/cloud_vllm_preflight8192.json'
PYTHON = '/root/ovr-course-env/vllm011/bin/python'


def choose_ids(rows):
    rng = random.Random(7404)
    return [rng.choice(sorted(str(r['question_id']) for r in rows
                              if int(r['level']) == level and str(r['question_id']) not in {'40', '51'}))
            for level in range(1, 6)]


def validate(config, rows):
    if (config['preflight_ids'] != choose_ids(rows)
            or config['generation'] != {'do_sample': True, 'temperature': 0.6, 'top_p': 0.95,
                                         'top_k': 40, 'repetition_penalty': 1.0, 'num_beams': 1}
            or config['max_new_tokens'] != 8192 or config['max_context_tokens'] != 16384
            or config['diagnostic_max_answer_generations'] != 10
            or config['diagnostic_max_probe_generations'] != 3
            or config['results_namespace'] != 'cloud_vllm_preflight8192_v1'
            or config['diagnostic_only'] is not True):
        raise ValueError('Preflight differs from the fixed bounded plan')


def inspect_headers(load):
    """Read safetensors headers only, without loading tensors or using CUDA."""
    tensors = {}
    for path in sorted(Path(load).glob('*.safetensors')):
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 64 * 2**20:
                raise ValueError('Unexpected tensor header size')
            header = json.loads(f.read(size))
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError('Duplicate tensor name')
            tensors[name] = {'shape': spec['shape'], 'dtype': spec['dtype']}
    vision = {k: v for k, v in tensors.items() if k.startswith(('visual.', 'model.visual.'))}
    if not vision or not any('embed_tokens.weight' in k for k in tensors):
        raise ValueError('Vision or language tensors are missing from checkpoint headers')
    return {'tensor_count': len(tensors), 'vision_tensor_count': len(vision),
            'vision_header_sha256': fingerprint(vision),
            'vision_examples': dict(list(sorted(vision.items()))[:5]),
            'scope': 'Header presence only; not evidence that loaded inference uses images.'}


def first_logprobs(response):
    content = (response['choices'][0].get('logprobs') or {}).get('content') or []
    if not content:
        raise ValueError('Visual probe did not return first-token log probabilities')
    return {x['token']: x['logprob'] for x in content[0]['top_logprobs']}


def visual_effect(probes):
    a, repeat, blank = [first_logprobs(p['api_response']) for p in probes]
    common = a.keys() & repeat.keys()
    noise = max((abs(a[k] - repeat[k]) for k in common), default=None)
    common_blank = a.keys() & blank.keys()
    effect = max((abs(a[k] - blank[k]) for k in common_blank), default=0.0)
    choices = [p['api_response']['choices'][0]['token_ids'] for p in probes]
    prompt_ids = [p['api_response'].get('prompt_token_ids') for p in probes]
    same_inputs = prompt_ids[0] is not None and prompt_ids[0] == prompt_ids[1] == prompt_ids[2]
    stable = choices[0] == choices[1] and noise is not None and noise <= 0.001
    changed = choices[0] != choices[2] or (noise is not None and effect > max(0.01, 10 * noise))
    return {'same_prompt_token_ids': same_inputs, 'same_image_control_stable': stable,
            'max_common_token_logprob_control_difference': noise,
            'max_common_token_logprob_image_difference': effect,
            'image_affects_output_distribution': same_inputs and stable and changed,
            'interpretation': 'A positive result supports pixel influence on serving output; not visual understanding. A negative result is inconclusive.'}


@contextlib.contextmanager
def server(config, model, load, out):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [PYTHON, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(load),
               '--served-model-name', 'ovr-' + model, '--host', '127.0.0.1', '--port', str(port),
               '--tensor-parallel-size', '1', '--gpu-memory-utilization', '0.6',
               '--dtype', 'bfloat16', '--max-model-len', '16384', '--max-num-seqs', '1',
               '--max-num-batched-tokens', '16384', '--generation-config', 'auto',
               '--seed', '42', '--enforce-eager', '--limit-mm-per-prompt', '{"image":1,"video":0}',
               '--mm-processor-kwargs', json.dumps(config['image_processor']), '--disable-log-requests']
    logdir = Path('/root/ovr-private-logs')
    logdir.mkdir(exist_ok=True)
    state = {'command': command, 'model': model, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    path = out / (model + '-server.json')
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
           'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    with (logdir / ('repetition-preflight-' + model + '.log')).open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    try:
        state['pid'] = process.pid
        write_json(path, state)
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('vLLM exited; inspect the private server log')
            try:
                http(port, '/health', timeout=2)
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError('vLLM startup exceeded six minutes')
        state['served_models'] = http(port, '/v1/models')
        if str(load) not in {x.get('root') for x in state['served_models']['data']}:
            raise ValueError('Served model differs from pinned checkpoint')
        write_json(path, state)
        yield port
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log_path = logdir / ('repetition-preflight-' + model + '.log')
        state['attention_backend_log_evidence'] = [line[-500:] for line in log_path.read_text(errors='replace').splitlines()
            if 'attention backend' in line.lower() or 'flash attention backend' in line.lower()][-5:]
        state.update(stopped=True, exit_code=process.returncode,
                     ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(path, state)


def request(config, port, model, item, image_bytes, load, events, probe=False):
    body = {'model': 'ovr-' + model, 'seed': 42, 'max_tokens': 1 if probe else 8192,
            'return_token_ids': True,
            'temperature': 0.0 if probe else 0.6, 'top_p': 1.0 if probe else 0.95,
            'top_k': -1 if probe else 40, 'repetition_penalty': 1.0,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode()}},
                {'type': 'text', 'text': item['prompt']}]}]}
    if probe:
        body.update(logprobs=True, top_logprobs=5)
    expected = expected_input(config, model, item, load)
    if expected['input_tokens'] + body['max_tokens'] > config['max_context_tokens']:
        raise ValueError('Context budget too small; do not silently truncate input')
    append(events, {'event': 'submitted', 'model': model, 'question_id': item['question_id'], 'probe': probe})
    started = time.monotonic()
    result = http(port, '/v1/chat/completions', body, timeout=300)
    append(events, {'event': 'received', 'model': model, 'question_id': item['question_id'], 'probe': probe,
                    'image_sha256': hashlib.sha256(image_bytes).hexdigest(), 'api_response': result})
    choice = result['choices'][0]
    if sha_json(result.get('prompt_token_ids')) != expected['prompt_token_ids_sha256']:
        raise ValueError('Prompt tokens differ from the verified processor path')
    tokens = result['usage']['completion_tokens']
    if tokens > body['max_tokens'] or tokens != len(choice['token_ids']) or choice['finish_reason'] not in {'stop', 'length'}:
        raise ValueError('Unexpected completion usage or termination')
    return {'model': model, 'question_id': item['question_id'], 'diagnostic_only': True,
            'response': choice['message'].get('content') or '', 'output_tokens': tokens,
            'finish_reason': choice['finish_reason'], 'latency_s': time.monotonic() - started,
            'image_tensor_checked': True, 'image_verification_scope': 'HTTP image sent; input IDs checked; separate visual sensitivity probe',
            'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
            'request_decoding': {k: v for k, v in body.items() if k != 'messages'},
            'api_response': result, 'expected_tf_input': expected}


def run():
    from PIL import Image
    from .cloud_backend import checkpoint, gpu_check
    from .scoring import score
    config = read_json(CONFIG)
    rows, shared = inputs(config)
    validate(config, rows)
    base = ROOT / 'results'
    ns = config['results_namespace']
    out = base / 'runs' / ns
    raw = base / 'raw' / ns
    events = out / 'events.jsonl'
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out / 'controller.lock'):
        if events.exists() or list(raw.glob('*.jsonl')):
            raise ValueError('Existing attempts; inspect instead of automatically resampling')
        hardware = gpu_check(config)
        runtime = json.loads(subprocess.check_output([PYTHON, '-c',
            "import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in ['vllm','torch','transformers']}))"], text=True, timeout=20))
        if runtime != {'vllm': '0.11.0', 'torch': '2.8.0+cu128', 'transformers': '4.57.1'}:
            raise ValueError('vLLM serving environment differs from the audited runtime')
        write_json(out / 'plan.json', {'config': config, 'hardware': hardware, 'serving_runtime': runtime,
                   'source_sha256': digest(Path(__file__)), 'helper_sha256': digest(ROOT / 'ovr/diagnostic_vllm.py'),
                   'formal_generations': 0, 'hard_wallclock_limit_minutes': 40})
        def timeout(*_):
            raise TimeoutError('Forty-minute diagnostic budget reached')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(40 * 60)
        try:
            effect = None
            completions = {}
            for model in ['rl', 'coldstart']:
                load, manifest = checkpoint(config, model)
                write_json(out / (model + '-headers.json'), {'checkpoint': manifest, 'headers': inspect_headers(load)})
                selected = [{r['question_id']: r for r in rows}[qid] for qid in config['preflight_ids']]
                with server(config, model, load, out) as port:
                    if model == 'rl':
                        item = next(r for r in rows if r['question_id'] == '40')
                        image = shared / item['image']
                        original = image.read_bytes()
                        with Image.open(image) as im:
                            buf = io.BytesIO()
                            Image.new('RGB', im.size, 'white').save(buf, format='JPEG')
                        probes = []
                        for condition, data in [('original', original), ('same_image_repeat', original), ('white_same_size', buf.getvalue())]:
                            row = request(config, port, model, item, data, load, events, probe=True)
                            row['probe_condition'] = condition
                            append(raw / 'visual_probes.jsonl', row)
                            probes.append(row)
                        effect = visual_effect(probes)
                        write_json(base / 'reports' / ns / 'visual_effect.json', effect)
                        if not effect['image_affects_output_distribution']:
                            raise ValueError('Visual sensitivity unresolved; inspect before spending on more answers')
                    answers = []
                    for item in selected:
                        row = request(config, port, model, item, (shared / item['image']).read_bytes(), load, events)
                        append(raw / (model + '.jsonl'), row)
                        answers.append(row)
                        print(json.dumps({'model': model, 'question_id': item['question_id'], 'tokens': row['output_tokens'], 'finish': row['finish_reason']}), flush=True)
                scored = base / 'scored' / ns / (model + '.jsonl')
                stats = score(raw / (model + '.jsonl'), shared / 'references.jsonl', scored)
                completions[model] = {**stats, 'stop': sum(x['finish_reason'] == 'stop' for x in answers),
                                      'length': sum(x['finish_reason'] == 'length' for x in answers)}
            write_json(base / 'reports' / ns / 'complete.json', {
                'infrastructure_valid': True, 'visual_effect': effect, 'models': completions,
                'all_servers_stopped': True, 'formal_generations': 0,
                'accuracy_is_not_a_pass_threshold': True,
                'next': 'Review protocol compatibility and measured cost before separately authorizing 100 questions; no auto-start.'})
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-bounded-gpu-preflight', action='store_true')
    args = parser.parse_args()
    if not args.run_bounded_gpu_preflight:
        parser.error('Explicit bounded GPU execution flag required; no formal evaluation mode exists')
    run()
