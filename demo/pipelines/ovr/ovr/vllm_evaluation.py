"""Prepared shared100 vLLM runner; --check is offline, --run needs a time cap.

Uses immutable per-question records and the existing official scoring/export
pipeline. Never provisions instances, retries model errors, or starts a pair.
"""
import argparse
import contextlib
import json
import math
import signal
import subprocess
import time
from pathlib import Path

from .common import ROOT, digest, read_json, read_jsonl, unique
from .cloud import execute, inputs, namespace, raw_path, report, run_lock
from .repetition_preflight import PYTHON, request, server

CONFIG = ROOT / 'configs/cloud_vllm_shared1008192.json'
PREFLIGHT = 'cloud_vllm_preflight8192_probability_review_v1'


def preflight_ready(config, root=ROOT):
    checked = read_json(ROOT / 'configs/cloud_vllm_preflight8192.json')
    keys = ['models', 'backend', 'dtype', 'quantization', 'seed', 'max_new_tokens',
            'max_context_tokens', 'image_processor', 'generation', 'inputs_sha256',
            'references_sha256', 'prompt_version', 'image_processor_use_fast']
    if any(config.get(k) != checked.get(k) for k in keys):
        raise ValueError('Formal protocol differs from the tested preflight')
    path = root / 'results' / 'reports' / PREFLIGHT
    complete = read_json(path / 'complete.json')
    review = read_json(path / 'probability_review.json')
    if not (complete.get('five_level_answers_completed') and complete.get('all_servers_stopped')
            and review.get('pixel_influence_supported_by_review')
            and review.get('post_hoc_engineering_review')):
        raise ValueError('Preflight incomplete or pixel influence unresolved')
    evidence = {'complete.json': digest(path / 'complete.json'), 'probability_review.json': digest(path / 'probability_review.json')}
    for model in ['coldstart', 'rl']:
        raw = root / 'results' / 'raw' / PREFLIGHT / (model + '.jsonl')
        scored = root / 'results' / 'scored' / PREFLIGHT / (model + '.jsonl')
        provenance = read_json(str(scored) + '.provenance.json')
        if digest(raw) != provenance['raw_sha256'] or digest(scored) != provenance['scored_sha256']:
            raise ValueError('Preflight raw/scored provenance changed')
        evidence[model + '/raw'] = digest(raw)
        evidence[model + '/scored'] = digest(scored)
        records = read_jsonl(raw)
        if list(unique(records)) != checked['preflight_ids'] or complete['models'][model]['count'] != 5:
            raise ValueError('Preflight IDs or coverage differ')
        if any(r['model'] != model or r['finish_reason'] not in {'stop', 'length'} for r in records):
            raise ValueError('Preflight runtime/termination invalid')
    return {'ready_for_separate_budget_review': True, 'evidence_sha256': evidence, 'formal_generations_started': False,
            'post_hoc_pixel_review_disclosed': True,
            'correctness_is_not_a_gate': True,
            'warning': 'Seed does not guarantee bitwise reproducibility in online vLLM serving. Preserve each first attempt.'}


def pending_attempts(config, model, root=ROOT):
    completed = unique(read_jsonl(raw_path(root, 'pilot', model, config)))
    logs = root / 'results' / 'runs' / namespace(config) / 'pilot' / model
    pending = set()
    for path in logs.glob('http-events-*.jsonl'):
        for event in read_jsonl(path):
            if event.get('event') == 'submitted' and str(event['question_id']) not in completed:
                pending.add(str(event['question_id']))
    if pending:
        raise ValueError('Inspect uncertain previous HTTP attempts before resampling IDs: ' + ','.join(sorted(pending)))


class Backend:
    def __init__(self, config, model, load, port, rows, events):
        self.config, self.model, self.load, self.port, self.events = config, model, load, port, events
        self.rows = {(ROOT / config['shared_dir'] / r['image']).resolve(): r for r in rows}
        self.metadata = {'backend': 'vllm-0.11.0-openai-local-http',
                         'generation': config['generation'], 'max_new_tokens': config['max_new_tokens'],
                         'max_context_tokens': config['max_context_tokens'], 'seed_scope': '42 per request',
                         'image_validation': 'Original JPEG plus processor token-ID check; pixel influence supported by disclosed preflight engineering review',
                         'latency_scope': 'HTTP generation and response validation; CPU input preprocessing excluded',
                         'formal_subset': True, 'full_mathvision_reproduction': False}

    def infer(self, prompt, image_path, max_time_s):
        item = self.rows[Path(image_path).resolve()]
        if prompt != item['prompt'] or max_time_s != 300:
            raise ValueError('Prompt or request deadline differs from the fixed protocol')
        result = request(self.config, self.port, self.model, item, Path(image_path).read_bytes(), self.load, self.events)
        result['diagnostic_only'] = False
        result['scope'] = 'first attempt in the fixed 100-question subset'
        return result


def run(model, minutes):
    from .cloud_backend import checkpoint, gpu_check
    if not math.isfinite(minutes) or minutes < 10:
        raise ValueError('Explicit finite runtime cap of at least ten minutes required')
    config = read_json(CONFIG)
    evidence = preflight_ready(config)
    rows, _ = inputs(config)
    hardware = gpu_check(config)
    runtime = json.loads(subprocess.check_output([PYTHON, '-c',
        "import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in ['vllm','torch','transformers']}))"], text=True, timeout=20))
    if runtime != {'vllm': '0.11.0', 'torch': '2.8.0+cu128', 'transformers': '4.57.1'}:
        raise ValueError('Serving runtime differs from preflight')
    load, manifest = checkpoint(config, model)
    sources = ['ovr/vllm_evaluation.py', 'ovr/repetition_preflight.py', 'ovr/diagnostic_vllm.py',
               'ovr/cloud.py', 'ovr/cloud_backend.py']
    identity = {'hardware': hardware, 'checkpoint': manifest, 'runtime': runtime,
                'code': {p: digest(ROOT / p) for p in sources}, 'preflight_evidence': evidence['evidence_sha256']}
    out = ROOT / 'results' / 'runs' / namespace(config) / 'pilot' / model
    out.mkdir(parents=True, exist_ok=True)
    def timeout(*_):
        raise TimeoutError('Explicit run time cap reached; inspect saved attempts before resuming')
    with run_lock(out.parent / 'gpu-controller.lock'):
        pending_attempts(config, model)
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(math.ceil(minutes * 60))
        try:
            with contextlib.ExitStack() as stack:
                def factory():
                    stamp = str(time.time_ns())
                    session = out / ('backend-' + stamp)
                    session.mkdir()
                    port = stack.enter_context(server(config, model, load, session))
                    return Backend(config, model, load, port, rows, out / ('http-events-' + stamp + '.jsonl'))
                result = execute(config, model, 'pilot', identity, factory, session_minutes=minutes)
        finally:
            signal.alarm(0)
    coverage = report(config, phase='pilot')
    return {'model': model, 'state': result['state'], 'completed': result['completed'],
            'paired_coverage': coverage, 'model_server_stopped': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--model', choices=['coldstart', 'rl'])
    parser.add_argument('--minutes', type=float)
    args = parser.parse_args()
    if args.check:
        config = read_json(CONFIG)
        inputs(config)
        print(json.dumps(preflight_ready(config), indent=2))
    else:
        if args.model is None or args.minutes is None:
            parser.error('--run requires --model and an explicitly reviewed --minutes budget')
        print(json.dumps(run(args.model, args.minutes), indent=2))
