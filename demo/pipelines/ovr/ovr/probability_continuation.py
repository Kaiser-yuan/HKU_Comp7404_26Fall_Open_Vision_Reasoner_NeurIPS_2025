"""Review coarse probability mass, then run only the ten unattempted answers.

The original maximum-logprob control failure is retained unchanged. This is an
explicit post-hoc engineering review, not a prespecified statistical result.
No probes are regenerated; the original 13-request/40-minute budget is shared.
"""
import argparse
import datetime as dt
import math
import signal
import time

from .common import ROOT, digest, read_json, read_jsonl, write_json
from .cloud import append, inputs, run_lock
from .repetition_preflight import CONFIG, PYTHON, first_logprobs, inspect_headers, request, server, validate

OLD_NS = 'cloud_vllm_preflight8192_v1'
NS = 'cloud_vllm_preflight8192_probability_review_v1'


def probability_review(probes):
    if len(probes) != 3 or [p['probe_condition'] for p in probes] != ['original', 'same_image_repeat', 'white_same_size']:
        raise ValueError('Expected exactly the original three probes in order')
    maps = [first_logprobs(p['api_response']) for p in probes]
    if any(not math.isfinite(v) or v > 0 for m in maps for v in m.values()):
        raise ValueError('Invalid log probabilities')
    common = set.intersection(*(set(m) for m in maps))
    if not common:
        raise ValueError('No common probability bins')
    bins = []
    for m in maps:
        values = [math.exp(m[k]) for k in sorted(common)]
        if sum(values) > 1.00001:
            raise ValueError('Probability mass exceeds one')
        bins.append(values + [max(0.0, 1.0 - sum(values))])
    tv = lambda a, b: 0.5 * sum(abs(x - y) for x, y in zip(a, b))
    noise, effect = tv(bins[0], bins[1]), tv(bins[0], bins[2])
    prompts = [p['api_response'].get('prompt_token_ids') for p in probes]
    same_prompts = prompts[0] is not None and prompts[0] == prompts[1] == prompts[2]
    same_token = probes[0]['api_response']['choices'][0]['token_ids'] == probes[1]['api_response']['choices'][0]['token_ids']
    supported = same_prompts and same_token and noise <= 0.001 and effect > max(0.001, 10 * noise)
    return {'post_hoc_engineering_review': True, 'original_gate_replaced_or_marked_passed': False,
            'common_token_bins': sorted(common), 'residual_bin': 'all_other_tokens',
            'method': 'Coarsen each full distribution into tokens present in all three top-k lists plus all other tokens. TV of this partition is a lower bound on full-distribution TV.',
            'same_image_total_variation': noise, 'white_image_total_variation': effect,
            'effect_to_control_ratio': effect / noise if noise else None,
            'pixel_influence_supported_by_review': supported,
            'limits': 'Three one-token traces only. Does not demonstrate visual understanding or establish that caching caused the numerical difference. The original strict logprob gate remains failed.'}


def run():
    from .cloud_backend import checkpoint, gpu_check
    from .scoring import score
    config = read_json(CONFIG)
    rows, shared = inputs(config)
    validate(config, rows)
    old = ROOT / 'results' / 'runs' / OLD_NS
    probes_path = ROOT / 'results' / 'raw' / OLD_NS / 'visual_probes.jsonl'
    old_events = read_jsonl(old / 'events.jsonl')
    if len(old_events) != 6 or any(not e.get('probe') for e in old_events):
        raise ValueError('Original attempts differ; do not exceed the shared request budget')
    if not read_json(old / 'rl-server.json').get('stopped'):
        raise ValueError('Original model server must be stopped')
    if list((ROOT / 'results' / 'raw' / OLD_NS).glob('rl.jsonl')) or list((ROOT / 'results' / 'raw' / OLD_NS).glob('coldstart.jsonl')):
        raise ValueError('Original answer attempts already exist')
    before = digest(probes_path)
    review = probability_review(read_jsonl(probes_path))
    review['original_probe_sha256'] = before
    review['original_strict_gate'] = read_json(ROOT / 'results' / 'reports' / OLD_NS / 'visual_effect.json')
    write_json(ROOT / 'results' / 'reports' / NS / 'probability_review.json', review)
    if not review['pixel_influence_supported_by_review']:
        raise ValueError('Pixel influence unresolved after explicit engineering review')
    started = dt.datetime.fromisoformat(read_json(ROOT / '.build/repetition-preflight-controller.json')['started_utc'])
    deadline = started + dt.timedelta(minutes=40)
    remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining < 600:
        raise TimeoutError('Insufficient original time budget for continuation; no automatic extension')
    out = ROOT / 'results' / 'runs' / NS
    raw = ROOT / 'results' / 'raw' / NS
    events = out / 'events.jsonl'
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out / 'controller.lock'):
        if events.exists() or list(raw.glob('*.jsonl')):
            raise ValueError('Continuation already attempted; no automatic resampling')
        hardware = gpu_check(config)
        write_json(out / 'plan.json', {'original_config': config, 'hardware': hardware,
                   'shared_original_deadline_utc': deadline.isoformat(), 'max_additional_requests': 10,
                   'original_probe_requests': 3, 'combined_request_limit': 13,
                   'formal_generations': 0, 'source_sha256': digest(ROOT / 'ovr/probability_continuation.py'),
                   'helper_sha256': digest(ROOT / 'ovr/repetition_preflight.py')})
        def timeout(*_):
            raise TimeoutError('Original forty-minute budget reached')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(max(1, math.floor(remaining)))
        summary = {}
        try:
            for model in ['rl', 'coldstart']:
                load, manifest = checkpoint(config, model)
                write_json(out / (model + '-headers.json'), {'checkpoint': manifest, 'headers': inspect_headers(load)})
                selected = [{r['question_id']: r for r in rows}[qid] for qid in config['preflight_ids']]
                answers = []
                with server(config, model, load, out) as port:
                    for item in selected:
                        if (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() < 300:
                            raise TimeoutError('Not enough budget to begin another full per-question allowance')
                        row = request(config, port, model, item, (shared / item['image']).read_bytes(), load, events)
                        row.update(source_sha256=digest(ROOT / 'ovr/probability_continuation.py'),
                                   scope='independent five-level diagnostic after explicit engineering review')
                        append(raw / (model + '.jsonl'), row)
                        answers.append(row)
                        print(__import__('json').dumps({'model': model, 'question_id': item['question_id'], 'tokens': row['output_tokens'], 'finish': row['finish_reason']}), flush=True)
                scored = ROOT / 'results' / 'scored' / NS / (model + '.jsonl')
                summary[model] = {**score(raw / (model + '.jsonl'), shared / 'references.jsonl', scored),
                                  'normal_stop': sum(x['finish_reason'] == 'stop' for x in answers),
                                  'length': sum(x['finish_reason'] == 'length' for x in answers)}
            assert digest(probes_path) == before
            write_json(ROOT / 'results' / 'reports' / NS / 'complete.json', {
                'five_level_answers_completed': True, 'models': summary, 'all_servers_stopped': True,
                'original_strict_probe_gate_passed': False, 'reviewed_pixel_influence_supported': True,
                'review_type': 'explicit post-hoc engineering audit; see probability_review.json',
                'combined_http_generations': 13, 'formal_generations': 0,
                'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'next': 'Review protocol, error traces and measured cost before separately authorizing the held 100-question evaluation.'})
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-remaining-answers', action='store_true')
    if not parser.parse_args().run_remaining_answers:
        parser.error('Explicit continuation flag required; no 100-question mode exists')
    run()
