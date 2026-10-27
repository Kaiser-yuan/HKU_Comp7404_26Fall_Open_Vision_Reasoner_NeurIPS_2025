"""Isolated four-generation decoding diagnostic; never starts a formal pilot."""
import argparse
import copy
import json
from pathlib import Path
from .common import ROOT, digest, fingerprint, read_json
from .cloud import execute, inputs

BASE_CONFIG = ROOT / 'configs/cloud_shared100.json'
DIAGNOSTIC_CONFIG = ROOT / 'configs/cloud_diagnostic_sampling.json'
SAMPLING = {'do_sample': True, 'num_beams': 1, 'repetition_penalty': 1.0,
            'temperature': 0.6, 'top_p': 0.95, 'top_k': 0}


def configuration(base):
    config = copy.deepcopy(base)
    config.update(
        protocol_id='ovr-shared100-decoding-diagnostic-sampling-v1',
        results_namespace='cloud_diag_sampling_20261022',
        config_source='configs/cloud_diagnostic_sampling.json',
        status='Isolated decoding diagnostic; not a frozen formal evaluation protocol',
        purpose='Same first two engineering questions per checkpoint; change decoding only',
        limits='Four sampled diagnostic outputs only. Never replace greedy outputs, combine best attempts, or include these in formal accuracy.',
        generation=SAMPLING.copy(),
        diagnostic_only=True,
        diagnostic_max_generations=4,
        diagnostic_comparator='cloud_shared100/canary',
    )
    return config


def apply_sampling(backend, config):
    # The formal backend clears stochastic knobs for greedy evaluation. Set the
    # diagnostic knobs explicitly after initialization, without editing that core.
    if config['generation'] != SAMPLING:
        raise ValueError('Unexpected diagnostic sampling parameters')
    for key, value in SAMPLING.items():
        setattr(backend.generation, key, value)
    backend.generation.validate()
    backend.metadata['effective_generation'] = backend.generation.to_dict()
    backend.metadata['diagnostic_only'] = True
    return backend


def run(model):
    from .cloud_backend import checkpoint, gpu_check, TransformersBackend
    base = read_json(BASE_CONFIG)
    config = read_json(DIAGNOSTIC_CONFIG)
    if config != configuration(base) or config['canary_questions'] != 2 or len(config['models']) != 2:
        raise ValueError('Diagnostic differs from the prespecified four-generation comparison')
    inputs(config)
    hardware = gpu_check(config)
    load_dir, manifest = checkpoint(config, model)
    identity = {'hardware': hardware, 'checkpoint': manifest,
                'code': {p.name: digest(p) for p in
                         [ROOT/'ovr/cloud.py', ROOT/'ovr/cloud_backend.py', Path(__file__)]},
                'diagnostic_only': True}
    return execute(config, model, 'canary', identity,
                   lambda: apply_sampling(TransformersBackend(config, load_dir), config),
                   session_minutes=25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['coldstart', 'rl'], required=True)
    args = parser.parse_args()
    result = run(args.model)
    print(json.dumps({k: result.get(k) for k in ['model', 'phase', 'state', 'completed', 'expected', 'elapsed_s']}, indent=2))
    return 0 if result['state'] in {'complete', 'already_complete'} else 3


if __name__ == '__main__':
    raise SystemExit(main())
