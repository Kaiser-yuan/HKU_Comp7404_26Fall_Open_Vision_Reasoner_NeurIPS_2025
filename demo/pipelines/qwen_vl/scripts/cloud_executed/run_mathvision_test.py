"""Small paired MathVision pilot; conservative scores are NOT official metrics."""
import argparse
import csv
import hashlib
import json
import re
import time
from fractions import Fraction
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_question(row):
    question = re.sub(r'<image\d+>', '', row['question']).strip()
    options = row.get('options', [])
    if options:
        question += '\nOptions:\n' + '\n'.join(
            f'{chr(65+i)}. {text}' for i, text in enumerate(options))
    return question + '\nSolve the problem step by step. Put your final answer in \\boxed{}. ' + (
        'For this multiple-choice question, put only the option letter in the box.'
        if options else 'Put only the final value or expression in the box.')


def extract_answer(text):
    # Preserve nested LaTeX braces such as \boxed{\frac{1}{2}}.
    boxes = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        start = match.end()
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            if depth == 0:
                boxes.append(text[start:pos].strip())
                break
    return boxes[-1] if boxes else None


def normalize(text):
    return re.sub(r'\s+', '', str(text)).replace('$', '').replace('\\left', '').replace('\\right', '')


def conservative_match(predicted, expected, multiple_choice=False):
    if predicted is None:
        return False
    p, e = normalize(predicted), normalize(expected)
    if multiple_choice:
        # Strip harmless wrappers, but never extract arbitrary A-E from prose.
        p = re.sub(r'^\\(?:text|mathrm)\{([A-Ea-e])\}$', r'\1', p)
        return p.strip('().').upper() == e.upper()
    if p == e:
        return True
    # Exact rational comparison only: no unsafe eval or fuzzy judge.
    def rational(s):
        s = re.sub(r'\\(?:dfrac|tfrac)', r'\\frac', s)
        s = re.sub(r'^\\frac\{(-?\d+)\}\{(-?\d+)\}$', r'\1/\2', s)
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', s):
            raise ValueError(s)
        return Fraction(s)
    try:
        return rational(p) == rational(e)
    except (ValueError, ZeroDivisionError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--max-pixels', type=int, default=1280*28*28)
    args = parser.parse_args()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = read_jsonl(args.dataset / 'questions.jsonl')
    assert len(rows) == 20 and len({r['id'] for r in rows}) == 20
    digest = hashlib.sha256((args.dataset/'questions.jsonl').read_bytes())
    for row in rows:
        digest.update((args.dataset / row['image']).read_bytes())
    config = {
        'model_path': str(args.model.resolve()), 'dataset_sha256': digest.hexdigest(),
        'max_new_tokens': args.max_new_tokens, 'max_pixels': args.max_pixels,
        'min_pixels': 256*28*28, 'dtype': 'bfloat16', 'attention': 'sdpa',
        'do_sample': False, 'prompt_version': 'boxed-options-v1',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'scoring': 'conservative exact/rational match, NOT official MathVision evaluator',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Configuration changed. Use a new --output directory.')
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    predictions_path = args.output / 'predictions.jsonl'
    previous = read_jsonl(predictions_path) if predictions_path.exists() else []
    done = {str(r['id']): r for r in previous if r.get('status') == 'ok'}
    model = processor = None
    if len(done) < len(rows):
        print('Loading local weights...', flush=True)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
            min_pixels=config['min_pixels'], max_pixels=args.max_pixels)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map='auto',
            attn_implementation='sdpa', local_files_only=True).eval()
        # Greedy decoding does not use sampling temperature, top_p or top_k.
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    for i, row in enumerate(rows, 1):
        if str(row['id']) in done:
            print(f'[{i}/20] ID={row["id"]}: already saved; skip', flush=True)
            continue
        result = {'id': str(row['id']), 'status': 'error'}
        inputs = output = new_ids = None
        try:
            with Image.open(args.dataset / row['image']) as im:
                image = im.convert('RGB')
            question = build_question(row)
            messages = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': question}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            new_ids = output[0, inputs.input_ids.shape[1]:]
            raw = processor.decode(new_ids, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            truncated = len(new_ids) >= args.max_new_tokens and int(new_ids[-1]) not in eos_ids
            result.update(status='ok', prompt=question, raw_output=raw,
                extracted_answer=extract_answer(raw), generated_tokens=len(new_ids),
                input_tokens=int(inputs.input_ids.shape[1]), truncated=truncated,
                inference_seconds=seconds,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
            done[str(row['id'])] = result
            print(f'[{i}/20] ID={row["id"]}: {len(new_ids)} tokens, {seconds:.1f}s, truncated={truncated}', flush=True)
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            print(f'[{i}/20] ID={row["id"]}: {result["error"]}', flush=True)
        finally:
            with predictions_path.open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                stream.flush()
            del inputs, output, new_ids
            if result['status'] != 'ok':
                torch.cuda.empty_cache()
    # Read labels only after generating: labels never enter the model prompt.
    answers = {str(r['id']):r for r in read_jsonl(args.dataset / 'answers.jsonl')}
    scored = []
    for row in rows:
        result = done.get(str(row['id']), {})
        truth = answers[str(row['id'])]
        scored.append({
            'id': row['id'], 'subject': truth['subject'], 'level': truth['level'],
            'expected_answer': truth['answer'], 'predicted_answer': result.get('extracted_answer'),
            'conservative_match': conservative_match(result.get('extracted_answer'), truth['answer'], bool(row['options'])),
            'status': result.get('status', 'error'), 'truncated': result.get('truncated'),
            'needs_manual_review': result.get('extracted_answer') is None or result.get('truncated', False),
            'generated_tokens': result.get('generated_tokens'), 'inference_seconds': result.get('inference_seconds'),
        })
    with (args.output/'scores.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
        writer.writeheader(); writer.writerows(scored)
    summary = {
        'total': len(rows), 'successful': len(done), 'errors': len(rows)-len(done),
        'conservative_matches': sum(r['conservative_match'] for r in scored),
        'conservative_accuracy': sum(r['conservative_match'] for r in scored)/len(rows),
        'truncated': sum(bool(r['truncated']) for r in scored),
        'extraction_failures': sum(r['status']=='ok' and r['predicted_answer'] is None for r in scored),
        'warning': 'Pilot only. All mismatches need review; not official scoring or a full benchmark result.',
    }
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary['errors']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
