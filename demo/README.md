# Open Vision Reasoner Reproduction Demo

This English edition combines our saved Qwen, VL-Rethinker, OVR-ColdStart and OVR-RL experiments, their inference code, fixed MathVision inputs and reproducible offline analyses. Open **START_HERE.html** for the presentation view, or follow the commands below to reproduce the saved results on a CPU.

The package evaluates released checkpoints. We did not repeat the paper's SFT or PPO training. Original questions, images, model outputs, code and source hashes are preserved. Original team documents and historical records remain in their source language; English explanations cover the primary demonstration workflow.

## Quick start

Use Python 3.11 or 3.12 on macOS, Linux or Windows WSL2. Run from this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python scripts/reproduce_english.py --verify-only
python scripts/reproduce_english.py
```

Dependency installation needs internet access. Once dependencies are installed, reproduction uses saved outputs only: no GPU, model downloads, inference requests or paid APIs. The success marker is **ALL_OFFLINE_REPRODUCTION_VERIFIED**. A fresh `outputs/<timestamp>/` folder contains `RESULTS.md`, `VERIFICATION.json` and detailed tables; OVR also creates its own isolated reproduction folder, referenced in the verification file.

## Experiments remain separate

| Experiment | Data and purpose | Recorded result |
|---|---|---|
| Original checkpoint evaluation | 100 fixed MathVision questions; shared inputs and official rescoring | Qwen 28/100; VL-Rethinker 32/100; ColdStart 34/100; RL 31/100 |
| Effect of Output Budget | Exploratory, difficulty-balanced 20-question follow-up for ColdStart and RL | ColdStart 20% to 50%; RL 30% to 40% |
| Reasoning-length analysis | 1,000 released teacher traces and the 40 existing student traces | Mean reasoning: 4,444; 8,673; at least 10,969 tokens |

The 20-question follow-up is not combined with original Qwen answers to create a new same-20-question comparison. No added GPU launch entry point is included.

## What each reproduction step does

1. Verify distributed SHA256 hashes, IDs, original image bytes, prompts and labels.
2. Merge the disjoint Qwen 20- and 80-question runs, then replay the original Qwen/VL answer review.
3. Re-run the archived, unmodified MathVision official scorer on both OVR 100-question runs and the budget follow-up.
4. Recount the saved teacher and student reasoning with the same tokenizer.
5. Apply the same official scorer to all four original 100-question model runs, preserving all historical scores separately.
6. Verify unchanged source files and write an English result summary.

The original Qwen/VL custom review gives 28/100 and 32/100; the official scorer happens to produce the same totals. This does not make the two scoring implementations equivalent.

## Main interpretation

Observed performance is sensitive to output budget and answer finalization/extraction. Additional tokens helped some responses finish, but four RL responses still hit the 16k limit and longer generation did not guarantee a correct answer. These observations do not establish that RL intrinsically reduces reasoning ability.

The original four-model runs used different generation protocols: Qwen/VL used greedy decoding with a 4,096-token cap in Transformers; OVR used sampling with an 8,192-token cap in vLLM. Shared questions and a common scorer do not establish a controlled four-way capability ranking.

## Read and present

- `START_HERE.html`: English presentation overview; works offline.
- `docs/01_WALKTHROUGH.md`: commands, purpose, outputs and expected results for each step.
- `docs/02_RESULTS_AND_LIMITS.md`: interpretation and poster-ready wording.
- `docs/03_SOURCES_AND_VERSIONS.md`: models, revisions, provenance and licenses.
- `docs/04_OUTPUT_BUDGET.md`: detailed 8k/16k result and case notes.
- `docs/05_REASONING_LENGTHS.md`: teacher/student length definitions and limitations.
- `pipelines/ovr/demo/OVR_SHARED100_CASES.html`: six archived 100-question OVR cases with embedded images and saved responses; its interface is already English.
- `pipelines/qwen_vl/results/model_comparison100/CASEBOOK.md`: original Qwen/VL paired cases, in English.
- `reference_results/`: verified English summary and original numeric tables.

## Archive structure

`pipelines/qwen_vl/` and `pipelines/ovr/` retain both teams' original code and evidence, including historical instructions. `sources/team_reports/` retains the original Word reports. `archive/original_overviews/` retains the Chinese top-level introductions. Use this README and the English walkthrough as the current demonstration instructions.

`provenance/source_inventory.json` identifies the original source files; `provenance/english_edition.json` identifies the English edition's base archive. `MANIFEST.sha256.json` covers distributed files. Model weights, virtual environments and credentials are not distributed.

Code and presentation preparation used Codex assistance; actual model runs were conducted by the project team. Preserve the upstream data, image and evaluator attributions when presenting the work.
