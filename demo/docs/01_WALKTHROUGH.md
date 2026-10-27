# Offline reproduction walkthrough

This workflow reproduces analyses from the exact saved responses. It does not generate new responses or train a model. The original 100-question evaluation, later budget follow-up and descriptive length analysis remain separate.

## 1 Install dependencies and check inputs

Follow the environment commands in the root README, then run:

```bash
python scripts/reproduce_english.py --verify-only
```

This checks the distribution hashes and confirms that the original four model runs share the same 100 IDs, order, prompts, image bytes and reference labels. It also verifies that Qwen's original 20 questions and additional 80 are disjoint. Expected marker: `INPUT_AND_FILE_CHECKS_PASSED`.

## 2 Reproduce all saved analyses

```bash
python scripts/reproduce_english.py
```

The runner creates a new `outputs/<timestamp>/` folder. Each stage writes a log there. Official scoring and tokenization can take a few minutes; they are CPU operations. The underlying original reproduction script is retained unchanged; the English wrapper adds an English output summary.

## 3 Merge Qwen's original 20 plus 80

The merge checks IDs and generation settings before combining predictions. It does not add summary percentages or merge duplicate questions. Expected output: 100 unique predictions in `qwen_vl/merged/`, with the original conservative match count of 26/100. Inspect `qwen_merge.log` and the merge provenance.

Only the disjoint original 20- and 80-question runs are merged. The later budget20 questions overlap the original 100 and must not be appended to make 120 observations.

## 4 Replay the original Qwen/VL review

The archived review uses complete boxed answers, unambiguous whole-response answers, unique option-content matching and simple rational equivalence. It does not use a model judge. Expected reviewed counts: Qwen 28/100 and VL-Rethinker 32/100.

Expected paired categories: 16 both correct, 16 VL improvements, 12 VL regressions and 56 both incorrect. Outputs include `qwen_vl/comparison/metrics.csv`, `paired_results.csv`, `REPORT.md` and `CASEBOOK.md`. This is the historical custom review, distinct from official rescoring.

## 5 Re-score the OVR 100-question evaluation

The archived unmodified MathVision evaluator receives the saved responses, never the original files as writable targets. Expected official counts are ColdStart 34/100 and RL 31/100. Length-limit hits are 48/100 and 70/100.

OVR creates a separate `pipelines/ovr/results/reproduced/<timestamp>/` folder. Its report paths are under `results/reports/cloud_vllm_shared1008192_v1/pilot/`. The top-level verification file records the exact location. Per-difficulty and per-subject tables are available there.

Official correctness is a scorer verdict. The scorer can extract candidates from unfinished text, so a correct score does not necessarily mean a naturally finished answer or a valid reasoning chain.

## 6 Reproduce Effect of Output Budget

The later follow-up uses 20 questions, four per difficulty, selected with seed 20261025. Seven engineering-development overlap IDs are excluded. Selection uses IDs and difficulty, not previous correctness or truncation.

Each model produced one trajectory per question with a 16,384-output-token cap. Its first 8,192 token IDs form the paired lower-budget view: 40 actual generations and 80 scored views, not 80 independent generations. The lower-budget views need not equal corresponding outputs from the original 100-question experiment.

Expected official counts and length-limit hits:

| Model | Correct at 8k | Correct at 16k | Limit hits at 8k | Limit hits at 16k |
|---|---:|---:|---:|---:|
| ColdStart | 4/20 | 10/20 | 11/20 | 0/20 |
| RL | 6/20 | 8/20 | 15/20 | 4/20 |

Outputs: `comparison.json`, `summary.csv`, `paired.csv` under the isolated OVR reproduction's `results/reports/cloud_vllm_budget20_prefix_v1/`. The summary distinguishes correctness, natural stopping, complete boxed answers, closed reasoning tags and output length. Token-boundary times are client chunk-arrival approximations, not latency from independent 8k runs.

Do not add an original Qwen20 comparison to this follow-up or replace the original 100-question scores.

## 7 Recount teacher and student reasoning

The package includes 1,000 fixed teacher traces and the 40 saved student trajectories, plus the common tokenizer. No teacher dataset re-download is needed.

Expected mean reasoning lengths: teacher 4444.473, ColdStart 8672.9 and RL 10968.75 tokens. Four RL reasoning traces remain right-censored, so the RL mean is an observed lower bound. Teacher stopping metadata is unavailable. Different tasks and modalities prevent a causal teacher-to-student capability claim from length alone.

## 8 Re-score all four original 100-question runs uniformly

The same official scorer is applied separately to saved Qwen, VL-Rethinker, ColdStart and RL outputs. Expected counts are 28, 32, 34 and 31 out of 100. See `official_four_models/metrics.csv`, `paired_correctness.csv` and per-model official JSONL files.

This unifies scoring, not generation. Qwen/VL use 4k greedy Transformers outputs; OVR uses 8k sampled vLLM outputs. Treat the table as descriptive and retain these differences in any poster.

## 9 Check completion and present cases

Expected marker: `ALL_OFFLINE_REPRODUCTION_VERIFIED`. `VERIFICATION.json` confirms no new model generations, no GPU use, matching inputs and unchanged distributed source files. The original OVR verification also checks byte-identical per-question scores and key reports.

Open `START_HERE.html` for the English overview, then the linked six-case OVR viewer or Qwen/VL casebook. Cases include improvements, regressions and failures. Statements such as "let me look again" are response text, not proof of a new visual operation.
