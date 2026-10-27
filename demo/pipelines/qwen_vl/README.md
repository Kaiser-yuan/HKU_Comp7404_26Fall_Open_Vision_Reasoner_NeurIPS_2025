# Visual reasoning checkpoint comparison demo

This package contains the actual Qwen and VL-Rethinker runs on 100 fixed MathVision
test questions, the inference and setup code, and a reproducible offline analysis.
It performs inference using released weights, not model training. OVR checkpoint
experiments are handled separately by teammates and are not included as completed runs.

## Start here for an exhibition

1. Read results/model_comparison100/REPORT.md for the results and limitations.
2. Open results/model_comparison100/CASEBOOK.md for paired improvement, regression,
   shared-correct and shared-failure examples. Nearby images/ contains original figures.
3. Open paired_results.csv and metrics.csv for the complete comparison.
4. Demonstrate offline re-analysis with:

```bash
bash scripts/06_reproduce_analysis.sh
```

Offline analysis requires Python 3.10+ only; no GPU, model weights, network connection,
API key or paid service is needed. If your system uses python3, set:

```bash
PYTHON=python3 bash scripts/06_reproduce_analysis.sh
```

Each invocation creates a new results/reproduced.* directory, keeping the included
raw and derived results unchanged. The analysis re-merges Qwen's 20+80 runs, verifies
matching configurations and IDs, re-applies the same answer review to both models,
and regenerates the paired table, report and casebook.

## Results at a glance

| Metric | Qwen2.5-VL-7B-Instruct | VL-Rethinker-7B |
|---|---:|---:|
| Successful questions | 100 | 100 |
| Original conservative matches | 26 | 30 |
| Unified final-answer matches | 28 | 32 |
| Truncated outputs | 2 | 0 |

Paired outcomes: 16 shared correct, 16 VL improvements, 12 VL regressions and 56 shared
incorrect. These are subset final-answer matches under our stated rules, not official
benchmark accuracy, general model superiority, or validation of the reasoning traces.

## Package layout

- scripts/: setup, checkpoint download, smoke test, inference, sample extraction,
  merge, review, comparison and backup tools.
- scripts/cloud_executed/: exact Python scripts recovered from the cloud backup.
- data/: original20, additional80 and their fixed100 union, original images, labels,
  repository revision, sampling rules, licenses and hashes.
- results/qwen_mathvision20_4096 and qwen_mathvision_extra80_4096: unchanged raw Qwen runs.
- results/vl_rethinker_mathvision100_4096: unchanged raw VL run.
- results/qwen_mathvision100_merged: derived union with merge provenance.
- results/qwen100_review and vl_rethinker100_review: separate standardized reviews.
- results/model_comparison100: tables, summary, report and casebook.
- logs/: actual cloud package list and GPU information.
- provenance/: VL checkpoint commit and SHA256 values of the original backup archives.
- docs/: method notes, command archive, observed status and AI-use declaration.
- SHA256SUMS: checksums for every distributed file except the checksum file itself.

## Re-run GPU inference (optional)

Tested environment: AutoDL Linux, Python 3.12, PyTorch 2.8.0+cu128, Transformers 4.57.1,
NVIDIA RTX 6000D (about 83 GiB visible memory). See logs/requirements.txt for actual
packages. No model weights or credentials are distributed. Use your own account for
any gated checkpoint and accept its conditions yourself.

The original convenience scripts expect /root/autodl-tmp/ovr. Upload this package to
your server and place its contents there. Preserve existing files before updating
scripts. Do not blindly run every command in the historical archive.

```bash
cd /root/autodl-tmp/ovr
bash scripts/01_check_environment.sh
bash scripts/02_install_dependencies.sh
bash scripts/03_download_model.sh Qwen/Qwen2.5-VL-7B-Instruct /root/autodl-tmp/ovr/models/qwen
python scripts/qwen_smoke_test.py
```

Download VL with its recorded exact revision when reproducing this run:

```bash
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="TIGER-Lab/VL-Rethinker-7B",
    revision="5439d11fc9fa1474ab015104e6604223f165ccf5",
    local_dir="/root/autodl-tmp/ovr/models/vl-rethinker", max_workers=1)
PYTHON
```

```bash
python -u scripts/run_mathvision_test_v3.py \
 --model /root/autodl-tmp/ovr/models/vl-rethinker \
 --dataset /root/autodl-tmp/ovr/data/dataset_test100 \
 --output /root/autodl-tmp/ovr/results/vl_rethinker_new_run_4096 \
 --max-new-tokens 4096
```

Use a new output directory for new experiments. Same-settings reruns skip saved
successful IDs and retry errors. If interrupted during JSONL writing, back up and
inspect any partial last line before resuming. Do not alter generation settings to
make the score resemble a published result.

## Comparison protocol and limits

Images and questions are identical; user-prompt construction, BF16, SDPA, greedy
sampling, pixel limits and 4096-token cap match. Each checkpoint uses its own template
and processor. The fast processor was selected by default in the tested environment.
The v3 VL runner explicitly uses KV cache; the Qwen run did not record this flag.
Timing is descriptive rather than a controlled speed comparison. VL uses fewer output
tokens in these runs; that alone must not be interpreted as better reasoning.

The reviewer extracts boxed answers, allows unambiguous single-value full responses,
and accepts unique matching option content and simple rational equivalents. General
symbolic grading is not implemented. Remaining mismatches need team review. The 100
questions were sampled in two fixed stages without using model outcomes; this is not
official testmini or the full test benchmark. Finite output budgets can disadvantage
long-reasoning models. Correct answers can coexist with flawed reasoning.

## Attribution and disclosure

Read docs/AI_USAGE.md and cite all borrowed figures and data in the poster. Source
MathVision images retain their original MIT license in each data directory; we do not
claim authorship of those images. Code was developed with Codex assistance and actual
runs were performed by the project team. OVR results are not fabricated or inferred.

- Qwen: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- VL: https://huggingface.co/TIGER-Lab/VL-Rethinker-7B
- OVR paper: https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner
- MathVision: https://github.com/mathllm/MATH-V
- Data revision: a1f5cc3add200c0cd080fad463e500f44ef1fb41

Qwen's exact original checkpoint revision was not recorded at download time; retain
that reproducibility limitation. Its actual outputs and configs are supplied unchanged.
