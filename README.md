# HKU_Comp7404_26Fall_Open_Vision_Reasoner_NeurIPS_2025

English reproduction demo for **Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning**, NeurIPS 2025, prepared for HKU COMP7404.

This project evaluates released checkpoints and reproduces analyses from saved outputs. It does not repeat the paper's large-scale SFT or PPO training.

## View the demo

Clone or download this repository, then open **[demo/START_HERE.html](demo/START_HERE.html)** in your browser. It works offline. GitHub's file view shows HTML source; download or clone the files to view the interactive page.

The English landing page presents three separate experiments:

1. The original 100-question checkpoint evaluation.
2. The exploratory 20-question Effect of Output Budget follow-up.
3. Descriptive teacher/student reasoning-length analysis.

The [full English guide](demo/README.md) explains the package. Six saved OVR cases with embedded images are available in [the case viewer](demo/pipelines/ovr/demo/OVR_SHARED100_CASES.html). Original source-language team reports and raw records remain unchanged for provenance.

## Reproduce the saved results on CPU

Use Python 3.11 or 3.12 on macOS, Linux or Windows WSL2:

```bash
git clone https://github.com/Kaiser-yuan/HKU_Comp7404_26Fall_Open_Vision_Reasoner_NeurIPS_2025.git
cd HKU_Comp7404_26Fall_Open_Vision_Reasoner_NeurIPS_2025/demo
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python scripts/reproduce_english.py --verify-only
python scripts/reproduce_english.py
```

Installing dependencies requires internet access. Reproduction thereafter uses the saved responses and included tokenizer: no GPU, model downloads, new inference or paid API calls are required. The expected success marker is `ALL_OFFLINE_REPRODUCTION_VERIFIED`; each run writes a fresh output folder without modifying archived evidence.

## Recorded results

| Experiment | Result |
|---|---|
| Original fixed 100 questions, common official scorer | Qwen 28/100; VL-Rethinker 32/100; OVR-ColdStart 34/100; OVR-RL 31/100 |
| Separate output-budget follow-up, 20 questions | ColdStart 20% to 50%; RL 30% to 40% accuracy |
| Length-limit hits in the budget follow-up | ColdStart 55% to 0%; RL 75% to 20% |
| Descriptive reasoning lengths | Teacher 4,444; ColdStart 8,673; RL at least 10,969 tokens |

The original four-model runs use different generation settings. The budget follow-up uses one 16k trajectory per model/question and scores its 8k token prefix, rather than independent 8k and 16k requests. It remains separate from the original evaluation; no additional same-20-question Qwen comparison is included.

Observed performance is sensitive to output budget and answer finalization/extraction. Longer generation does not guarantee correctness, and these results do not establish that RL intrinsically reduces reasoning ability.

## Documentation and provenance

- [Step-by-step walkthrough](demo/docs/01_WALKTHROUGH.md)
- [Results and limitations](demo/docs/02_RESULTS_AND_LIMITS.md)
- [Sources and versions](demo/docs/03_SOURCES_AND_VERSIONS.md)
- [Output-budget experiment](demo/docs/04_OUTPUT_BUDGET.md)
- [Reasoning lengths](demo/docs/05_REASONING_LENGTHS.md)
- [Verified result summary](demo/reference_results/RESULTS.md)

The distribution includes original code, fixed questions and images, saved responses, archived official scoring code, environment records and SHA256 manifests. It excludes model weights, Python environments and credentials. Original upstream licenses and attribution are retained; no blanket license supersedes them. Code and presentation preparation used Codex assistance, while model runs were conducted by the project team.

Paper and official project: [Open Vision Reasoner](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner). Benchmark and official evaluator: [MathVision](https://github.com/mathllm/MATH-V).
