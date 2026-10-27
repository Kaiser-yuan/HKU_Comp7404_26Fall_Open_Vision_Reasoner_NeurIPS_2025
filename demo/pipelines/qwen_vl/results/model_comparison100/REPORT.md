# Qwen and VL-Rethinker on 100 fixed MathVision questions

## Main result

Both models completed all 100 identical questions without execution errors. Under one
post-generation answer extraction and equivalence protocol, Qwen matches 28 reference
answers and VL-Rethinker matches 32. This is a 4-percentage-point difference on this
particular subset, not evidence of a general advantage or an official benchmark score.
The original conservative scorer remains unchanged at 26 and 30 respectively.

| Measure | Qwen | VL-Rethinker |
|---|---:|---:|
| Questions completed | 100 | 100 |
| Original conservative matches | 26 | 30 |
| Unified final-answer matches | 28 | 32 |
| Truncated responses | 2 | 0 |
| Original answer extraction failures | 4 | 2 |
| Total output tokens | 57,334 | 49,720 |
| Sum of measured question inference time | 853.56 s | 741.92 s |

## Paired outcomes

- Both final answers match: 16 questions.
- Qwen incorrect, VL-Rethinker correct: 16 questions.
- Qwen correct, VL-Rethinker incorrect: 12 questions.
- Neither final answer matches: 56 questions.

A higher aggregate score does not mean improvement on every question. Preserve and
discuss both improvements and regressions. See paired_results.csv for all IDs.

## Scoring and interpretation

The same saved-output review script is applied to both models. It extracts the last
complete boxed answer, accepts a whole response only if it is a simple unambiguous
answer, and maps an answer to an option only when it uniquely matches that option's
content. Exact rational equivalence is supported; general symbolic equivalence is not.
It is not the official MathVision evaluator. Remaining mismatches should be inspected
by the team before a final poster claim. No labels or solutions enter inference prompts.

Qwen gains matches at IDs 1324 and 2725 by option-content equivalence. VL-Rethinker gains
matches at IDs 51 and 272 by recovering complete single-number responses without boxes.
Correct final answers do not certify correct reasoning; e.g., Qwen ID1324 assumes an
equilateral triangle without justification despite arriving at the reference expression.

## Shared settings and limitations

All compared IDs, prompts and images were checked. Recorded settings match for BF16,
SDPA, greedy decoding, pixel limits, Transformers 4.57.1, PyTorch 2.8.0+cu128 and a
4096-token output cap. Each model uses its own downloaded chat template/processor;
identical pixel limits do not prove that all preprocessing details are identical.
VL-Rethinker explicitly enables KV caching; Qwen's original run did not record that
flag explicitly. Output lengths also differ, so total time does not establish a
controlled model-speed advantage. The cap can constrain a reasoning model and this
is a shared-budget comparison, not each author's complete published evaluation.

The dataset is a two-stage fixed random sample (20 plus 80 from the remaining pool),
not official testmini. Results must not be tuned to match the paper's reported score.
No claim about OVR training or causal behavior transfer follows from these two models.
OVR-ColdStart/RL results are to be provided by teammates under aligned protocols.

## Reproducibility

Raw predictions, initial scores, configs, exact executed scripts and environment
logs are preserved in the demo. Derived reviews and this comparison are separate.
See comparison_summary.json, metrics.csv and CASEBOOK.md. Run the commands in
scripts/06_reproduce_analysis.sh from a demo folder to reproduce the offline analysis
without GPU access or model weights.
