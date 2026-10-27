# Qwen MathVision pilot review

## Verified run

The uploaded cloud archive contains 20 successful responses, no execution errors,
the actual run configuration, environment package list, GPU information and raw scores.
The source question records and image bytes match the local fixed subset.
Original raw files are preserved unchanged.

- Original conservative final-answer matches: 4/20 (20%).
- Reviewed final-answer matches: 5/20 (25%).
- Review accepts answer content matching a unique multiple-choice option.
- This is a small pilot and not official MathVision benchmark accuracy.
- Matching final answers does not establish correct or faithful reasoning.
- Total measured per-question inference time: 251.99 s (about 4.20 min), excluding
  model loading and some setup. Total generated tokens: 16,992.
- Median per-question inference time: 7.81 s. Prompt lengths and output lengths vary.

## Material review findings

| ID | Raw output or issue | Reference | Decision |
|---|---|---|---|
| 1324 | Boxed `80-2\pi` instead of a letter | B, whose content is `80-2 \pi` | Accept final answer by unique option-content match. The explanation incorrectly assumes an equilateral triangle; do not present this as fully correct reasoning. |
| 51 | Entire response `10`, no box | 3 | Resolve extraction to 10; answer remains incorrect. |
| 1542 | Enumerates a large artificial numeric grid until the 4096-token cap | 3 | No complete final answer; truncated. |
| 2336 | Repeats `12-8\sqrt{3}/3` many times until the cap | C | No complete final answer; truncated. |

Other explicit final answers either match the reference (302, 1414, 2536, 2760)
or remain incorrect. Review final-answer matches independently from reasoning quality.
The three original extraction failures overlap the truncation cases: IDs 51, 1542, 2336;
they are not five independent failures.

## Next experiment

Keep this 4096-token baseline intact. Evaluate OVR-ColdStart and OVR-RL with the
same 20 questions, precision, image processing policy and prompt protocol first.
Do not assume increasing the cap fixes repetitive degeneration. If testing an 8192-token
cap or changed decoding later, save a separate run and identify it as an additional
experiment. Use the same review rules for all checkpoints.

The installed processor emitted a fast-processor default warning. The current result
is valid as a pilot under that setup; cross-checkpoint comparisons must preserve and
record the processor settings rather than silently changing them.

## Reproduce the offline review

From the demo root:

```bash
python scripts/review_mathvision_pilot.py \
  --run results/qwen_mathvision20_4096 \
  --dataset data/dataset_test \
  --output results/qwen_review_reproduced
```

This reads saved predictions and requires no GPU or model weights. Its helper module
imports PyTorch only for inference, so the offline review itself uses Python stdlib.
See reviewed_scores.csv for per-question decisions and review_summary.json for totals.

## Remaining limitations

The sample is 20 uniformly sampled test questions, not the official testmini; 13/20
are levels 4 or 5. No confidence about full-test performance follows from this subset.
OVR checkpoint runs, official evaluation and human team verification remain pending.
