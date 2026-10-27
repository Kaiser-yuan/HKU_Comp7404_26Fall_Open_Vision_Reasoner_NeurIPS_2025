# Effect of Output Budget

This follow-up asks whether the low observed scores were partly affected by responses hitting the output cap before producing a final answer. It is separate from the original 100-question evaluation.

## Recorded results

| Model | Cap | Official correct | Wilson 95% interval | Limit hits | Mean output tokens |
|---|---:|---:|---|---:|---:|
| ColdStart | 8192 | 4/20 (20%) | 8.1% to 41.6% | 11/20 (55%) | 6742.1 |
| ColdStart | 16384 | 10/20 (50%) | 29.9% to 70.1% | 0/20 (0%) | 8964.15 |
| RL | 8192 | 6/20 (30%) | 14.5% to 51.9% | 15/20 (75%) | 7338.1 |
| RL | 16384 | 8/20 (40%) | 21.9% to 61.3% | 4/20 (20%) | 11214.2 |

ColdStart has six incorrect-to-correct transitions and no correct-to-incorrect transitions. RL has three incorrect-to-correct transitions and one correct-to-incorrect transition. All views contain nonempty text; natural completion is not equivalent to correctness.

## Protocol

Twenty questions are selected from the original 100, with four per difficulty level and seed 20261025. Selection uses IDs and difficulty only, excluding seven earlier engineering-development overlap questions. The subset has already been evaluated and is not an independent holdout.

Each checkpoint generates a single trajectory per question, capped at 16,384 output tokens. The first 8,192 token IDs provide the paired lower-budget view. This produces 40 actual generations and 80 scored views. The 8k views are not separate requests and need not reproduce outputs from the original 100-question run.

Both OVR models retain the recorded revisions, images, prompts and official templates: BF16, vLLM 0.11.0, temperature 0.6, top_p 0.95, top_k 40, repetition_penalty 1 and seed 42. The context cap is 24,576. The archived unmodified MathVision scorer is used for all views.

## Cases

- ColdStart ID2911: the 8k prefix stops during derivation; the 16k response naturally completes with boxed 384 and is scored correct.
- RL ID598: the 8k view has an unfinished candidate B that the official scorer accepts; the 16k response naturally ends with an incorrect boxed E. Do not call the earlier candidate a completed correct answer.
- RL IDs334, 612, 1118 and 2816: all still hit the 16k limit. More budget did not ensure completion.

Two RL 8k views have no complete boxed answer but receive correct scores from text candidates. Finalization and scoring therefore need separate reporting.

## Conclusion

The observations support sensitivity to output budget and answer finalization/extraction. Additional tokens help some responses finish, but they do not guarantee a correct answer or eliminate RL truncation. They do not establish that RL intrinsically reduces reasoning ability. The original ColdStart 34/100 and RL 31/100 results remain unchanged.
