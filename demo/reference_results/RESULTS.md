# Verified offline reproduction results

Saved responses were re-scored; no new inference or training was performed. Distributed source files remain unchanged.

| Model | Official correct | Limit hits | Output cap | Decoding |
|---|---:|---:|---:|---|
|qwen|28/100|2|4096|greedy|
|vl_rethinker|32/100|0|4096|greedy|
|coldstart|34/100|48|8192|sampling T0.6 p0.95 k40|
|rl|31/100|70|8192|sampling T0.6 p0.95 k40|

The original 100-question runs share inputs and scoring, but differ in generation budget, decoding and backend. This is not a controlled capability ranking.

## Separate output-budget follow-up

| Model | Correct at 8k | Correct at 16k | Limit hits at 8k | Limit hits at 16k |
|---|---:|---:|---:|---:|
| ColdStart | 4/20 | 10/20 | 11/20 | 0/20 |
| RL | 6/20 | 8/20 | 15/20 | 4/20 |

Each 8k view is the token prefix of the same 16k trajectory. The exploratory follow-up does not replace the original 100-question scores. No additional same-20-question Qwen comparison is included.

Observed scores depend on output budget and answer finalization/extraction. More tokens helped some responses finish, but four RL responses still hit the 16k cap; longer generation did not guarantee correctness. These observations do not establish that RL intrinsically reduces reasoning ability.

## Descriptive reasoning lengths

Mean reasoning lengths are 4444.473 tokens for 1000 teacher traces, 8672.9 for ColdStart, and at least 10968.75 for RL on the 20 student questions. Four RL traces are right-censored; teacher stopping status is unknown. Task and modality differences preclude capability-transfer conclusions from length alone.
