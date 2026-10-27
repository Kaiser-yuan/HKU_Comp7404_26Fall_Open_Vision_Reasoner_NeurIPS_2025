# Results and interpretation

## Scope

This is a released-checkpoint inference and analysis reproduction. We did not repeat the paper's large-scale language SFT or multimodal PPO training. Our fixed 100-question subset is not the full 3,040-question MathVision benchmark.

## Original checkpoint evaluation

| Model | Official correct | Length-limit hits | Output cap | Decoding and backend |
|---|---:|---:|---:|---|
| Qwen2.5-VL-7B-Instruct | 28/100 | 2/100 | 4096 | Greedy, Transformers SDPA |
| VL-Rethinker-7B | 32/100 | 0/100 | 4096 | Greedy, Transformers SDPA |
| OVR-ColdStart | 34/100 | 48/100 | 8192 | Sampling, vLLM 0.11.0 |
| OVR-RL | 31/100 | 70/100 | 8192 | Sampling, vLLM 0.11.0 |

All use the same original 100 questions and images. Shared official scoring does not remove differences in decoding, generation budget, backend or checkpoint-specific templates. The table is not a controlled four-way capability ranking.

## Separate output-budget follow-up

| Model | Accuracy 8k to 16k | Limit-hit rate 8k to 16k | Incorrect to correct | Correct to incorrect |
|---|---|---|---:|---:|
| ColdStart | 20% to 50% | 55% to 0% | 6 | 0 |
| RL | 30% to 40% | 75% to 20% | 3 | 1 |

The follow-up uses 20 previously evaluated questions, balanced across difficulty levels. Each model/question has one 16k trajectory, with its 8k prefix as the paired comparison. It is exploratory and does not replace the original 100-question scores. No new same-20-question Qwen comparison is included.

Additional budget helped some responses finish and become scorer-correct. Four RL responses still hit the 16k cap, and longer generation did not guarantee correctness. Observed performance is sensitive to output budget and answer finalization/extraction; the data do not establish that RL intrinsically reduces reasoning ability.

Natural stopping, a complete boxed answer, closed reasoning tags and an official correct score are separate measurements. Two RL 8k views were scored correct without a complete boxed answer. At ID598, an 8k scorer-correct intermediate candidate becomes an incorrect final boxed answer at 16k; this is not evidence of a completed correct final answer being overturned.

## Descriptive reasoning-length analysis

Using one tokenizer, mean reasoning lengths are 4444.473 tokens for 1,000 released teacher traces, 8672.9 for ColdStart and at least 10968.75 for RL on the 20 student questions. RL has four right-censored traces. Teacher generation stopping metadata is unavailable, so its true truncation rate is unknown.

Teacher text tasks and student image tasks differ. Longer text does not prove better reasoning, faithful CoT, actual visual attention or causal capability transfer. The paired 16-question subset with completed reasoning in both students is completion-selected and potentially biased.

## Poster-ready wording

We evaluated released visual reasoning checkpoints on a fixed MathVision subset and conducted a separate exploratory output-budget follow-up. On 20 difficulty-balanced questions, each OVR checkpoint generated one trajectory with a maximum of 16,384 output tokens; its first 8,192 tokens formed the paired lower-budget view. Official accuracy increased from 20% to 50% for ColdStart and from 30% to 40% for RL. Length-limit hits decreased from 55% to 0% and from 75% to 20%, respectively. Additional tokens helped some responses complete, but four RL trajectories remained truncated and longer generation did not guarantee correctness. These observations suggest sensitivity to output budget and answer finalization/extraction, rather than establishing that RL intrinsically degrades reasoning.
