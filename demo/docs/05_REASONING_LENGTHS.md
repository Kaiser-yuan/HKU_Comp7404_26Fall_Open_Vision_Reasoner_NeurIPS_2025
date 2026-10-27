# Teacher and student reasoning lengths

This analysis compares 1,000 fixed demonstrations from the released OVR cold-start dataset with the existing 20-question visual outputs from each student checkpoint. No new generation or training is performed.

| Group | Traces | Mean reasoning tokens | Median reasoning tokens | Mean response tokens | Known generation truncation |
|---|---:|---:|---:|---:|---|
| Teacher demonstrations | 1000 | 4444.473 | 2522.5 | 4941.407 | Unknown |
| OVR-ColdStart | 20 | 8672.9 | 9624.5 | 8964.15 | 0/20 |
| OVR-RL | 20 | At least 10968.75 | 11508.5 observed | 11214.2 observed | 4/20 |

RL produces longer observed reasoning in 17 of 20 paired questions. Its observed mean is about 26.5% higher than ColdStart's. On the 16 questions where both reasoning sections close, mean lengths are approximately 8011 and 9615 tokens; that subset is completion-selected and is not an unbiased replacement for all 20 questions.

## Sampling and token counting

The teacher source is revision `e377de8ec93f9d1ff48744dcdc8c51c4801c6da2`. Five verified original shards contain 1,637,924 rows, totaling 11,784,309,108 bytes. The fixed global sample selects 1,000 rows uniformly without replacement, seed 20261026, before inspecting responses. Shard counts are 24, 174, 408, 341 and 53. Selection is not based on length, correctness or reflection keywords.

All saved text is re-tokenized with the same verified Qwen tokenizer. Reasoning excludes think delimiters and the final-answer section. The students' opening think delimiter is supplied by the checkpoint template. All 1,000 sampled teacher responses have segmentable reasoning; their raw source records are included for recounting.

Four RL traces hit the 16k cap without closing reasoning. Their lengths are right-censored lower bounds. Teacher finish metadata is unavailable, so its truncation rate is unknown, not zero. These different task, modality and generation conditions make the teacher/student comparison descriptive.

Longer reasoning text alone does not establish stronger reasoning, faithful chains of thought, actual visual attention or transfer of the teacher's capabilities. Use the original 100-question results, output-budget follow-up and length analysis as separate evidence.
