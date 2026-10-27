# MathVision combined100

Source: official MATH-V `test` split, 3040 questions.
Revision: a1f5cc3add200c0cd080fad463e500f44ef1fb41
This directory contains 100 questions and original images; it is NOT official testmini.
The original 20 IDs are retained; 80 new IDs were drawn without replacement from the
remaining 3020 rows using seed 740480. No model outputs, scores, subjects or difficulty
labels were used to choose the new rows. The two-stage selection is documented in manifest.json.

questions.jsonl contains no answers; answers.jsonl is for post-generation scoring.
Images are relative to this directory. samples_full.jsonl preserves source records.
Do not send answer or solution fields to the model. Keep the previous 20-question result
unchanged. Use identical prompt, precision, image processing and generation settings for
additional questions and all model checkpoints before combining results.
