# Fixed MathVision pilot subset

20 questions sampled without replacement from the 3,040-row official `test` split.
Seed: 7404. This is NOT the official `testmini` split.
Repository: https://github.com/mathllm/MATH-V
Revision: a1f5cc3add200c0cd080fad463e500f44ef1fb41

- questions.jsonl: model inputs, without answers or solutions.
- answers.jsonl: labels and analysis metadata, used only after generation.
- samples_full.jsonl: original complete records for inspection.
- images/: original image bytes, not cropped or re-encoded.
- manifest.json: source revision, sampling indices, IDs and image SHA256 values.
- LICENSE: original MIT license.

Image paths are relative to this directory. The inference script removes original
<imageN> markers and passes the composite original image to the processor once.
Options are rendered as A, B, C, etc. Never pass label or solution fields to the model.
These 20 questions support debugging and paired comparison, not a full benchmark claim.
