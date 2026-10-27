# Reproducing the teacher ColdStart RL length comparison

The teacher reference is a frozen uniform sample of 1,000 rows from Kangheng/OVR-Dataset-ColdStart at revision e377de8ec93f9d1ff48744dcdc8c51c4801c6da2. Source: https://huggingface.co/datasets/Kangheng/OVR-Dataset-ColdStart . The source dataset card declares Apache 2.0; its original README is included under workstreams/teacher_lengths/teacher_source_README.md. Our student traces are the saved 20-question MathVision evaluation, one generation per model and question.

The full 11.8GB published corpus is retained on the owner's Mac under data/cache/teacher_length_v1/ and excluded from this sharing ZIP. The ZIP includes the exact 1,000 selected teacher question/answer pairs, all 40 saved student responses, the common tokenizer, per-record measurements, selection/provenance and source-file SHA receipts. It contains no model weights, environment, token or SSH credential.

Extract this ZIP, verify each member against SHA256SUMS.json, and use Python 3.11. Install requirements-trace-lengths.txt in a local environment. Run:

    python -m ovr.trace_lengths --from-sample

This re-tokenizes the included teacher sample and student responses on CPU. It does not download data, run models, or use GPUs. Selection indices, revision and upstream file identities are checked. The initial 1,000-row sampling protocol is byte-identical to commit eb4949f, made before teacher responses were available.

To repeat the original full-file acquisition instead, run scripts/download_teacher_resumable.py, which downloads only the pinned public data with resumable ranges and whole-file SHA checks, followed by:

    python -m ovr.trace_lengths --teacher

Teacher text tasks differ from student visual questions. Teacher generation stop metadata is unavailable; four student RL trajectories hit the 16k cap. Lengths describe recorded output behavior and do not establish capability transfer or superior reasoning. The 16-question subset where both students completed reasoning is separately identified and is completion-selected.
