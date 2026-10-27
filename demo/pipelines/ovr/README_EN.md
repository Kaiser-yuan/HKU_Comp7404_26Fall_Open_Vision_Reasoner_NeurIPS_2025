# OVR experiments and offline reproduction

This folder retains the teammate's original OVR code and evidence. It covers the original 100-question ColdStart/RL evaluation, a separate 20-question 8k/16k follow-up, and descriptive teacher/student reasoning lengths. We evaluate released checkpoints; we do not repeat SFT or RL training.

For the English demonstration, open [START_HERE.html](../../START_HERE.html). For commands, use the root [README](../../README.md) and [walkthrough](../../docs/01_WALKTHROUGH.md). Run the English wrapper from the package root:

```bash
python scripts/reproduce_english.py
```

The original OVR 100-question results are ColdStart 34/100 and RL 31/100, with 48 and 70 length-limit hits. The separate exploratory budget follow-up gives 4/20 to 10/20 for ColdStart and 6/20 to 8/20 for RL. Each 8k view is the prefix of one 16k trajectory; this is not a new same-20-question Qwen comparison. Detailed English notes are in the root docs folder.

The local case viewer at `demo/OVR_SHARED100_CASES.html` already has an English interface. Original Chinese documents and historical cloud instructions remain unchanged for provenance. They are not required for the current CPU demonstration.
