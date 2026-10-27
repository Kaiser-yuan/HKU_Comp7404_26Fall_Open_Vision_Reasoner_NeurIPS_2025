# Historical setup plan

VL inference has since completed; use README.md and results/model_comparison100/REPORT.md for current status. The following preserves earlier setup guidance.

# Qwen baseline and VL-Rethinker comparison

The user handles Qwen and VL-Rethinker. Teammates handle OVR-ColdStart and OVR-RL.
All checkpoints should be evaluated on the same 100 question IDs and coordinated
image/prompt/decoding/scoring settings. No messages have been sent to teammates.

## Qwen baseline verified from uploaded raw data

100 unique successful outputs, no execution errors. The 20 and 80-question runs have
identical recorded inference settings apart from the data fingerprint. Raw exact/rational
matches: 26/100. Unified option-content equivalence review: 28/100, including added
matches 1324 (80-2pi -> B) and 2725 (1/3 -> C). These are answer matches, not validation
of all intermediate reasoning or official benchmark accuracy. Two truncated answers,
four initial extraction failures. Per-question recorded inference time sums to 853.56 s.

The merged outputs retain original records and merge provenance. See
results/qwen_mathvision100_merged and results/qwen100_review.

## VL-Rethinker setup (pending cloud execution)

Repository: https://huggingface.co/TIGER-Lab/VL-Rethinker-7B
Original weights are approximately 16.6 GB; check free disk space before downloading.
Use the existing PyTorch/Transformers environment. Download script resolves a specific
repository commit and records it in download_metadata.json.

The published model config has use_cache=false. v3 explicitly uses use_cache=True
in generation to avoid repeatedly recomputing the entire prefix. This is an inference
cache optimization; it preserves prompts, image pixel limits, BF16, SDPA, greedy decoding
and the 4096-token budget. Its inference has not yet been verified on this cloud instance.
Do not compare per-question timing as strictly identical implementation performance
without separately verifying cache and processor settings for both checkpoints.

After uploading vl_rethinker100_test_bundle.zip to /root:

```bash
unzip -n /root/vl_rethinker100_test_bundle.zip -d /root/autodl-tmp/ovr
df -h /root/autodl-tmp
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/ovr/hf-cache
export HF_TOKEN_PATH=/root/.cache/huggingface/token
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_ENABLE_HF_TRANSFER
python /root/autodl-tmp/ovr/scripts/download_checkpoint.py \
  --repo TIGER-Lab/VL-Rethinker-7B \
  --output /root/autodl-tmp/ovr/models/vl-rethinker
```

After successful download:

```bash
python -u /root/autodl-tmp/ovr/scripts/run_mathvision_test_v3.py \
  --model /root/autodl-tmp/ovr/models/vl-rethinker \
  --dataset /root/autodl-tmp/ovr/data/dataset_test100 \
  --output /root/autodl-tmp/ovr/results/vl_rethinker_mathvision100_4096 \
  --max-new-tokens 4096
```

The 4096-token cap is a shared-budget pilot, not a reproduction of every model's
published evaluation. Report truncation; save any longer-budget experiment separately.
Do not add forced-rethinking prompts only to one model in this shared-prompt comparison.

Backup after completion:

```bash
cd /root/autodl-tmp/ovr
tar -czf qwen_vl_rethinker_backup.tar.gz results scripts logs data/dataset_test data/dataset_test_extra80 data/dataset_test100 models/vl-rethinker/download_metadata.json
```

This excludes model weight files. Download raw outputs and apply the same offline
review to each model before making paired charts.
