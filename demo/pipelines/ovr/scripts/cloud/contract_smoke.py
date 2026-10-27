"""CPU API smoke using a tiny RANDOM model, not OVR evaluation. No 7B weights."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from ovr.common import read_json, read_jsonl, write_json
    from ovr.cloud_backend import TransformersBackend, versions, OFFICIAL_ENDPOINT
    from ovr.cloud import CONFIG, namespace
    cfg = read_json(CONFIG)
    spec = cfg["models"]["coldstart"]
    assets = ROOT / ".build/contract-assets"
    snapshot_download(spec["repo"], revision=spec["revision"], local_dir=assets, max_workers=2,
                      endpoint=OFFICIAL_ENDPOINT, token=False,
                      allow_patterns=["config.json", "preprocessor_config.json", "chat_template.json", "tokenizer*.json",
                                      "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"])
    tiny = ROOT / ".build/contract-tiny-random"
    tiny.mkdir(parents=True, exist_ok=True)
    for p in assets.iterdir():
        if p.is_file() and p.name != "config.json":
            shutil.copy2(p, tiny / p.name)
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": len(tokenizer), "hidden_size": 64, "intermediate_size": 128,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "max_position_embeddings": 8192,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                       "out_hidden_size": 64, "fullatt_block_indexes": [0, 1]},
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    parameters = sum(p.numel() for p in model.parameters())
    model.save_pretrained(tiny)
    del model
    cfg["max_new_tokens"] = 2
    backend = TransformersBackend(cfg, tiny, device="cpu")
    item = read_jsonl(ROOT / cfg["shared_dir"] / "inputs.jsonl")[0]
    result = backend.infer(item["prompt"], ROOT / cfg["shared_dir"] / item["image"], 30)
    assert result["image_tensor_checked"] and result["image_grid_thw"]
    assert result["output_tokens"] == 2
    assert "<|vision_start|><|image_pad|><|vision_end|>" in result["formatted_prompt"]
    assert result["formatted_prompt"].endswith("<think>\n")
    out = {"status": "PASSED", "kind": "CPU interface check with tiny RANDOM weights",
           "real_ovr_generations": 0, "gpu_verified": False, "tiny_parameters": parameters,
           "source_processor": spec, "protocol_id": cfg["protocol_id"],
           "image_processor_class": backend.metadata["image_processor_class"], "packages": versions(), "image_grid_thw": result["image_grid_thw"],
           "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
           "chat_template_sha256": backend.metadata["chat_template_sha256"],
           "verified": ["official processor multimodal template and <think> prefill", "real image tensor",
                        "tiny model forward and generate", "same backend used by cloud entrypoint"],
           "not_verified": ["OVR weights loading on CUDA", "OVR-RL gated access", "BF16 GPU kernels and VRAM"]}
    write_json(ROOT / "results/reports" / ("cloud_contract_smoke.json" if namespace(cfg) == "cloud_pilot20" else namespace(cfg) + "_contract_smoke.json"), out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
