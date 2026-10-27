"""Optional HF/CUDA dependencies are imported only for the requested cloud action."""
import copy
import hashlib
import fnmatch
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from .common import ROOT, digest, fingerprint, read_json, write_json

PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "safetensors", "tokenizers", "Pillow")
OFFICIAL_ENDPOINT = "https://huggingface.co"
PUBLIC_MIRROR = "https://hf-mirror.com"
DOWNLOAD_PATTERNS = ["*.json", "*.txt", "*.jinja", "*.safetensors"]


def download_transport(model, info):
    """Mirrors are anonymous and restricted to the public ColdStart checkpoint."""
    requested = os.environ.get("OVR_PUBLIC_MODEL_ENDPOINT", PUBLIC_MIRROR).rstrip("/")
    if requested not in {PUBLIC_MIRROR, OFFICIAL_ENDPOINT}:
        raise ValueError("OVR_PUBLIC_MODEL_ENDPOINT must be the supported public mirror or official HF endpoint")
    if model != "coldstart" or info.get("gated", True):
        return OFFICIAL_ENDPOINT
    return requested


def snapshot_with_fallback(spec, source, endpoint):
    from huggingface_hub import snapshot_download
    workers = int(os.environ.get("OVR_DOWNLOAD_WORKERS", "2"))
    if workers not in (1, 2):
        raise ValueError("OVR_DOWNLOAD_WORKERS must be 1 or 2")
    kwargs = {"revision": spec["revision"], "local_dir": source, "allow_patterns": DOWNLOAD_PATTERNS, "max_workers": workers}
    try:
        # token=False prevents implicit forwarding of cached/env HF credentials.
        snapshot_download(spec["repo"], endpoint=endpoint, token=False if endpoint != OFFICIAL_ENDPOINT else None, **kwargs)
        return {"endpoint": endpoint, "anonymous_mirror": endpoint != OFFICIAL_ENDPOINT}
    except Exception as exc:
        if endpoint == OFFICIAL_ENDPOINT:
            raise
        print("Public mirror unavailable (" + type(exc).__name__ + "); trying official HF once", flush=True)
        snapshot_download(spec["repo"], endpoint=OFFICIAL_ENDPOINT, token=None, **kwargs)
        return {"endpoint": OFFICIAL_ENDPOINT, "anonymous_mirror": False, "fallback_from": endpoint}


def verify_source(source, files):
    """Check mirror bytes against official Git/LFS identities, including configs."""
    source = Path(source)
    expected = {f["name"]: f for f in files}
    actual = {p.name for p in source.iterdir() if p.is_file() and not p.name.startswith(".")}
    if actual != set(expected):
        raise ValueError("Downloaded file set differs from pinned official metadata")
    hashes = {}
    for name, info in expected.items():
        path = source / name
        if path.stat().st_size != info["size"]:
            raise ValueError("Downloaded size mismatch: " + name)
        sha = digest(path)
        if info.get("sha256"):
            if sha != info["sha256"]:
                raise ValueError("Downloaded LFS SHA256 mismatch: " + name)
        else:
            content = path.read_bytes()
            git_hash = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if git_hash != info.get("git_blob_sha1"):
                raise ValueError("Downloaded config Git hash mismatch: " + name)
        hashes[name] = sha
    return hashes


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def access(config):
    """HEAD every weight; never download weights or serialize signed URLs/tokens."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    # Never let a global HF_ENDPOINT send authenticated checks to a mirror.
    api = HfApi(endpoint=OFFICIAL_ENDPOINT)
    result = {}
    for name, spec in config["models"].items():
        try:
            info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
            if info.sha != spec["revision"]:
                raise ValueError("Revision did not resolve to the pinned commit")
            weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
            if not weights:
                raise ValueError("No safetensors weights")
            get_hf_file_metadata(hf_hub_url(spec["repo"], "config.json", revision=spec["revision"], endpoint=OFFICIAL_ENDPOINT))
            files = []
            for s in weights:
                get_hf_file_metadata(hf_hub_url(spec["repo"], s.rfilename, revision=spec["revision"], endpoint=OFFICIAL_ENDPOINT))
                lfs = s.lfs
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                if not sha or len(sha) != 64 or not s.size:
                    raise ValueError("Weight size and upstream SHA256 are required")
                files.append({"name": s.rfilename, "size": s.size, "sha256": sha})
            artifacts = []
            for s in info.siblings:
                if "/" in s.rfilename or not any(fnmatch.fnmatch(s.rfilename, pat) for pat in DOWNLOAD_PATTERNS):
                    continue
                weight = next((w for w in files if w["name"] == s.rfilename), None)
                if weight:
                    artifacts.append(weight)
                else:
                    lfs_sha = s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
                    if lfs_sha and len(lfs_sha) == 64 and s.size is not None:
                        artifacts.append({"name": s.rfilename, "size": s.size, "sha256": lfs_sha})
                        continue
                    if not s.blob_id or len(s.blob_id) != 40 or s.size is None:
                        raise ValueError("Official config Git hash and size are required")
                    artifacts.append({"name": s.rfilename, "size": s.size, "git_blob_sha1": s.blob_id})
            result[name] = {"accessible": True, **spec, "gated": bool(info.gated), "files": artifacts,
                            "weights": files, "bytes": sum(s["size"] for s in files)}
        except Exception as exc:
            result[name] = {"accessible": False, **spec, "error_type": type(exc).__name__,
                            "next_step": "Check HF login, gated access approval and network; no weights downloaded."}
    return {"complete": all(r["accessible"] for r in result.values()), "models": result}


def gpu_check(config=None):
    import torch
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Cloud execution requires Linux and working CUDA PyTorch")
    runtime = (config or {}).get("runtime", {})
    expected_python = runtime.get("python", "3.11")
    if ".".join(platform.python_version_tuple()[:2]) != expected_python:
        raise RuntimeError("Python differs from the recorded runtime protocol")
    if runtime.get("torch_cuda") and torch.version.cuda != runtime["torch_cuda"]:
        raise RuntimeError("CUDA build differs from the recorded runtime protocol")
    installed = versions()
    expected = {"torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.1", "accelerate": "1.11.0",
                "huggingface-hub": "0.35.3", "safetensors": "0.6.2", "tokenizers": "0.22.1", "Pillow": "11.3.0"}
    if any(installed[k].split("+")[0] != v for k, v in expected.items()):
        raise RuntimeError("Core package versions differ; run scripts/cloud/setup.sh in an isolated environment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on this GPU")
    prop = torch.cuda.get_device_properties(0)
    if prop.total_memory < 40 * 2**30:
        raise RuntimeError("This prepared BF16 route requires at least 40 GiB VRAM; use the recommended 48/80GB card")
    return {"name": prop.name, "memory_bytes": prop.total_memory, "capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda": torch.version.cuda, "packages": installed, "python": platform.python_version(),
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               text=True, timeout=15).strip().splitlines()[0]}


def model_directory(config, model):
    spec = config["models"][model]
    return ROOT / "models" / f"{model}-{spec['revision'][:12]}"


def build_load_view(source, target):
    """Keep upstream immutable; normalize the merged filename only in a symlink view."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError("Checkpoint has no safetensors files")
    merged = len(weights) == 1
    if not merged or (source / "model.safetensors.index.json").exists():
        index = read_json(source / "model.safetensors.index.json")
        if set(index["weight_map"].values()) != {p.name for p in weights}:
            raise ValueError("Shard index and downloaded weight files disagree")
    for file in sorted(source.iterdir()):
        if not file.is_file() or file.name.startswith("."):
            continue
        # Expose the single verified weight under the standard Transformers name.
        # Preserve the upstream index in source, where it keeps its original name.
        if merged and file.name == "model.safetensors.index.json":
            continue
        name = "model.safetensors" if merged and file.suffix == ".safetensors" else file.name
        dest = target / name
        if dest.is_symlink():
            if dest.resolve() != file.resolve():
                raise ValueError("Existing load view targets a different source")
        elif dest.exists():
            raise ValueError("Refuse to overwrite an existing checkpoint view file")
        else:
            dest.symlink_to(os.path.relpath(file, target))
    return {"source_weight_files": [p.name for p in weights], "single_weight_alias": merged,
            "upstream_files_modified": False}


def download(config, checked):
    if not checked["complete"]:
        raise ValueError("Both models must pass access HEAD checks before downloading either")
    for model, spec in config["models"].items():
        directory = model_directory(config, model)
        source = directory / "source"
        directory.mkdir(parents=True, exist_ok=True)
        required = sum(max(0, w["size"] - ((source / w["name"]).stat().st_size if (source / w["name"]).exists() else 0))
                       for w in checked["models"][model]["weights"])
        if shutil.disk_usage(directory).free < required + 10 * 2**30:
            raise RuntimeError("Insufficient disk for weights plus a 10GiB reserve")
        endpoint = download_transport(model, checked["models"][model])
        transport = snapshot_with_fallback(spec, source, endpoint)
        verified = verify_source(source, checked["models"][model]["files"])
        view = build_load_view(source, directory / "load")
        artifacts = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                      "sha256": verified[p.name]}
                     for p in sorted(source.iterdir()) if p.is_file() and not p.name.startswith(".")]
        write_json(directory / "manifest.json", {**spec, "files": artifacts, "view": view, "download_transport": transport})
        print(f"{model}: weights SHA256 verified and local load view ready", flush=True)


def checkpoint(config, model):
    directory = model_directory(config, model)
    manifest = read_json(directory / "manifest.json")
    spec = config["models"][model]
    if any(manifest[k] != spec[k] for k in ("repo", "revision")):
        raise ValueError("Local checkpoint revision differs from protocol")
    for file in manifest["files"]:
        p = directory / "source" / file["name"]
        if p.stat().st_size != file["size"] or p.stat().st_mtime_ns != file["mtime_ns"]:
            raise ValueError("Checkpoint changed since verified download; re-run download verification")
        if p.suffix != ".safetensors" and digest(p) != file["sha256"]:
            raise ValueError("Checkpoint configuration changed")
    build_load_view(directory / "source", directory / "load")
    return directory / "load", {"repo": spec["repo"], "revision": spec["revision"],
                                "files": [{k: f[k] for k in ("name", "size", "sha256")} for f in manifest["files"]],
                                "view": manifest["view"]}


class TransformersBackend:
    def __init__(self, config, load_path, device="cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
        self.torch, self.config, self.device = torch, config, device
        self.is_cuda = device.startswith("cuda")
        set_seed(config["seed"])
        self.processor = AutoProcessor.from_pretrained(load_path, local_files_only=True, trust_remote_code=False,
                                                       use_fast=config.get("image_processor_use_fast", False), **config["image_processor"])
        if not self.processor.chat_template:
            raise ValueError("Official processor chat template is missing")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            load_path, dtype=torch.bfloat16 if self.is_cuda else torch.float32, device_map={"": device},
            attn_implementation=config["attention"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True).eval()
        self.generation = copy.deepcopy(self.model.generation_config)
        for k, value in config["generation"].items():
            setattr(self.generation, k, value)
        self.generation.temperature = None
        self.generation.top_p = None
        self.generation.top_k = None
        self.generation.max_new_tokens = config["max_new_tokens"]
        self.generation.use_cache = True
        self.generation.num_return_sequences = 1
        self.generation.return_dict_in_generate = False
        self.generation.output_scores = False
        self.generation.max_time = None
        self.metadata = {"chat_template_sha256": hashlib.sha256(self.processor.chat_template.encode()).hexdigest(),
                         "effective_generation": self.generation.to_dict(), "attention": config["attention"],
                         "image_processor_class": type(self.processor.image_processor).__name__,
                         "image_processor_use_fast_requested": config.get("image_processor_use_fast", False)}

    def infer(self, prompt, image_path, max_time_s):
        from PIL import Image
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch, cfg = self.torch, self.config
        call_started = time.monotonic()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text.count("<|image_pad|>") != 1:
            raise ValueError("Official chat template did not insert exactly one image placeholder")
        with Image.open(image_path) as im:
            batch = self.processor(text=[text], images=[im.convert("RGB")], padding=True, return_tensors="pt")
        if "pixel_values" not in batch or "image_grid_thw" not in batch or batch["pixel_values"].numel() == 0:
            raise ValueError("Processor produced no image tensor")
        input_tokens = batch["input_ids"].shape[-1]
        if input_tokens + cfg["max_new_tokens"] > cfg["max_context_tokens"]:
            raise ValueError("Input plus output exceeds protocol context budget; do not truncate the image silently")
        grid = batch["image_grid_thw"].tolist()
        batch = batch.to(self.device)
        class Deadline(StoppingCriteria):
            def __init__(self):
                self.started = time.monotonic()
                self.reached = False
            def __call__(self, input_ids, scores, **kwargs):
                self.reached = time.monotonic() - self.started >= max_time_s
                return self.reached
        deadline = Deadline()
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(**batch, generation_config=self.generation,
                                            stopping_criteria=StoppingCriteriaList([deadline]))
        if self.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        tokens = generated[0, input_tokens:].tolist()
        eos = self.generation.eos_token_id or self.model.config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = "stop" if tokens and tokens[-1] in eos else (
            "time_limit" if deadline.reached else "length" if len(tokens) >= cfg["max_new_tokens"] else "other")
        response = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        response_special = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
        return {"response": response,
                "response_with_special_tokens": response_special,
                "output_tokens": len(tokens), "generated_token_ids": tokens, "input_tokens": input_tokens,
                "image_grid_thw": grid, "image_tensor_checked": True,
                "finish_reason": reason, "latency_s": time.monotonic() - call_started,
                "generation_s": elapsed, "latency_scope": "end-to-end single question, excluding checkpoint load",
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if self.is_cuda else 0,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if self.is_cuda else 0,
                "formatted_prompt": text,
                "formatted_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
