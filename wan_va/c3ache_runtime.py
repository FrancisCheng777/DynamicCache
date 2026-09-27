"""Reproducibility metadata shared by the inference server and benchmark tooling."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path


UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"


def checkpoint_manifest(path):
    """Fingerprint local filenames/sizes/mtimes, not the multi-GB weight contents.

    This deliberately requires an unchanged local checkpoint for a paired run.
    The returned label describes the limitation; it is not a content checksum.
    """
    root = Path(path).resolve()
    files = [p for p in root.rglob("*") if p.is_file()
             and not {".cache", ".git"}.intersection(p.relative_to(root).parts)]
    entries = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
               for p in sorted(files)]
    if not entries:
        raise ValueError(f"No checkpoint files found at {root}")
    digest = hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()
    return {"path": str(root), "fingerprint": digest, "fingerprint_kind": "relative_path_size_mtime_ns",
            "file_count": len(entries), "total_bytes": sum(e[1] for e in entries)}


def git_state(root):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        # Ignore generated/untracked evaluation outputs; record changes to tracked code.
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=root, text=True).strip())
        return {"commit": commit, "tracked_files_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "tracked_files_dirty": None}


def build_metadata(config, cache_config, *, device):
    import torch

    native_fields = (
        "frame_chunk_size", "action_per_frame", "action_dim", "height", "width",
        "obs_cam_keys", "patch_size", "attn_window", "num_inference_steps",
        "action_num_inference_steps", "video_exec_step", "snr_shift", "action_snr_shift",
        "guidance_scale", "action_guidance_scale", "used_action_channel_ids",
        "action_norm_method", "norm_stat", "env_type", "world_size",
    )
    packages = {}
    for name in ["torch", "diffusers", "transformers"]:
        packages[name] = importlib.metadata.version(name)
    props = torch.cuda.get_device_properties(device)
    execution_mode = ("shadow" if getattr(config, "diagnose_c3ache", False)
                      else "cached" if cache_config.enabled else "baseline")
    return {
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "code": git_state(Path(__file__).resolve().parents[1]),
        "checkpoint": checkpoint_manifest(config.wan22_pretrained_model_name_or_path),
        "native": {**{key: getattr(config, key) for key in native_fields},
                   "dtype": str(config.param_dtype), "attention_backend": "torch",
                   "enable_offload": getattr(config, "enable_offload", True),
                   "save_debug": getattr(config, "save_debug", True)},
        "c3ache": asdict(cache_config),
        "execution_mode": execution_mode,
        "timing_is_benchmark": execution_mode != "shadow",
        "profile_inference": getattr(config, "profile_inference", False),
        "runtime": {"python": platform.python_version(), "packages": packages,
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "gpu": props.name, "gpu_memory_bytes": props.total_memory},
    }
