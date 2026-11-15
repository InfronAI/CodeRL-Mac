import gc
import platform
import random
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import torch


def device_for(name="auto"):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Use --device cpu, or an arm64 Python on macOS with MPS support."
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device auto or cpu.")
    return torch.device(name)


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment(device):
    def sysctl(key):
        if sys.platform != "darwin":
            return None
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_bytes": sysctl("hw.memsize"),
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "versions": {p: version(p) for p in ["torch", "transformers", "numpy", "tokenizers", "safetensors"]},
        "determinism": "Seeded PRNGs; MPS floating-point kernels may differ between runs/hardware.",
    }
