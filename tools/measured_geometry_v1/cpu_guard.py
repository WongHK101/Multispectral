"""Process-local safeguards for the explicitly CPU-only preflight command."""

from __future__ import annotations

import importlib.abc
import os
import sys


GPU_MODULES = {"torch", "cupy", "pynvml", "pynvml_utils", "gsplat", "nerfstudio",
               "diff_gaussian_rasterization", "gaussian_renderer"}
_GUARD_INSTALLED = False


class NoGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in GPU_MODULES:
            raise RuntimeError(f"CPU-only preflight rejects import: {fullname}")
        return None


def install_cpu_guard():
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        evidence()
        return
    if GPU_MODULES.intersection(sys.modules):
        raise RuntimeError("Run preflight in a fresh process without GPU modules")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, NoGpuImports())

    def audit(event, args):
        if event in {"subprocess.Popen", "os.system", "socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError(f"CPU-only preflight forbids external execution/network: {event}")
        if event == "ctypes.dlopen" and any(token in str(args).lower() for token in ("cuda", "nvml", "nvidia")):
            raise RuntimeError("CPU-only preflight rejects GPU library")

    sys.addaudithook(audit)
    _GUARD_INSTALLED = True


def evidence():
    if (not _GUARD_INSTALLED or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"):
        raise RuntimeError("CPU-only process guard is absent or its environment was changed")
    if GPU_MODULES.intersection(sys.modules):
        raise RuntimeError("GPU module imported during preflight")
    return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_modules_imported": [], "external_processes_allowed": False,
            "network_allowed": False, "gpu_experiment_authorized": False,
            "resume_policy": "wait_for_explicit_user_gpu_available_message"}
