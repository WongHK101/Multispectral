"""Supervised, deadline-limited native packet kernel qualification on 901.

Nothing launches on import. The parent requires a hash-bound user notification,
CPU gate evidence and a freshly inspected server ownership snapshot. It also
checks NVIDIA idle state itself immediately before starting the isolated child.
This is not the long-experiment launcher and never powers off a server.
"""
from __future__ import annotations

import argparse
import csv
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

from .campaign import qualification_start_decision, write_json_exclusive
from .contracts import read_json, safe_member, sha256, verify_sha, verify_source_snapshot


def check_source(root, expected):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if head != expected or status:
        raise ValueError("Method source commit/clean state mismatch")
    return {"head": head, "status": status}


def check_snapshot(root, manifest_sha):
    root = Path(root).resolve()
    manifest = root / "SOURCE_SNAPSHOT.json"
    verify_sha(manifest, manifest_sha)
    data = read_json(manifest)
    if (data["schema"] != "umgs_committed_source_snapshot_v1"
            or data["generator_worktree_clean"] is not True
            or len(data["git_commit"]) != 40):
        raise ValueError("Invalid committed source snapshot")
    files = {"SOURCE_SNAPSHOT.json"}
    for row in data["files"]:
        if row["path"] in files:
            raise ValueError("Duplicate snapshot member")
        files.add(row["path"])
        p = safe_member(root, row["path"])
        verify_sha(p, row["sha256"])
        if p.stat().st_size != row["bytes"]:
            raise ValueError("Snapshot size mismatch")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if files != actual:
        raise ValueError("Missing or extra source snapshot file")
    module = Path(__file__).resolve()
    if module != root / "tools/measured_geometry_v1/kernel_qualification.py":
        raise ValueError("Not running the bound orchestration snapshot")
    return data


def idle_samples():
    samples = []
    for index in range(3):
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=15)
        rows = list(csv.reader(io.StringIO(raw)))
        if len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError("Require one identified GPU, not an ambiguous device selection")
        uuid, utilization, used, total = [x.strip() for x in rows[0]]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                        "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        if not uuid.startswith("GPU-") or apps or float(utilization) > 5 or float(used) > 1024:
            raise ValueError("GPU is not idle; no foreign process is stopped")
        samples.append({"uuid": uuid, "utilization_percent": float(utilization),
                        "used_mib": float(used), "total_mib": float(total), "compute_apps": []})
        if index < 2:
            time.sleep(1)
    if len({r["uuid"] for r in samples}) != 1:
        raise ValueError("GPU identity changed during preflight")
    return samples


def bounded_child(argv, *, cwd, env, seconds, output):
    """Kill only the newly created child process group on deadline/interruption."""
    if os.name != "posix" or not 0 < seconds <= 1800:
        raise ValueError("901 POSIX bounded execution required")
    started = time.monotonic()
    timed_out = False
    with (output / "console.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            try:
                code = child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    return {"exit_code": code, "timeout": timed_out, "pid": child.pid,
            "elapsed_seconds": time.monotonic()-started, "deadline_seconds": seconds,
            "termination_scope": "new_child_process_group_only"}


def validate_request(config):
    r = config["request"]
    if (config["schema"] != "umgs_gsplat_native_kernel_qualification_v1"
            or r["method"] != "ms_splatting_neural" or r["operation"] != "kernel_packet_parity"
            or r["max_iterations"] != 0):
        raise ValueError("This executor only implements zero-iteration gsplat kernel qualification")
    if config["method_commit"] != "9e7e128821c84c823edf6597e6817777cbd69df6":
        raise ValueError("Unreviewed method identity")
    if config["expected_host"] != platform.node():
        raise ValueError("Unexpected server hostname")
    expected_packages = {"torch": "2.8.0", "gsplat": "1.4.0", "nerfstudio": "1.1.5"}
    if config["runtime_packages"] != expected_packages or not config["runtime_source_files"]:
        raise ValueError("Unbound or different native runtime")
    if config["torch_build"] != {"version": "2.8.0+cu128", "cuda": "12.8",
                                 "git_version": "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"}:
        raise ValueError("Unbound PyTorch build; package metadata is not the CUDA build version")


def build_environment(config, output, gpu_uuid):
    """Bind existing 901 build tools without relying on an interactive shell PATH."""
    cuda_home = Path("/usr/local/cuda-12.8")
    method_bin = Path(config["method_python"]).parent
    binaries = {"nvcc": cuda_home / "bin/nvcc", "ninja": method_bin / "ninja",
                "gcc": Path("/usr/bin/gcc"), "g++": Path("/usr/bin/g++")}
    records = {}
    for name, path in binaries.items():
        version = subprocess.check_output([str(path), "--version"], text=True, timeout=30)
        if name == "nvcc" and "release 12.8," not in version:
            raise ValueError("CUDA compiler differs from the bound PyTorch CUDA build")
        records[name] = {"path": str(path), "sha256": sha256(path), "version": version}
    env = dict(os.environ)
    for name in ("PYTHONPATH", "LD_PRELOAD", "CUDA_PATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=gpu_uuid,
               CUDA_HOME=str(cuda_home), CC=str(binaries["gcc"]), CXX=str(binaries["g++"]),
               PATH=os.pathsep.join([str(method_bin), str(cuda_home / "bin"), env.get("PATH", "")]),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_EXTENSIONS_DIR=str(output / "build_cache"), MAX_JOBS="1")
    return env, records


def parent(config_path, expected_sha):
    verify_sha(config_path, expected_sha)
    cfg = read_json(config_path)
    validate_request(cfg)
    source = check_snapshot(cfg["orchestration_root"], cfg["orchestration_manifest_sha256"])
    check_source(cfg["method_root"], cfg["method_commit"])
    loaded = {}
    for key in ("plan", "authorization", "ownership_snapshot"):
        spec = cfg[key]
        verify_sha(spec["path"], spec["sha256"])
        loaded[key] = read_json(spec["path"])
    auth = loaded["authorization"]
    verify_sha(cfg["user_notification_path"], auth["user_message_evidence_sha256"])
    if auth.get("explicit_user_gpu_available") is not True:
        raise ValueError("No explicit user GPU notification")
    gates = {}
    for key, spec in cfg["cpu_gates"].items():
        verify_sha(spec["path"], spec["sha256"])
        gates[key] = {"status": spec["status"], "verified_report_sha256": spec["sha256"]}
    # Fail before querying a GPU if CPU/authorization/ownership inputs are invalid.
    snapshot = dict(loaded["ownership_snapshot"])
    dry_snapshot = {**snapshot, "three_idle_samples_passed": True}
    gate = qualification_start_decision(loaded["plan"], auth, gates, dry_snapshot,
                                        cfg["request"], now=time.time())
    if not gate["allowed"]:
        raise ValueError("Qualification denied: " + ",".join(gate["reasons"]))
    samples = idle_samples()
    snapshot["three_idle_samples_passed"] = True
    gate = qualification_start_decision(loaded["plan"], auth, gates, snapshot,
                                        cfg["request"], now=time.time())
    if not gate["allowed"]:
        raise ValueError("Qualification stale after idle preflight: " + ",".join(gate["reasons"]))
    output = Path(cfg["output_root"]).resolve()
    allowed = Path("/root/autodl-tmp/umgs-tgrs").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Qualification output must remain inside the project run root")
    output.mkdir(parents=True, exist_ok=False)
    env, build_tools = build_environment(cfg, output, samples[0]["uuid"])
    argv = [cfg["method_python"], "-B", "-m", "tools.measured_geometry_v1.kernel_qualification",
            "--child", "--config", str(Path(config_path).resolve()), "--config_sha256", expected_sha]
    write_json_exclusive(output / "launch.json", {"decision": gate, "config_sha256": expected_sha,
        "orchestration_commit": source["git_commit"], "idle_samples": samples, "argv": argv,
        "build_tools": build_tools, "cuda_home": env["CUDA_HOME"],
        "ownership_snapshot_sha256": cfg["ownership_snapshot"]["sha256"],
        "output_class": "nonformal_qualification_only", "full_experiments_started": False})
    run = bounded_child(argv, cwd=cfg["orchestration_root"], env=env,
                        seconds=cfg["request"]["max_gpu_seconds"], output=output)
    child_path = output / "native_kernel_report.json"
    run["child_report_sha256"] = sha256(child_path) if child_path.is_file() else None
    run["status"] = "PASS_KERNEL_ONLY" if (run["exit_code"] == 0 and not run["timeout"]
        and child_path.is_file() and read_json(child_path)["status"] == "PASS_SYNTHETIC_NATIVE_KERNEL_ONLY") else "BLOCKED"
    run["full_experiment_qualification"] = False
    run["power_operation_performed"] = False
    write_json_exclusive(output / "exit_report.json", run)
    return run


def child(config_path, expected_sha):
    verify_sha(config_path, expected_sha)
    cfg = read_json(config_path)
    validate_request(cfg)
    output = Path(cfg["output_root"])
    launch = read_json(output / "launch.json")
    if launch["config_sha256"] != expected_sha or launch["decision"]["allowed"] is not True:
        raise ValueError("Missing parent qualification permit")
    check_snapshot(cfg["orchestration_root"], cfg["orchestration_manifest_sha256"])
    check_source(cfg["method_root"], cfg["method_commit"])
    for package, version in cfg["runtime_packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Native runtime package mismatch: " + package)
    for row in cfg["runtime_source_files"]:
        verify_sha(row["path"], row["sha256"])
    verify_source_snapshot(cfg["reference_root"], cfg["reference_manifest_sha256"])
    from .preflight import _load_reference_module
    reference = _load_reference_module(Path(cfg["reference_root"]), "metric_depth_packet")
    from .native_kernel_smoke import gsplat_synthetic
    try:
        import mmsplat.mmsplat_model as model
        import torch
        build = {"version": torch.__version__, "cuda": torch.version.cuda, "git_version": torch.version.git_version}
        if build != cfg["torch_build"]:
            raise ValueError("Actual PyTorch build mismatch")
        if Path(model.__file__).resolve() != Path(cfg["method_root"]).resolve() / "mmsplat/mmsplat_model.py":
            raise ValueError("Unexpected imported method path")
        result = gsplat_synthetic(reference=reference)
        result["runtime_packages"] = cfg["runtime_packages"]
        result["torch_build"] = build
        result["runtime_source_files"] = cfg["runtime_source_files"]
        check_source(cfg["method_root"], cfg["method_commit"])
    except Exception as exc:
        result = {"status": "BLOCKED_KERNEL_QUALIFICATION", "error": str(exc), "traceback": traceback.format_exc(),
                  "training_started": False, "formal_metrics_generated": False}
    write_json_exclusive(output / "native_kernel_report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config_sha256", required=True)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    result = (child if args.child else parent)(args.config, args.config_sha256)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
