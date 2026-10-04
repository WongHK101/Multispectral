"""Resolve installed upstream CLI and exercise CPU-only method components.

No Trainer/whole-model setup, image tensor cache, renderer, trained checkpoint
or GPU is used. The actual camera parser and shared initial PLY run on CPU.
Run in the isolated method environment with CUDA explicitly hidden.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

from .training_recipes import command, recipe, validate_resolved

COMPAT_COMMIT = "9e7e128821c84c823edf6597e6817777cbd69df6"


def json_safe(value):
    if isinstance(value, type):
        return value.__module__ + "." + value.__qualname__
    if dataclasses.is_dataclass(value):
        return {f.name: json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (Path, enum.Enum)):
        return str(value) if isinstance(value, Path) else json_safe(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    raise TypeError("Unrecognized resolved config value: " + type(value).__name__)


def check(root, data, planned_output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU smoke requires CUDA_VISIBLE_DEVICES='' explicitly")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if head != COMPAT_COMMIT or status:
        raise ValueError("Unexpected or dirty method source")
    import torch
    import tyro
    from nerfstudio.scripts.train import AnnotatedBaseConfigUnion
    from nerfstudio.field_components.encodings import NeRFEncoding
    from mmsplat.mmsplat_neural_surface import MultiSpectralFeatureDecoder
    import mmsplat.mmsplat_model as model_module

    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    model_path = Path(model_module.__file__).resolve()
    if model_path != (root / "mmsplat/mmsplat_model.py").resolve():
        raise ValueError("Imported method is not the isolated source")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rows, resolved = [], {}
    for method in ("jo", "sig_mechanism", "ms_splatting_neural"):
        entrypoint = Path(sys.executable).with_name("ns-train")
        if not entrypoint.is_file():
            raise FileNotFoundError(entrypoint)
        binding = command(method, entrypoint=str(entrypoint),
                          data=str(data), output=str(planned_output / method),
                          split=str(data / "train_split.json"))
        config = tyro.cli(AnnotatedBaseConfigUnion, args=binding["argv"][1:])
        if config.data != data or config.output_dir != planned_output / method:
            raise ValueError("Resolved source/output path mismatch")
        if config.pipeline.datamanager.dataparser.json_list_path != data / "train_split.json":
            raise ValueError("Resolved split path mismatch")
        # The official train.main applies this top-level --data alias, without setup.
        config.pipeline.datamanager.data = config.data
        safe = json_safe(config)
        verification = validate_resolved(safe, recipe(method))
        resolved[method] = {"config": safe, "command": binding, "verification": verification}
        rows.append({"test": "upstream_cli_" + method, "status": "PASS"})

    # The upstream manager applies this alias before constructing its parser.
    # Do not construct the manager or its complete RGB cache in the CPU smoke.
    config.pipeline.datamanager.dataparser.data = data
    parser = config.pipeline.datamanager.dataparser.setup()
    split_record = json.loads((data / "train_split.json").read_text())
    frames = json.loads((data / "transforms.json").read_text())
    frames_by_path = {f["file_path"]: f for f in frames["frames"]}
    parsed = {}
    for split, key in (("train", "train"), ("test", "eval")):
        output = parser.get_dataparser_outputs(split=split)
        filenames = [p.relative_to(data).as_posix() for p in output.image_filenames]
        assert len(filenames) == len(set(filenames)) and set(filenames) == set(split_record[key])
        assert all(p.is_file() for p in output.image_filenames)
        dims = [(int(output.cameras.width[i].item()), int(output.cameras.height[i].item()))
                for i in range(len(filenames))]
        assert all(d == (frames_by_path[p]["w"], frames_by_path[p]["h"]) for p, d in zip(filenames, dims))
        assert torch.isfinite(output.cameras.camera_to_worlds).all()
        assert torch.isfinite(output.metadata["points3D_xyz"]).all()
        parsed[split] = {"image_count": len(filenames), "image_names": filenames,
                         "dimensions": sorted(set(dims)), "dataparser_scale": output.dataparser_scale,
                         "dataparser_transform": output.dataparser_transform.tolist(),
                         "initial_point_count": len(output.metadata["points3D_xyz"]),
                         "camera_device": str(output.cameras.camera_to_worlds.device)}
        rows.append({"test": "actual_dataparser_" + split + "_identity_native_grid", "status": "PASS"})
    assert not set(parsed["train"]["image_names"]) & set(parsed["test"]["image_names"])
    assert parsed["train"]["dataparser_transform"] == parsed["test"]["dataparser_transform"]
    assert parsed["train"]["dataparser_scale"] == parsed["test"]["dataparser_scale"]
    rows.append({"test": "actual_dataparser_split_and_common_normalization", "status": "PASS"})

    encoding = NeRFEncoding(in_dim=3, num_frequencies=0, min_freq_exp=0,
                            max_freq_exp=0, include_input=True, implementation="torch")
    directions = torch.tensor([[.2, -.3, .5], [-1., .7, .8]], dtype=torch.float32)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    encoded = encoding(directions)
    assert encoded.shape == (2, 3) and torch.equal(encoded, directions)
    rows.append({"test": "zero_frequency_direction_identity", "status": "PASS"})
    decoder = MultiSpectralFeatureDecoder(11, output_dim=7, hidden_depth=32,
                                         hidden_layers=1, hidden_activation_function="ELU", mlp_mode=True)
    affine = [[m.in_features, m.out_features] for m in decoder.modules() if isinstance(m, torch.nn.Linear)]
    count = sum(p.numel() for p in decoder.parameters())
    assert affine == [[11, 32], [32, 32], [32, 7]] and count == 1671
    x = torch.arange(22, dtype=torch.float32).reshape(2, 11).requires_grad_()
    y = decoder(x)
    y.sum().backward()
    assert y.shape == (2, 7) and torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    rows.append({"test": "actual_decoder_shape_and_cpu_backward", "status": "PASS",
                 "affine_layers": affine, "parameters": count})

    tree = ast.parse(model_path.read_text())
    methods = {n.name: n for cls in tree.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, ast.FunctionDef)}
    statements = [n for n in methods["get_outputs"].body if isinstance(n, ast.If)
                  and any(name in ast.unparse(n.test) for name in
                          ("render_channel.pos_optim_delay", "render_channel.opacity_optim_delay"))]
    regularizers = [n for n in methods["get_loss_dict"].body if isinstance(n, ast.If)
                    and "self.config.use_scale_regularization" in ast.unparse(n.test)]
    assert len(statements) == 2 and len(regularizers) == 1
    routing = compile(ast.Module(body=statements, type_ignores=[]), str(model_path), "exec")
    regularizer = compile(ast.Module(body=regularizers, type_ignores=[]), str(model_path), "exec")
    for method, channel, detached in (("jo", "MS_R", False), ("sig_mechanism", "MS_R", True),
                                       ("sig_mechanism", "D", False)):
        r = recipe(method)["pipeline"]["model"]

        def delay(values, wanted):
            pending = []
            for value in values:
                try:
                    number = int(value)
                except ValueError:
                    pending.append(value)
                else:
                    if wanted in pending:
                        return number
                    pending = []
            return 0

        parameters = {key: torch.nn.Parameter(torch.ones(2, 3))
                      for key in ("means", "quats", "scales", "opacities")}
        parameters["scales"] = torch.nn.Parameter(torch.log(torch.tensor([[20., 1., 1.], [20., 1., 1.]])))
        obj = SimpleNamespace(gauss_params=parameters, step=40000, device="cpu",
                              config=SimpleNamespace(use_scale_regularization=True, max_gauss_ratio=10.))
        context = {"torch": torch, "self": obj,
                   "render_channel": SimpleNamespace(pos_optim_delay=delay(r["pos_optim_delay_channels"], channel),
                                                     opacity_optim_delay=delay(r["opacity_optim_delay_channels"], channel))}
        exec(routing, context)
        sum(context["r_" + key].sum() for key in parameters).backward()
        assert all((p.grad is None) == detached for p in parameters.values())
        rows.append({"test": "actual_photometric_route_" + method + "_" + channel, "status": "PASS",
                     "original_parameter_gradients_detached": detached})
        if detached:
            exec(regularizer, context)
            context["scale_reg"].backward()
            assert parameters["scales"].grad is not None and parameters["scales"].grad.abs().sum() > 0
            rows.append({"test": "sig_scale_regularizer_not_geometry_freeze", "status": "PASS"})
    assert not torch.cuda.is_initialized()
    return {"status": "PASS_ACTUAL_UPSTREAM_CPU_ONLY", "tests": rows, "resolved": resolved,
            "actual_dataparser": parsed,
            "method_commit": head, "method_clean": True,
            "model_source_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "cuda_initialized": False, "training_started": False, "renderer_called": False,
            "model_or_dataset_instantiated": False, "camera_dataparser_instantiated": True,
            "gradient_test_scope": "actual AST-selected routing and regularizer; no full training or rasterization"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method_root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--planned_output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if any(args.report.resolve().is_relative_to(p.resolve()) for p in (args.method_root, args.data)):
        raise ValueError("Report must not be written inside a method source or dataset")
    try:
        result = check(args.method_root.resolve(), args.data.resolve(), args.planned_output.resolve())
    except BaseException as exc:
        result = {"status": "BLOCKED_CPU_SMOKE", "error": str(exc), "traceback": traceback.format_exc()}
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(args.report)}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
