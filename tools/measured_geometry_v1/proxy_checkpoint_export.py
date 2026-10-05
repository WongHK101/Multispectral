"""Fresh proxy renders on the frozen common Graphdeco camera grid.

This is not the native gsplat GCP/LiDAR adapter. Legacy six-array packets keep
their own numeric-valid mask; a separate v2 packet adds same-call real H.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .contracts import sha256, verify_sha, verify_source_snapshot
from .native_moments import RAW_NAMES, graphdeco_live_outputs, packet_from_reference, validate_raw_moments
from .ms_checkpoint_export import write_json


WRAPPER_SHA = "3d8e7463e98a6e90077965f90f43b8e4463915fbf31b0da5a728f46e3dd27c0b"
REGISTRY_SHA = "46d8ad6ec0ea198a0ce92e78723f86796daadd2b6955f1bf7b9d48ddc11a9d8b"
BINDING_SHA = "3a207f7cd267d6d7f6742190354c267bad245ac4af30fd263e24f956e7535785"
FORWARD_SHA = "a3187de5c7d256988b1b05b559a14992e42da8006b8765b813af1a95c82cfb1e"
LEGACY_NAMES = {"accumulated_opacity", "weighted_camera_z_sum", "weighted_camera_z2_sum",
                "expected_camera_z", "numeric_valid", "camera_z_variance"}


def source_moments_with_legacy_parity(planes, h, legacy_dp, legacy_source, scale):
    """Preserve the frozen proxy float32 conversion, including legacy validity."""
    if set(legacy_dp) != LEGACY_NAMES or set(legacy_source) != LEGACY_NAMES:
        raise ValueError("Legacy proxy packet must retain exactly six arrays")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid dataparser scale")
    raw = graphdeco_live_outputs(planes, h)
    source = {}
    for name, old_name in zip(RAW_NAMES[:3], (
            "accumulated_opacity", "weighted_camera_z_sum", "weighted_camera_z2_sum")):
        old = np.asarray(legacy_dp[old_name])
        if old.dtype != raw[name].dtype or old.tobytes() != raw[name].tobytes():
            raise ValueError("Same-call legacy accumulator mismatch: " + name)
        source[name] = np.asarray(legacy_source[old_name]).copy()
    # H has inverse-length units; do not reconstruct it by inverting M1/A.
    source[RAW_NAMES[3]] = (raw[RAW_NAMES[3]].astype(np.float64) * scale).astype(np.float32)
    return validate_raw_moments(source)


def export(args):
    if args.output.exists():
        raise ValueError("Output already exists")
    for path, digest in ((args.wrapper, WRAPPER_SHA), (args.registry, REGISTRY_SHA),
                         (args.checkpoint, args.checkpoint_sha256), (args.dataparser, args.dataparser_sha256)):
        verify_sha(path, digest)
    reference_identity = verify_source_snapshot(args.reference_root, args.reference_manifest_sha256)
    spec = importlib.util.spec_from_file_location("frozen_self5_proxy", args.wrapper)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    modules, exporter = wrapper.audited_modules(), wrapper.audited_exporter()
    registry = wrapper.load_registry(args.registry)
    scene = copy.deepcopy(wrapper.scene_record(registry, args.scene))
    normalization = json.loads(args.dataparser.read_text())
    transform = np.asarray(normalization["transform"], dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack((transform, [0, 0, 0, 1]))
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid saved dataparser transform")
    scale = float(normalization["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid saved dataparser scale")
    # Only the new model's actual normalization changes; source camera targets
    # and projection matrices remain the immutable registry definitions.
    scene["dataparser"] = {"path": str(args.dataparser), "sha256": args.dataparser_sha256,
                           "transform": transform.tolist(), "scale": scale}
    targets = scene["targets"]
    if not targets or len({t["image_name"] for t in targets}) != len(targets):
        raise ValueError("Missing or duplicate proxy target")

    import torch
    import gaussian_renderer
    import diff_gaussian_rasterization as rasterizer
    from .preflight import _load_reference_module

    runtime = wrapper.audited_runtime_root()
    expected_renderer = runtime / "gaussian_renderer/__init__.py"
    if Path(gaussian_renderer.__file__).resolve() != expected_renderer.resolve():
        raise ValueError("Wrong common renderer imported")
    verify_sha(expected_renderer, wrapper.AUDITED_RUNTIME_SHA256["gaussian_renderer/__init__.py"])
    verify_sha(runtime / "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu", FORWARD_SHA)
    verify_sha(rasterizer.__file__, BINDING_SHA)
    extension = Path(rasterizer._C.__file__).resolve()
    verify_sha(extension, args.extension_sha256)
    reference = _load_reference_module(args.reference_root, "metric_depth_packet")
    arrays, geometry = modules.e3.load_mss_checkpoint_geometry_e3(args.checkpoint)
    model = modules.e1.make_geometry_model(arrays)
    count = int(geometry["gaussian_count"])
    nan_rows = np.asarray(geometry.get("raw_log_scaling_complete_nan_row_indices", []), dtype=np.int64)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    color = torch.full((count, 3), .5, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "legacy").mkdir()
    (args.output / "v2").mkdir()
    rows = []
    with torch.no_grad():
        for target in targets:
            camera, audit = wrapper.build_renderer_camera(modules, scene, target)
            result = gaussian_renderer.render(camera, model, pipeline, background,
                override_color=color, use_trained_exp=False, separate_sh=False,
                return_expected_camera_z_packet=True, opacity_epsilon=1e-6, variance_clamp_tolerance=1e-6)
            if nan_rows.size:
                indices = torch.as_tensor(nan_rows, device=result["radii"].device)
                if not bool((result["radii"].index_select(0, indices) == 0).all()):
                    raise ValueError("Historical NaN-scale sentinel contributed to render")
            legacy_dp = exporter.packet_tensor_to_named_arrays(result["expected_camera_z_packet"],
                opacity_epsilon=1e-6, variance_clamp_tolerance=1e-6)
            legacy = wrapper.convert_packet_to_source_units(modules, legacy_dp, scale)
            legacy_validation = wrapper.m1_a_gate(modules, legacy, (target["height"], target["width"]))
            raw = source_moments_with_legacy_parity(result["expected_camera_z_packet"].cpu().numpy(),
                result["depth"].cpu().numpy(), legacy_dp, legacy, scale)
            packet, consistency = packet_from_reference(raw, reference)
            stem = wrapper.normalized_stem(target["image_name"])
            legacy_path, packet_path = args.output / "legacy" / (stem + ".npz"), args.output / "v2" / (stem + ".npz")
            modules.packets.deterministic_npz(legacy_path, legacy)
            with packet_path.open("xb") as stream:
                np.savez_compressed(stream, **packet)
            rows.append({"image_name": target["image_name"], "camera": audit,
                "legacy": {"path": legacy_path.relative_to(args.output).as_posix(), "sha256": sha256(legacy_path)},
                "v2": {"path": packet_path.relative_to(args.output).as_posix(), "sha256": sha256(packet_path)},
                "legacy_validation": legacy_validation, "packet_ref": consistency,
                "legacy_raw_accumulator_bitwise_parity": True})
            print(json.dumps({"image": target["image_name"], "completed": len(rows), "total": len(targets)}), flush=True)
            del result, legacy_dp, legacy, raw, packet, camera
    verify_sha(args.checkpoint, args.checkpoint_sha256)
    manifest = {"schema": "umgs_common_graphdeco_proxy_fresh_v2_v1", "scene": args.scene,
        "method_id": args.method_id, "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "dataparser": scene["dataparser"], "registry_sha256": REGISTRY_SHA, "wrapper_sha256": WRAPPER_SHA,
        "reference_identity": reference_identity, "geometry": geometry,
        "extension": {"path": str(extension), "sha256": args.extension_sha256},
        "binding_path": rasterizer.__file__, "binding_sha256": BINDING_SHA, "forward_sha256": FORWARD_SHA,
        "same_render_call_real_H": True, "H_source_formula": "s*H_model_float64_then_float32",
        "legacy_six_arrays_and_numeric_valid_unchanged": True,
        "v2_valid_does_not_replace_legacy_proxy_mask": True,
        "rasterizer_early_termination": "test_T < 1e-4", "native_gsplat_track": False,
        "method_specific_alignment": False, "target_count": len(rows), "packets": rows}
    write_json(args.output / "proxy_export_manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("wrapper", "registry", "checkpoint", "dataparser", "reference_root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("scene", "method_id", "checkpoint_sha256", "dataparser_sha256", "reference_manifest_sha256", "extension_sha256"):
        p.add_argument("--" + name, required=True)
    export(p.parse_args())


if __name__ == "__main__":
    main()
