"""Read-only final-checkpoint export through the pinned MS native renderer.

This produces new packets and held-out appearance arrays, not scientific scores.
The scheduler owns GPU allocation and the deadline. No train recipe is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .camera_bridge import Pinhole, camera_to_array_intrinsics
from .contracts import record_hash, sha256, verify_sha, verify_source_snapshot
from .native_moments import packet_from_reference, source_unit_wire


def image_lookup(manifest):
    result = {}
    names = set()
    for group in manifest["groups"]:
        if group["image_name"] in names or group["split"] not in {"train", "eval"}:
            raise ValueError("Duplicate image identity or unknown split")
        names.add(group["image_name"])
        for channel, image in group["output_images"].items():
            key = image["relative_path"]
            if key in result:
                raise ValueError("Duplicate adapter image path")
            result[key] = (group, channel, image)
    return result


def json_summary(value):
    if isinstance(value, dict):
        return {k: json_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_summary(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_summary(value), stream, indent=2, ensure_ascii=False, allow_nan=False)


def checkpoint_model_state(pipeline_state):
    result = {}
    for key, value in pipeline_state.items():
        if key.startswith("_model."):
            name = key[len("_model."):]
            if name.startswith("module."):
                name = name[len("module."):]
            if name in result:
                raise ValueError("Duplicate model checkpoint key")
            result[name] = value
    if not result or "gauss_params.means" not in result:
        raise ValueError("Missing actual model checkpoint state")
    return result


def export(args):
    if args.output.exists():
        raise ValueError("Output exists; exports never overwrite")
    head = subprocess.check_output(["git", "-C", str(args.method_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.method_root), "status", "--porcelain"], text=True).strip()
    if head != args.method_commit or dirty:
        raise ValueError("Method source identity/clean status mismatch")
    verify_sha(args.checkpoint, args.checkpoint_sha256)
    reference_identity = verify_source_snapshot(args.reference_root, args.reference_manifest_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "packets").mkdir()
    (args.output / "appearance").mkdir()
    started = time.time()

    import torch
    from nerfstudio.cameras import camera_utils
    from gsplat.rendering import rasterization
    from mmsplat.util.eval_utils import eval_setup
    from mmsplat.util.utils import get_viewmat
    from .live_packet_adapter import gsplat_moments
    from .preflight import _load_reference_module

    reference = _load_reference_module(args.reference_root, "metric_depth_packet")
    # This exact locally trained checkpoint was authenticated above. Upstream
    # omits weights_only and predates PyTorch 2.6's changed default.
    old_load_policy = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        config, pipeline, checkpoint_path, step = eval_setup(args.config, load_step=args.step, test_mode="test")
    finally:
        if old_load_policy is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_load_policy
    if Path(checkpoint_path).resolve() != args.checkpoint.resolve() or step != args.step:
        raise ValueError("Loader selected a different checkpoint")
    model = pipeline.model
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["step"] != step:
        raise ValueError("Checkpoint step mismatch")
    expected = checkpoint_model_state(saved["pipeline"])
    # Upstream pipeline retries non-strict loading; explicitly prohibit that here.
    model.load_state_dict(expected, strict=True)
    model.step = step  # Upstream model.load_state_dict resets it to 30000.
    model.eval()
    actual = model.state_dict()
    if set(actual) != set(expected):
        raise ValueError("Model state keys differ after strict reload")
    for key, value in actual.items():
        target = expected[key]
        if value.dtype != target.dtype or not torch.equal(value.detach().cpu(), target):
            raise ValueError("Model state not exactly restored: " + key)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite final checkpoint: " + key)
    del saved, expected, actual

    dm = pipeline.datamanager
    data = Path(dm.config.dataparser.data)
    if data.suffix == ".json":
        data = data.parent
    manifest = json.loads((data / "manifest.json").read_text())
    lookup = image_lookup(manifest)
    meta = json.loads((data / "transforms.json").read_text())
    frames = {f["file_path"]: f for f in meta["frames"]}
    if len(frames) != len(meta["frames"]) or set(frames) != set(lookup):
        raise ValueError("Transforms/adapter frame identity mismatch")
    norm_path = args.config.parent / "dataparser_transforms.json"
    normalization = json.loads(norm_path.read_text())
    t = np.asarray(normalization["transform"], dtype=np.float64)
    scale = float(normalization["scale"])
    if t.shape != (3, 4) or not np.isfinite(t).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid recorded normalization")
    transform = np.eye(4)
    transform[:3] = t
    # Reproduce the original batch operation, not single-matrix multiplication:
    # float32 BLAS accumulation order is part of the actual parser construction.
    ordered_frames = sorted(frames)
    input_poses = torch.from_numpy(np.asarray([frames[name]["transform_matrix"] for name in ordered_frames], dtype=np.float32))
    reproduced_poses, reproduced_transform = camera_utils.auto_orient_and_center_poses(
        input_poses, method=meta.get("orientation_override", dm.config.dataparser.orientation_method),
        center_method=dm.config.dataparser.center_method)
    reproduced_poses[:, :3, 3] *= scale
    np.testing.assert_array_equal(reproduced_transform.cpu().numpy(), t.astype(np.float32))
    reproduced_by_path = dict(zip(ordered_frames, reproduced_poses[:, :3, :4]))
    seen = set()
    packet_rows, appearance_rows, camera_rows = [], [], []
    loaded_counts = {}

    for split, outputs, dataset in (
        ("train", dm.train_dataparser_outputs, dm.train_dataset),
        ("eval", dm.eval_dataparser_outputs, dm.eval_dataset),
    ):
        np.testing.assert_array_equal(outputs.dataparser_transform.cpu().numpy(), t.astype(np.float32))
        if outputs.dataparser_scale != scale:
            raise ValueError("Current vs saved dataparser scale mismatch")
        loaded_counts[split] = len(outputs.image_filenames)
        for index, (path, channel) in enumerate(zip(outputs.image_filenames, outputs.metadata["mm_channel"])):
            relative = Path(path).relative_to(data).as_posix()
            if relative in seen:
                raise ValueError("Duplicate or leaked actual parser frame")
            seen.add(relative)
            group, expected_channel, image = lookup[relative]
            if channel != expected_channel or split != group["split"]:
                raise ValueError("Parser channel/split differs from frozen adapter")
            frame = frames[relative]
            camera = copy.deepcopy(dataset.cameras[index:index + 1])
            camera.metadata = {"cam_idx": index, "mm_channel": channel}
            k = camera.get_intrinsics_matrices()[0].cpu().numpy()
            width, height = int(camera.width.item()), int(camera.height.item())
            if (width, height) != (image["width"], image["height"]):
                raise ValueError("Parser changed native dimensions")
            expected_k = np.array([[frame["fl_x"], 0, frame["cx"]],
                                   [0, frame["fl_y"], frame["cy"]], [0, 0, 1]], dtype=np.float32)
            np.testing.assert_array_equal(k, expected_k)
            if not torch.equal(camera.camera_to_worlds[0].cpu(), reproduced_by_path[relative]):
                raise ValueError("Actual parser pose is not the exact frozen float32 construction")
            native = Pinhole(width, height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
            array_k = camera_to_array_intrinsics(native, sampling_convention="camera_corner_origin_samples_at_half_v1")

            if channel == "D":
                verify_sha(path, image["sha256"])
                with torch.no_grad():
                    model_w2c = get_viewmat(camera.camera_to_worlds.to(model.device))[0].detach().cpu().numpy().astype(np.float64)
                # Model XYZ=s*(T@source XYZ); normalized camera-z is scaled once.
                source_to_model = transform.copy()
                source_to_model[:3] *= scale
                source_w2c = (model_w2c @ source_to_model).copy()
                source_w2c[:3] /= scale
                camera_record = {
                    "image_name": group["image_name"], "image_id": group["image_id"],
                    "camera_id": group["camera_id"], "split": split,
                    "native_file": relative, "native_sha256": image["sha256"],
                    "renderer_corner_K_float32": k.tolist(), "native_camera_array": vars(array_k),
                    "model_camera_to_world_opengl_float32": camera.camera_to_worlds[0].cpu().numpy().tolist(),
                    "model_w2c_opencv": model_w2c.tolist(),
                    "source_w2c_opencv_inverse_normalized": source_w2c.tolist(),
                    "source_backprojection": "solve(M,z_source*ray-d); M=B@R; d=B@t+b/s; no transpose approximation or second inverse normalization",
                    "float32_camera_construction_exact": True,
                    "source_c2w_opengl_before_float32": frame["transform_matrix"],
                    "normalization": normalization,
                    "source_pose_precision_status": "EXPLICIT_FLOAT32_PARSER_PENDING_SCORING_GATE",
                    "pixel_conversion": "renderer_samples_j_plus_half_to_array_index_once_v1",
                }
                camera_record["record_sha256"] = record_hash(camera_record)
                camera_rows.append(camera_record)
                raw = gsplat_moments(native_rasterization=rasterization, native_get_viewmat=get_viewmat,
                                     model=model, camera=camera)
                wire, unit_record = source_unit_wire(raw, scale)
                packet, comparison = packet_from_reference(wire, reference)
                packet_path = args.output / "packets" / (Path(group["image_name"]).stem + ".npz")
                with packet_path.open("xb") as stream:
                    np.savez_compressed(stream, **packet)
                packet_rows.append({"image_name": group["image_name"], "split": split,
                    "packet": packet_path.relative_to(args.output).as_posix(), "sha256": sha256(packet_path),
                    "bytes": packet_path.stat().st_size, "camera_record_sha256": camera_record["record_sha256"],
                    "width": width, "height": height, "unit_conversion": unit_record,
                    "packet_ref": json_summary(comparison)})
                print(json.dumps({"phase": "packet", "completed": len(packet_rows), "image": group["image_name"]}), flush=True)

            if split == "eval":
                with torch.no_grad():
                    rendered = model.get_outputs(camera)["render"].detach().cpu().numpy()
                if rendered.shape != (height, width, 3 if channel == "D" else 1) or not np.isfinite(rendered).all():
                    raise ValueError("Appearance shape/nonfinite mismatch")
                appearance_path = args.output / "appearance" / (Path(group["image_name"]).stem + "__" + channel + ".npy")
                with appearance_path.open("xb") as stream:
                    np.save(stream, rendered, allow_pickle=False)
                appearance_rows.append({"image_name": group["image_name"], "channel": channel,
                    "prediction": appearance_path.relative_to(args.output).as_posix(), "sha256": sha256(appearance_path),
                    "shape": list(rendered.shape), "dtype": str(rendered.dtype),
                    "gt_native_file": relative, "gt_sha256": image["sha256"],
                    "common_mask": group["output_masks"]["common"], "scored": False})
    if seen != set(lookup):
        raise ValueError("Missing actual parser frames")
    if len(packet_rows) != len(manifest["groups"]) or len(appearance_rows) != loaded_counts["eval"]:
        raise ValueError("Incomplete export population")
    write_json(args.output / "camera_records.json", camera_rows)
    result = {"schema": "umgs_ms_native_checkpoint_export_v1", "status": "EXPORTED_PENDING_SCIENTIFIC_SCORING",
        "method": args.method_id, "scene": manifest["scene_id"], "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256, "checkpoint_step": step,
        "config": str(args.config), "config_sha256": sha256(args.config),
        "method_commit": head, "method_clean": True, "generator_source_sha256": sha256(__file__),
        "reference": reference_identity, "data_root": str(data), "adapter_manifest_sha256": sha256(data / "manifest.json"),
        "normalization": normalization, "normalization_sha256": sha256(norm_path),
        "loaded_frame_counts": loaded_counts, "gaussian_count": int(model.gauss_params["means"].shape[0]),
        "packet_schema": "ms_gcp_metric_depth_packet_v2", "primary_tensor": "alpha_normalized_expected_camera_z",
        "semantics": "camera_z", "formula": "M1/A", "packet_units": "source_model_not_survey_metres",
        "packet_views": packet_rows, "appearance_views": appearance_rows,
        "camera_records_sha256": sha256(args.output / "camera_records.json"),
        "wall_seconds": time.time() - started, "training_run": False, "formal_metrics_generated": False}
    write_json(args.output / "export_manifest.json", result)
    print(json.dumps({"status": result["status"], "packets": len(packet_rows), "appearance_arrays": len(appearance_rows),
                      "wall_seconds": result["wall_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "checkpoint", "method-root", "reference-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("checkpoint-sha256", "method-commit", "reference-manifest-sha256", "method-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--step", type=int, default=119999)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
