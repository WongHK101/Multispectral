"""Native UMGS export from authenticated fixed-endpoint RGB and band models.

The source-world frame is unchanged. Geometry/opacity equality is established
before sharing RGB moment packets with the four frozen-support band stages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .contracts import record_hash, sha256, verify_sha, verify_source_snapshot
from .ms_checkpoint_export import write_json, json_summary
from .native_moments import packet_from_reference, source_unit_wire

SUPPORT_FIELDS = ("xyz", "scaling", "rotation", "opacity")


def support_identity(values):
    import hashlib
    if set(values) != set(SUPPORT_FIELDS):
        raise ValueError("Require complete ordered geometry/opacity")
    n = len(values["xyz"])
    result = {}
    for key, width in zip(SUPPORT_FIELDS, (3, 3, 4, 1)):
        a = np.asarray(values[key])
        if n == 0 or a.shape != (n, width) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Invalid Gaussian property: " + key)
        result[key] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    return {"gaussian_count": n, "ordered_property_sha256": result}


def require_locked_support(anchor, candidate):
    a, b = support_identity(anchor), support_identity(candidate)
    if a != b:
        raise ValueError("Band changed Gaussian order/count/geometry/opacity")
    return a


def export(args):
    started = time.monotonic()
    verify_sha(args.job, args.job_sha256)
    job = json.loads(args.job.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    reference_identity = verify_source_snapshot(args.reference_root, args.reference_manifest_sha256)
    method = Path(job["method_root"])
    for name, digest in job["method_source_sha256"].items():
        verify_sha(method / name, digest)
    data = Path(job["data_root"])
    verify_sha(data / "manifest.json", job["adapter_manifest_sha256"])
    manifest = json.loads((data / "manifest.json").read_text())
    groups = {g["image_name"]: g for g in manifest["groups"]}
    if len(groups) != len(manifest["groups"]) or set(job["models"]) != {"D", "MS_G", "MS_R", "MS_RE", "MS_NIR"}:
        raise ValueError("Incomplete/duplicate native scene or model set")
    meta = json.loads((data / "transforms.json").read_text())
    frames = {f["file_path"]: f for f in meta["frames"]}
    sys.path.insert(0, str(method))
    import torch
    import gaussian_renderer
    import diff_gaussian_rasterization as rasterizer
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import loadCam
    from utils.graphics_utils import fov2focal, getWorld2View2
    from .live_packet_adapter import graphdeco_moments
    from .preflight import _load_reference_module

    if Path(gaussian_renderer.__file__).resolve() != (method / "gaussian_renderer/__init__.py").resolve():
        raise ValueError("Unexpected native renderer")
    verify_sha(rasterizer.__file__, job["rasterizer_binding_sha256"])
    verify_sha(rasterizer._C.__file__, job["rasterizer_extension_sha256"])
    reference = _load_reference_module(args.reference_root, "metric_depth_packet")
    models, model_records, support_anchor = {}, {}, None
    for channel in ("D", "MS_G", "MS_R", "MS_RE", "MS_NIR"):
        row = job["models"][channel]
        verify_sha(row["checkpoint"], row["checkpoint_sha256"])
        verify_sha(row["ply"], row["ply_sha256"])
        captured, iteration = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        if iteration != row["iteration"] or len(captured) != 12 or captured[0] != 3:
            raise ValueError("Unexpected UMGS checkpoint endpoint/schema/SH")
        model = GaussianModel(3)
        model.load_ply(row["ply"], use_train_test_exp=False)
        props = (model._xyz, model._features_dc, model._features_rest, model._scaling, model._rotation, model._opacity)
        for actual, expected in zip(props, captured[1:7]):
            if not torch.equal(actual.detach().cpu(), expected) or not torch.isfinite(expected).all():
                raise ValueError("PLY does not exactly reproduce checkpoint")
            actual.requires_grad_(False)
        support = dict(zip(SUPPORT_FIELDS, [captured[i].detach().numpy() for i in (1, 4, 5, 6)]))
        identity = support_identity(support) if support_anchor is None else require_locked_support(support_anchor, support)
        if support_anchor is None:
            support_anchor = {k: v.copy() for k, v in support.items()}
        if channel != "D":
            for values in props[1:3]:
                if not torch.equal(values[..., 0], values[..., 1]) or not torch.equal(values[..., 0], values[..., 2]):
                    raise ValueError("Band SH carrier channels are not tied")
        models[channel] = model
        model_records[channel] = dict(row, support=identity)
        del captured, props, support

    scene_root = Path(job["rgb_input"])
    for name, digest in job["rgb_input_metadata_sha256"].items():
        verify_sha(scene_root / "sparse/0" / name, digest)
    info = readColmapSceneInfo(str(scene_root), "images", "", True, False)
    camera_infos = {c.image_name: c for c in info.train_cameras + info.test_cameras}
    if set(camera_infos) != set(groups):
        raise ValueError("Native loader/source camera population differs")
    opts = SimpleNamespace(resolution=1, data_device="cpu", train_test_exp=False, modality_kind="rgb",
        target_band="", single_band_mode=False, single_band_replicate_to_rgb=True, input_dynamic_range="uint8", radiometric_mode="raw_dn")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    args.output.mkdir(parents=True)
    (args.output / "packets").mkdir()
    (args.output / "appearance").mkdir()
    packet_rows, appearance_rows, camera_rows = [], [], []
    normalization = {"transform": np.eye(4)[:3].tolist(), "scale": 1.0, "convention": "identity_source_world_no_dataparser_rescale"}
    for index, (name, group) in enumerate(sorted(groups.items())):
        image = group["output_images"]["D"]
        ci = camera_infos[name]
        verify_sha(ci.image_path, image["sha256"])
        cam = loadCam(opts, index, ci, 1.0, False, group["split"] == "eval")
        w, h = cam.image_width, cam.image_height
        if (w, h) != (image["width"], image["height"]) or ci.is_test != (group["split"] == "eval"):
            raise ValueError("Native camera dimensions/split changed")
        view = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        np.testing.assert_array_equal(view, getWorld2View2(ci.R, ci.T))
        k = dict(width=w, height=h, fx=fov2focal(cam.FoVx, w), fy=fov2focal(cam.FoVy, h), cx=(w-1)/2, cy=(h-1)/2)
        frame = frames[image["relative_path"]]
        np.testing.assert_allclose([k["fx"], k["fy"], k["cx"]+.5, k["cy"]+.5],
                                   [frame[f] for f in ("fl_x", "fl_y", "cx", "cy")], rtol=0, atol=1e-9)
        record = {"image_name": name, "image_id": group["image_id"], "camera_id": group["camera_id"], "split": group["split"],
            "native_file": image["relative_path"], "native_sha256": image["sha256"], "native_camera_array": k,
            "model_w2c_opencv": view.tolist(), "source_w2c_opencv_inverse_normalized": view.tolist(),
            "source_c2w_opengl_before_float32": frame["transform_matrix"], "normalization": normalization,
            "float32_camera_construction_exact": True, "camera_construction": "unchanged_UMGS_loadCam_and_Camera",
            "projection_matrix_transposed": cam.projection_matrix.detach().cpu().numpy().tolist(),
            "full_proj_transform_transposed": cam.full_proj_transform.detach().cpu().numpy().tolist(),
            "pixel_conversion": "graphdeco_ndc_to_array_((ndc+1)*size-1)/2_v1"}
        record["record_sha256"] = record_hash(record)
        camera_rows.append(record)
        raw = graphdeco_moments(native_render=gaussian_renderer.render, camera=cam, gaussians=models["D"], pipeline=pipeline, background=background)
        wire, units = source_unit_wire(raw, 1.0)
        packet, comparison = packet_from_reference(wire, reference)
        p = args.output / "packets" / (Path(name).stem + ".npz")
        with p.open("xb") as stream:
            np.savez_compressed(stream, **packet)
        packet_rows.append({"image_name": name, "split": group["split"], "packet": p.relative_to(args.output).as_posix(),
            "sha256": sha256(p), "bytes": p.stat().st_size, "camera_record_sha256": record["record_sha256"],
            "width": w, "height": h, "unit_conversion": units, "packet_ref": json_summary(comparison)})
        if group["split"] == "eval":
            for channel, model in models.items():
                with torch.no_grad():
                    rendered = gaussian_renderer.render(cam, model, pipeline, background, use_trained_exp=False)["render"].permute(1, 2, 0).cpu().numpy()
                if channel != "D":
                    np.testing.assert_array_equal(rendered[..., 0], rendered[..., 1])
                    np.testing.assert_array_equal(rendered[..., 0], rendered[..., 2])
                    rendered = rendered[..., :1].copy()
                if not np.isfinite(rendered).all():
                    raise ValueError("Nonfinite native appearance")
                p = args.output / "appearance" / (Path(name).stem + "__" + channel + ".npy")
                with p.open("xb") as stream:
                    np.save(stream, rendered, allow_pickle=False)
                gt = group["output_images"][channel]
                appearance_rows.append({"image_name": name, "channel": channel, "prediction": p.relative_to(args.output).as_posix(),
                    "sha256": sha256(p), "shape": list(rendered.shape), "dtype": str(rendered.dtype),
                    "gt_native_file": gt["relative_path"], "gt_sha256": gt["sha256"], "common_mask": group["output_masks"]["common"], "scored": False})
        print(json.dumps({"phase": "packet", "completed": index+1, "image": name}), flush=True)
        del cam
    if len(packet_rows) != manifest["counts"]["registered_image_groups"] or len(appearance_rows) != manifest["counts"]["eval_image_groups"]*5:
        raise ValueError("Incomplete export population")
    write_json(args.output / "camera_records.json", camera_rows)
    write_json(args.output / "support_lock.json", model_records)
    result = {"schema": "umgs_graphdeco_native_checkpoint_export_v1", "status": "EXPORTED_PENDING_SCIENTIFIC_SCORING",
        "method": "umgs", "scene": manifest["scene_id"], "checkpoint": job["models"]["D"]["checkpoint"],
        "checkpoint_sha256": job["models"]["D"]["checkpoint_sha256"], "models": model_records,
        "generator_source_sha256": sha256(__file__), "job_sha256": args.job_sha256, "reference": reference_identity,
        "data_root": str(data), "adapter_manifest_sha256": job["adapter_manifest_sha256"],
        "packet_schema": "ms_gcp_metric_depth_packet_v2", "primary_tensor": "alpha_normalized_expected_camera_z",
        "semantics": "camera_z", "formula": "M1/A", "packet_units": "source_model_not_survey_metres",
        "packet_views": packet_rows, "appearance_views": appearance_rows, "normalization": normalization,
        "camera_records_sha256": sha256(args.output / "camera_records.json"), "support_lock_sha256": sha256(args.output / "support_lock.json"),
        "rgb_anchor_geometry_shared_only_after_bitwise_support_check": True,
        "wall_seconds": time.monotonic()-started, "training_run": False, "formal_metrics_generated": False}
    write_json(args.output / "export_manifest.json", result)
    print(json.dumps({"status": result["status"], "packets": len(packet_rows), "appearance_arrays": len(appearance_rows)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ("job", "reference-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("job-sha256", "reference-manifest-sha256"):
        p.add_argument("--"+name, required=True)
    export(p.parse_args())


if __name__ == "__main__":
    main()
