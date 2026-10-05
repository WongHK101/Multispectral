"""Native UMGS export from authenticated fixed-endpoint RGB and band models.

The source-world frame is unchanged. Geometry/opacity equality is established
before sharing RGB moment packets with the four frozen-support band stages.
"""
from __future__ import annotations

import argparse
import hashlib
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


def archived_camera_parameters(row, frame, image, source_image):
    """Recover saved camera_to_JSON semantics, never fit a new camera."""
    c2w = np.eye(4)
    c2w[:3, :3] = row["rotation"]
    c2w[:3, 3] = row["position"]
    if not np.isfinite(c2w).all():
        raise ValueError("Nonfinite archived camera")
    np.testing.assert_allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(c2w[:3, :3]), 1, rtol=0, atol=1e-9)
    expected = np.asarray(frame["transform_matrix"], dtype=np.float64) @ np.diag([1., -1., -1., 1.])
    np.testing.assert_allclose(c2w, expected, rtol=0, atol=1e-9)
    ow, oh = row["width"], row["height"]
    w, h = image["width"], image["height"]
    if (ow, oh) != (source_image["width"], source_image["height"]) or (round(ow/8), round(oh/8)) != (w, h):
        raise ValueError("Archived R8 dimensions differ from frozen adapter")
    if min(row["fx"], row["fy"], ow, oh) <= 0:
        raise ValueError("Invalid archived focal length")
    np.testing.assert_allclose([row["fx"]*w/ow, row["fy"]*h/oh, w/2, h/2],
        [frame[k] for k in ("fl_x", "fl_y", "cx", "cy")], rtol=0, atol=1e-9)
    view = np.linalg.inv(c2w)
    np.testing.assert_allclose(view, np.linalg.inv(expected), rtol=0, atol=1e-9)
    return view


def validate_model_artifact(row):
    kind = row.get("artifact_kind", "checkpoint_and_final_ply")
    if kind == "archived_final_ply":
        if row.get("checkpoint") or row.get("checkpoint_sha256"):
            raise ValueError("Archived PLY must not claim checkpoint availability")
        if not row.get("archive_authority_manifest_sha256"):
            raise ValueError("Archived PLY requires frozen authority binding")
    elif kind != "checkpoint_and_final_ply":
        raise ValueError("Unknown model artifact kind")
    for key in (("ply_sha256", "archive_authority_manifest_sha256") if kind == "archived_final_ply"
                else ("ply_sha256", "checkpoint_sha256")):
        digest = row[key]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid model authority SHA")
    return kind


def verify_archive_binding(job, groups):
    authority = job["archive_authority"]
    verify_sha(authority["manifest"], authority["sha256"])
    entries = {}
    for line in Path(authority["manifest"]).read_text().splitlines():
        digest, size, name = line.split("\t", 2)
        if name in entries:
            raise ValueError("Duplicate archive authority member")
        entries[name] = (digest, int(size))
    root = Path(authority["payload_root"]).resolve()
    bound = []
    for row in job["models"].values():
        if row["archive_authority_manifest_sha256"] != authority["sha256"]:
            raise ValueError("Model archive authority mismatch")
        bound.append((row["ply"], row["ply_sha256"]))
    for key in ("archived_cameras", "archived_training_audit"):
        row = job[key]; bound.append((row["path"], row["sha256"]))
        verify_sha(row["path"], row["sha256"])
    for path, digest in bound:
        p = Path(path).resolve(); name = p.relative_to(root).as_posix()
        if entries.get(name) != (digest, p.stat().st_size):
            raise ValueError("Asset does not match historical archive authority")
    audit = json.loads(Path(job["archived_training_audit"]["path"]).read_text())
    for split, key in (("train", "train_cameras"), ("eval", "test_cameras")):
        names = sorted(n for n, g in groups.items() if g["split"] == split)
        # Historical training audit hashes sorted names joined without a trailing newline.
        digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
        if audit[key]["count"] != len(names) or audit[key]["names_sha256"] != digest:
            raise ValueError("Archived training/eval identity differs")
        sizes = sorted({str(groups[n]["output_images"]["D"]["width"])+"x"+
                        str(groups[n]["output_images"]["D"]["height"]) for n in names})
        if sorted(audit[key]["image_sizes"]) != sizes:
            raise ValueError("Archived training resolution differs")
    if audit["iterations"] != job["models"]["D"]["iteration"]:
        raise ValueError("Archived endpoint identity differs")


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
    if any(row.get("artifact_kind") == "archived_final_ply" for row in job["models"].values()):
        if not all(row.get("artifact_kind") == "archived_final_ply" for row in job["models"].values()):
            raise ValueError("Mixed archive and checkpoint model sets")
        verify_archive_binding(job, groups)
    meta = json.loads((data / "transforms.json").read_text())
    verify_sha(data / "transforms.json", manifest["artifacts"]["transforms_json"]["sha256"])
    frames = {f["file_path"]: f for f in meta["frames"]}
    sys.path.insert(0, str(method))
    import torch
    import gaussian_renderer
    import diff_gaussian_rasterization as rasterizer
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import CameraInfo, readColmapSceneInfo
    from utils.camera_utils import loadCam
    from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
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
        artifact_kind = validate_model_artifact(row)
        verify_sha(row["ply"], row["ply_sha256"])
        captured = None
        if artifact_kind == "checkpoint_and_final_ply":
            verify_sha(row["checkpoint"], row["checkpoint_sha256"])
            captured, iteration = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
            if iteration != row["iteration"] or len(captured) != 12 or captured[0] != 3:
                raise ValueError("Unexpected UMGS checkpoint endpoint/schema/SH")
        model = GaussianModel(3)
        model.load_ply(row["ply"], use_train_test_exp=False)
        props = (model._xyz, model._features_dc, model._features_rest, model._scaling, model._rotation, model._opacity)
        if model.active_sh_degree != 3 or props[1].shape[1:] != (1, 3) or props[2].shape[1:] != (15, 3):
            raise ValueError("Unexpected final PLY SH schema")
        for i, actual in enumerate(props):
            if not torch.isfinite(actual).all():
                raise ValueError("Nonfinite final PLY property")
            if captured is not None and not torch.equal(actual.detach().cpu(), captured[i+1]):
                raise ValueError("PLY does not exactly reproduce checkpoint")
            actual.requires_grad_(False)
        support = dict(zip(SUPPORT_FIELDS, [props[i].detach().cpu().numpy() for i in (0, 3, 4, 5)]))
        identity = support_identity(support) if support_anchor is None else require_locked_support(support_anchor, support)
        if support_anchor is None:
            support_anchor = {k: v.copy() for k, v in support.items()}
        if channel != "D":
            for values in props[1:3]:
                if not torch.equal(values[..., 0], values[..., 1]) or not torch.equal(values[..., 0], values[..., 2]):
                    raise ValueError("Band SH carrier channels are not tied")
        models[channel] = model
        model_records[channel] = dict(row, artifact_kind=artifact_kind, support=identity,
            checkpoint_parity_verified=captured is not None, ply_load_finite=True,
            native_sh_degree=model.active_sh_degree)
        del captured, props, support

    recovery = {}
    if job.get("archived_cameras"):
        source = job["archived_cameras"]
        verify_sha(source["path"], source["sha256"])
        saved = json.loads(Path(source["path"]).read_text())
        if len({r["img_name"] for r in saved}) != len(saved) or {r["img_name"] for r in saved} != set(groups):
            raise ValueError("Archived camera population differs")
        camera_infos = {}
        for row in saved:
            name = row["img_name"]; group = groups[name]; image = group["output_images"]["D"]
            frame = frames[image["relative_path"]]
            view = archived_camera_parameters(row, frame, image, group["source_images"]["D"])
            actual_c2w = np.eye(4); actual_c2w[:3,:3] = row["rotation"]; actual_c2w[:3,3] = row["position"]
            expected_c2w = np.asarray(frame["transform_matrix"]) @ np.diag([1.,-1.,-1.,1.])
            recovery[name] = dict(source="archived_camera_to_JSON_OpenCV_c2w_and_center",
                source_record=row, recovered_w2c_float64=view.tolist(),
                source_c2w_max_abs_error=float(np.max(np.abs(actual_c2w-expected_c2w))),
                source_w2c_max_abs_error=float(np.max(np.abs(view-np.linalg.inv(expected_c2w)))),
                pose_tolerance=1e-9, json_id_is_not_colmap_id=True)
            camera_infos[name] = CameraInfo(uid=group["camera_id"], R=view[:3,:3].T, T=view[:3,3],
                FovY=focal2fov(row["fy"], row["height"]), FovX=focal2fov(row["fx"], row["width"]),
                depth_params=None, image_path=str(data/image["relative_path"]), image_name=name, depth_path="",
                width=row["width"], height=row["height"], is_test=group["split"] == "eval")
    else:
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
        if name in recovery:
            record["archive_camera_recovery"] = recovery[name]
        record["record_sha256"] = record_hash(record)
        camera_rows.append(record)
        if not args.appearance_only:
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
    expected_packets = 0 if args.appearance_only else manifest["counts"]["registered_image_groups"]
    if len(packet_rows) != expected_packets or len(appearance_rows) != manifest["counts"]["eval_image_groups"]*5:
        raise ValueError("Incomplete export population")
    write_json(args.output / "camera_records.json", camera_rows)
    write_json(args.output / "support_lock.json", model_records)
    result = {"schema": "umgs_graphdeco_native_checkpoint_export_v1", "status": "EXPORTED_PENDING_SCIENTIFIC_SCORING",
        "method": "umgs", "scene": manifest["scene_id"], "checkpoint": job["models"]["D"].get("checkpoint"),
        "checkpoint_sha256": job["models"]["D"].get("checkpoint_sha256"), "models": model_records,
        "model_artifact_kind": model_records["D"]["artifact_kind"], "archived_cameras": job.get("archived_cameras"),
        "archive_authority": job.get("archive_authority"), "archived_training_audit": job.get("archived_training_audit"),
        "render_state": {"sh_degree": 3, "background": "black", "use_trained_exp": False,
                         "geometry_or_sh_mutation": False},
        "export_scope": "heldout_appearance_only" if args.appearance_only else "native_packets_and_heldout_appearance",
        "generator_source_sha256": sha256(__file__), "job_sha256": args.job_sha256, "reference": reference_identity,
        "data_root": str(data), "adapter_manifest_sha256": job["adapter_manifest_sha256"],
        "packet_schema": None if args.appearance_only else "ms_gcp_metric_depth_packet_v2",
        "primary_tensor": None if args.appearance_only else "alpha_normalized_expected_camera_z",
        "semantics": None if args.appearance_only else "camera_z", "formula": None if args.appearance_only else "M1/A",
        "packet_units": None if args.appearance_only else "source_model_not_survey_metres",
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
    p.add_argument("--appearance-only", action="store_true")
    export(p.parse_args())


if __name__ == "__main__":
    main()
