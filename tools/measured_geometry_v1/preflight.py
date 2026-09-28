"""Read-only, local UMGS GCP input adaptation. No training/rendering/scoring path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from . import cpu_guard


CORE_SCENES = ("gcp_3000_20260602", "gcp_5000_20260602")


def _load_reference_module(root, name):
    path = root / "code/gcp" / f"{name}.py"
    if name in sys.modules:
        raise ValueError(f"Ambiguous imported reference: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_embedded_model(reference, model):
    cameras, images = {}, {}
    for raw in model["cameras"]:
        record = reference.CameraRecord(int(raw["camera_id"]), raw["model"],
            int(raw["width"]), int(raw["height"]), tuple(map(float, raw["params"])))
        if record.camera_id in cameras or reference.camera_record_hash(record) != raw["record_sha256"]:
            raise ValueError("Duplicate or mismatched camera record")
        cameras[record.camera_id] = SimpleNamespace(id=record.camera_id, model=record.model,
            width=record.width, height=record.height, params=record.params)
    names = set()
    for raw in model["images"]:
        record = reference.ImageRecord(int(raw["image_id"]), raw["image_name"],
            int(raw["camera_id"]), tuple(map(float, raw["qvec"])), tuple(map(float, raw["tvec"])))
        if (record.image_id in images or record.image_name in names
                or record.camera_id not in cameras or Path(record.image_name).name != record.image_name
                or reference.image_pose_record_hash(record) != raw["record_sha256"]):
            raise ValueError("Duplicate/orphan/mismatched pose record")
        images[record.image_id] = SimpleNamespace(id=record.image_id, name=record.image_name,
            camera_id=record.camera_id, qvec=record.qvec, tvec=record.tvec)
        names.add(record.image_name)
    return cameras, images


def verify_raw_images(rows, raw_root):
    from PIL import Image
    from .contracts import safe_member, verify_sha

    records = {}
    for row in rows:
        name = row["raw_image_name"]
        identity = (row["source_image_sha256"], int(row["source_image_width"]),
                    int(row["source_image_height"]), row["source_exif_orientation_raw_value"])
        if name in records and records[name] != identity:
            raise ValueError(f"Conflicting raw image identity: {name}")
        records[name] = identity
    for name, (digest, width, height, orientation) in records.items():
        path = safe_member(raw_root, name)
        verify_sha(path, digest)
        with Image.open(path) as image:
            if image.size != (width, height) or image.mode != "RGB":
                raise ValueError(f"Raw image dimension/mode mismatch: {name}")
            actual_orientation = image.getexif().get(274)
            if ("" if actual_orientation is None else str(actual_orientation)) != orientation:
                raise ValueError(f"Raw EXIF metadata mismatch: {name}")
    return {"status": "PASS_BYTES_AND_IMAGE_HEADERS", "unique_raw_images": len(records),
            "orientation_policy": "ignore_exif_orientation_no_transpose",
            "rgb_pixel_matrix_redecode": "NOT_RUN_NOT_CLAIMED",
            "checks": "file SHA, declared RGB mode, dimensions, EXIF tag; handles closed"}


def audit_release(profile):
    from .contracts import read_json, safe_member, verify_sha, verify_source_snapshot

    reference_root = Path(profile["reference_root"])
    source = verify_source_snapshot(reference_root, profile["reference_manifest_sha256"])
    release = Path(profile["release_root"])
    root_record = release / "v1_3_0_release_root_digest.json"
    verify_sha(root_record, profile["release_root_record_sha256"])
    record = read_json(root_record)
    if record["payload_root_digest_sha256"] != profile["release_payload_root_sha256"]:
        raise ValueError("Release payload root mismatch")
    # Authenticate every file path before the frozen loader traverses the release.
    payload = read_json(safe_member(release, record["payload_manifest_path"]))
    for entry in payload["files"]:
        safe_member(release, entry["path"])
    for name in ("gcp_pixel_domain_v1_2", "gcp_pixel_domain_v1_3"):
        if name in sys.modules:
            raise ValueError("Reference namespace already in use")
    try:
        v12 = _load_reference_module(reference_root, "gcp_pixel_domain_v1_2")
        v13 = _load_reference_module(reference_root, "gcp_pixel_domain_v1_3")
        sidecars = v13.load_release_v13_sidecars(release)
        config = read_json(release / "gcp_benchmark_release_v1_3_0.json")
        if config["schema"] != v13.RELEASE_V130_SCHEMA:
            raise ValueError("Unexpected release schema")
        scenes = []
        if set(profile["raw_roots"]) != set(CORE_SCENES):
            raise ValueError("This revision preflight is scoped to 3K and 5K only")
        for scene in CORE_SCENES:
            camera_scene = sidecars["camera"]["scenes"][scene]
            verify_embedded_model(v12, camera_scene["source_model"])
            cameras, images = verify_embedded_model(v12, camera_scene["target_model"])
            annotation = release / f"{scene}_gcp_annotations_pixel_domain_v1_3_0.csv"
            rows = v12.read_csv(annotation)
            validated = v13.validate_release_v13_rows_for_evaluator(release_base=release,
                scene=scene, rows=rows, colmap_cameras=cameras, colmap_images=images,
                return_all_rows=True)
            formal = [r for r in validated if v13.parse_release_bool(r["formal_eligible"], "formal_eligible")]
            counts = config["frozen_counts"]
            if len(rows) != counts["scene_rows"][scene] or len(formal) != counts["scene_formal_eligible"][scene]:
                raise ValueError("Release scene count mismatch")
            roles = {role: sorted({r["point_name"] for r in formal if r["formal_role"] == role})
                     for role in ("control", "checkpoint")}
            if set(roles["control"]) & set(roles["checkpoint"]):
                raise ValueError("Control/checkpoint leakage")
            norms = []
            for row in formal:
                cam = v12.CameraRecord(int(row["source_camera_id"]), row["source_camera_model"],
                    int(row["source_camera_width"]), int(row["source_camera_height"]),
                    tuple(map(float, row["source_camera_params"].split(";"))))
                x, y, _ = v12.invert_simple_radial(cam, float(row["raw_manual_x"]), float(row["raw_manual_y"]))
                error = max(abs(x - float(row["normalized_x"])), abs(y - float(row["normalized_y"])))
                if error > 1e-12:
                    raise ValueError("Cached normalized ray mismatch")
                norms.append(error)
            scenes.append({"scene": scene, "rows_preserved": len(validated), "formal_rows": len(formal),
                "role_identities": roles, "quality_counts": dict(Counter(r["annotation_quality"] for r in rows)),
                "formal_image_names": sorted({r["raw_image_name"] for r in formal}),
                "max_raw_to_normalized_cache_error": max(norms),
                "raw_images": verify_raw_images(rows, Path(profile["raw_roots"][scene])),
                "camera_source": "authenticated_release_embedded_records_not_live_renderer",
                "packet_camera_pose_validation": "PENDING_REAL_RENDERER_BINDING"})
        return {"reference": source, "release_integrity": sidecars["integrity"], "scenes": scenes}
    finally:
        for name in ("gcp_pixel_domain_v1_2", "gcp_pixel_domain_v1_3"):
            sys.modules.pop(name, None)


def run(profile, output):
    from .contracts import sha256

    output = Path(output).resolve()
    for root in [profile["release_root"], profile["reference_root"], *profile["raw_roots"].values()]:
        if Path(root).resolve() == output or Path(root).resolve() in output.parents:
            raise ValueError("Output must not be inside an input/reference directory")
    if output.exists():
        raise FileExistsError(output)
    result = {"schema": "umgs_measured_geometry_cpu_preflight_v1", "formal_ready": False,
              "training_started": False, "depth_values_read": False,
              "patch_sampling": False, "sim3_fitting": False, "formal_metrics": False}
    exit_code = 0
    try:
        result.update(audit_release(profile))
        result["status"] = "PASS_CPU_INPUT_ADAPTATION_ONLY"
    except Exception as exc:
        exit_code = 1
        result.update(status="BLOCKED_CPU_INPUT_ADAPTATION", error=f"{type(exc).__name__}: {exc}")
    result["execution_boundary"] = cpu_guard.evidence()
    result["adapter_sources"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    result["pending"] = ["explicit_user_gpu_available_message", "remote_checkpoint_availability_and_hash",
        "shared_SfM_actual_model_binding", "method_specific_recipe_and_license_admission",
        "renderer_camera_and_ray_runtime_parity", "metric_packet_v2_export_and_packet_ref_checks"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "report": str(output), "formal_ready": False}))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_guard.install_cpu_guard()
    from .contracts import read_json
    return run(read_json(args.profile), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
