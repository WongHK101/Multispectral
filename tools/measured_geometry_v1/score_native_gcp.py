"""CPU GCP scoring of fresh native MS packets with a shared source transform.

Only the native camera/unit boundary is method-specific. Sampling, coverage,
group aggregation and residual statistics call the frozen reference unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from .camera_bridge import Pinhole
from .common_geometry import unique_rows, validate_source_sidecar
from .contracts import record_hash, safe_member, sha256, verify_sha, verify_source_snapshot
from .preflight import _load_reference_module


def source_camera_mapping(camera):
    v = np.asarray(camera["model_w2c_opencv"], dtype=np.float64)
    t = np.asarray(camera["normalization"]["transform"], dtype=np.float64)
    s = float(camera["normalization"]["scale"])
    if v.shape != (4, 4) or t.shape != (3, 4) or not np.isfinite(v).all() or not np.isfinite(t).all() or not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid native normalization/camera")
    forward = np.eye(4)
    forward[:3] = s * t
    result = v @ forward
    result[:3] /= s
    np.testing.assert_allclose(result, camera["source_w2c_opencv_inverse_normalized"], rtol=0, atol=1e-12)
    if not np.isfinite(result).all() or abs(np.linalg.det(result[:3, :3])) < 1e-10:
        raise ValueError("Singular native camera mapping")
    return result


def native_source_point(mapping, ray, source_z):
    ray = np.asarray(ray, dtype=np.float64)
    if ray.shape != (2,) or not np.isfinite(ray).all() or not np.isfinite(source_z) or source_z <= 0:
        raise ValueError("Invalid native point")
    return np.linalg.solve(mapping[:3, :3], float(source_z) * np.r_[ray, 1.] - mapping[:3, 3])


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(clean_json(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            row = clean_json(row)
            w.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def score(args):
    if args.output.exists():
        raise ValueError("Scoring output exists")
    if any(args.output.resolve() == root.resolve() or root.resolve() in args.output.resolve().parents
           for root in (args.reference_root, args.export_root)):
        raise ValueError("Scoring must not write into an input")
    started = time.monotonic()
    for path, expected in ((args.sidecar, args.sidecar_sha256), (args.geometry, args.geometry_sha256),
                           (args.targets, args.targets_sha256), (args.export_root / "export_manifest.json", args.export_manifest_sha256)):
        verify_sha(path, expected)
    verify_source_snapshot(args.reference_root, args.reference_manifest_sha256)
    protocol = _load_reference_module(args.reference_root, "m3m_native_quarter_protocol")
    packet_reference = _load_reference_module(args.reference_root, "metric_depth_packet")
    sidecar = json.loads(args.sidecar.read_text())
    source_views, canonical = validate_source_sidecar(sidecar)
    geometry = json.loads(args.geometry.read_text())
    if record_hash({k: v for k, v in geometry.items() if k != "records_root_sha256"}) != geometry["records_root_sha256"]:
        raise ValueError("Common geometry records hash mismatch")
    if geometry["scene"] != sidecar["scene"] or geometry["transform"]["method_specific_fit"] or geometry["transform"]["lidar_used_in_fit"]:
        raise ValueError("Invalid shared source geometry")
    transform = protocol.sim3_from_mapping(geometry)
    observations = unique_rows(geometry["observations"], "observation_id")
    if set(observations) != {key for key, row in canonical.items() if row["formal_eligible"]}:
        raise ValueError("Frozen formal population mismatch")
    with args.targets.open(encoding="utf-8-sig", newline="") as f:
        targets = unique_rows(list(csv.DictReader(f)), "point_name")
    fields = ("cgcs2000_gk_cm108_e_m", "cgcs2000_gk_cm108_n_m", "cgcs2000_normal_height_m")
    manifest = json.loads((args.export_root / "export_manifest.json").read_text())
    if manifest["status"] != "EXPORTED_PENDING_SCIENTIFIC_SCORING" or manifest["packet_units"] != "source_model_not_survey_metres":
        raise ValueError("Unsupported export or units")
    if (manifest["packet_schema"], manifest["primary_tensor"], manifest["semantics"], manifest["formula"]) != (
        "ms_gcp_metric_depth_packet_v2", "alpha_normalized_expected_camera_z", "camera_z", "M1/A"):
        raise ValueError("Wrong primary packet contract")
    if manifest["adapter_manifest_sha256"] != sidecar["adapter_manifest_sha256"]:
        raise ValueError("Export is not bound to the approved scene inputs")
    camera_path = args.export_root / "camera_records.json"
    verify_sha(camera_path, manifest["camera_records_sha256"])
    cameras = unique_rows(json.loads(camera_path.read_text()), "image_name")
    packets = unique_rows(manifest["packet_views"], "image_name")
    if set(cameras) != set(packets) or set(cameras) != set(source_views):
        raise ValueError("Incomplete packet/source camera population")
    mappings, pose_precision = {}, []
    flip = np.diag([1., -1., -1., 1.])
    for name, c in cameras.items():
        if record_hash({k: v for k, v in c.items() if k != "record_sha256"}) != c["record_sha256"]:
            raise ValueError("Export camera record hash mismatch")
        if packets[name]["camera_record_sha256"] != c["record_sha256"] or not c["float32_camera_construction_exact"]:
            raise ValueError("Packet/native camera identity mismatch")
        src = source_views[name]
        if c["image_id"] != src["image_id"] or c["camera_id"] != src["camera_id"] or c["native_sha256"] != src["native_sha256"]:
            raise ValueError("Native/source image identity mismatch")
        original = np.asarray(c["source_c2w_opengl_before_float32"]) @ flip
        np.testing.assert_allclose(original, src["source_c2w_opencv"], rtol=0, atol=1e-10)
        mappings[name] = source_camera_mapping(c)
        pose_precision.append({"image_name": name, "exact_input_pose_max_abs_difference": float(np.max(np.abs(original - src["source_c2w_opencv"]))),
            "inverse_float32_runtime_pose_max_abs_difference": float(np.max(np.abs(mappings[name] - src["source_w2c_opencv"]))),
            "runtime_pose_comparison_not_a_new_sfm_fit": True})

    by_image = defaultdict(list)
    for row in observations.values():
        raw = canonical[row["observation_id"]]
        if row["mapping_sha256"] != raw["mapping_sha256"] or row["role"] != raw["formal_role"] or row["point_name"] != raw["point_name"]:
            raise ValueError("Frozen grouping/identity changed")
        np.testing.assert_array_equal(row["source_camera_ray_xy"], raw["old_camera_ray_xy"])
        by_image[row["image_name"]].append(row)
    sampled, valid_by_point, ray_report = [], defaultdict(list), []
    for name in sorted(by_image):
        entry, camera = packets[name], cameras[name]
        path = safe_member(args.export_root, entry["packet"])
        verify_sha(path, entry["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            packet = {key: archive[key] for key in archive.files}
        if packet_reference.recompute_and_compare_packet(packet)["passed"] is not True:
            raise ValueError("Numeric packet/reference consistency failed")
        k = Pinhole(**camera["native_camera_array"])
        if packet["accumulated_alpha"].shape != (k.height, k.width):
            raise ValueError("Actual packet grid mismatch")
        for row in by_image[name]:
            ray = np.asarray(row["source_camera_ray_xy"], dtype=np.float64)
            pixel = k.project([ray])[0]
            recovered = k.unproject([pixel])[0]
            a, b = np.r_[ray, 1.], np.r_[recovered, 1.]
            ray_error = float(np.max(np.abs(ray - recovered)))
            angle = float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))
            if ray_error > 1e-12 or angle > 1e-7 or np.any(pixel < 0) or np.any(pixel >= [k.width, k.height]):
                raise ValueError("Runtime raw-ray mapping/bounds failed")
            sample = protocol.sample_raw_moment_camera_z(packet["accumulated_alpha"], packet["weighted_camera_z_sum"], *pixel)
            sensitivity = protocol.half_pixel_sensitivity(packet["accumulated_alpha"], packet["weighted_camera_z_sum"], *pixel)
            result = {**row, **sample, "packet_sha256": entry["sha256"], "camera_record_sha256": camera["record_sha256"],
                      "half_pixel_max_abs_camera_z_delta_source_units": sensitivity["max_abs_camera_z_delta_model_units"]}
            if sample["valid"]:
                xyz = native_source_point(mappings[name], recovered, sample["camera_z"])
                result.update(source_x=float(xyz[0]), source_y=float(xyz[1]), source_z=float(xyz[2]))
                valid_by_point[row["point_name"]].append({"model_xyz": xyz, "view_class": row["view_class"],
                    "azimuth_bin_45deg": row["azimuth_bin_45deg"], "image_name": name, "observation_id": row["observation_id"]})
            sampled.append(result)
            ray_report.append({"observation_id": row["observation_id"], "image_name": name, "runtime_array_xy": pixel.tolist(),
                "ray_coordinate_error": ray_error, "ray_angle_error_rad": angle,
                "pixel_difference_from_frozen_float64_K": float(np.linalg.norm(pixel - row["native_array_xy"]))})
    points, vectors = [], {"control": [], "checkpoint": [], "all": []}
    roles = {row["point_name"]: row["role"] for row in observations.values()}
    if sorted(n for n, r in roles.items() if r == "control") != geometry["transform"]["fit_controls"]:
        raise ValueError("Frozen control identities mismatch")
    if sorted(n for n, r in roles.items() if r == "checkpoint") != geometry["transform"]["excluded_checkpoints"]:
        raise ValueError("Frozen checkpoint identities mismatch")
    groups = {}
    for name, role in sorted(roles.items()):
        valid = valid_by_point[name]
        gate = protocol.coverage_gate(sum(r["point_name"] == name for r in observations.values()),
            [r["view_class"] for r in valid], [r["azimuth_bin_45deg"] for r in valid])
        point = {"point_name": name, "role": role, **gate}
        if gate["passed"]:
            source, diagnostic = protocol.aggregate_view_groups(valid)
            predicted = transform.apply(source)
            target = np.array([float(targets[name][field]) for field in fields])
            residual = predicted - target
            vectors[role].append(residual)
            vectors["all"].append(residual)
            point.update(source_x=source[0], source_y=source[1], source_z=source[2],
                predicted_e_m=predicted[0], predicted_n_m=predicted[1], predicted_z_m=predicted[2],
                target_e_m=target[0], target_n_m=target[1], target_z_m=target[2],
                residual_e_m=residual[0], residual_n_m=residual[1], residual_z_m=residual[2],
                error_h_m=np.linalg.norm(residual[:2]), error_z_m=abs(residual[2]), error_3d_m=np.linalg.norm(residual),
                aggregation_group_count=diagnostic["group_count"],
                multiview_scatter_median_m=transform.scale*diagnostic["scatter_median_m"],
                multiview_scatter_p90_m=transform.scale*diagnostic["scatter_p90_m"],
                multiview_scatter_max_m=transform.scale*diagnostic["scatter_max_m"])
            groups[name] = {"source_model_unit_aggregation": diagnostic, "survey_metres_per_source_unit": transform.scale}
        points.append(point)
    counts = Counter(roles.values())
    passed = Counter(p["role"] for p in points if p["passed"])
    ranking = protocol.scene_ranking_status(counts["checkpoint"], passed["checkpoint"])
    stats = {role: protocol.residual_statistics(values) for role, values in vectors.items()}
    for role, values in vectors.items():
        stats[role]["p90_3d_m"] = float(np.percentile(np.linalg.norm(values, axis=1), 90)) if values else None
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "observations.csv", sampled)
    write_csv(args.output / "points.csv", points)
    write_csv(args.output / "runtime_ray_mapping.csv", ray_report)
    write_json(args.output / "aggregation_groups.json", groups)
    write_json(args.output / "pose_precision.json", pose_precision)
    # Independent arithmetic starts from the published CSV, not cached vectors.
    with (args.output / "points.csv").open(encoding="utf-8-sig", newline="") as f:
        csv_points = list(csv.DictReader(f))
    independent = {}
    for role in ("control", "checkpoint", "all"):
        rows = [p for p in csv_points if p["passed"] == "True" and (role == "all" or p["role"] == role)]
        if not rows:
            independent[role] = {"count": 0, "status": "NO_VALID_POINTS"}
            continue
        delta = np.array([[float(p["predicted_"+axis+"_m"]) - float(p["target_"+axis+"_m"]) for axis in ("e", "n", "z")] for p in rows])
        h = float(np.sqrt(np.mean(delta[:, 0]**2 + delta[:, 1]**2)))
        z = float(np.sqrt(np.mean(delta[:, 2]**2)))
        d = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
        for key, value in (("rmse_h_m", h), ("rmse_z_m", z), ("rmse_3d_m", d)):
            if abs(value - stats[role][key]) > 1e-9:
                raise ValueError("Independent CSV/JSON residual recomputation mismatch")
        independent[role] = {"count": len(rows), "rmse_h_m": h, "rmse_z_m": z, "rmse_3d_m": d, "status": "PASS"}
    write_json(args.output / "independent_recomputation.json", independent)
    result = {"schema": "umgs_common_transform_gcp_result_v1", "scene": geometry["scene"], "method": manifest["method"],
        **ranking, "stage_review": "PENDING", "whole_matrix_row_complete": False,
        "formal_observations": len(sampled), "valid_observations": sum(r["valid"] for r in sampled),
        "point_counts": dict(counts), "passed_point_counts": dict(passed), "statistics": stats,
        "failures": dict(Counter(r["failure_reason"] for r in sampled if r["failure_reason"])),
        "common_transform": geometry["transform"], "method_specific_transform_fitted": False,
        "native_to_source_boundary": "exact_float64_solve_of_actual_V_and_recorded_parser_T_s",
        "max_ray_coordinate_error": max(r["ray_coordinate_error"] for r in ray_report),
        "max_ray_angle_error_rad": max(r["ray_angle_error_rad"] for r in ray_report),
        "inputs": {"export_manifest_sha256": args.export_manifest_sha256, "sidecar_sha256": args.sidecar_sha256,
                   "geometry_sha256": args.geometry_sha256, "targets_sha256": args.targets_sha256,
                   "reference_manifest_sha256": args.reference_manifest_sha256},
        "script_sha256": sha256(__file__), "gpu_used": False, "wall_seconds": time.monotonic()-started}
    write_json(args.output / "summary.json", result)
    write_json(args.output / "output_hashes.json", [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(args.output.iterdir())])
    print(json.dumps({"status": ranking["status"], "method": manifest["method"], "stats": stats, "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser()
    for name in ("export-root", "sidecar", "geometry", "targets", "reference-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("export-manifest", "sidecar", "geometry", "targets", "reference-manifest"):
        parser.add_argument("--"+name+"-sha256", required=True)
    args = parser.parse_args()
    from .cpu_guard import install_cpu_guard
    install_cpu_guard()
    score(args)


if __name__ == "__main__":
    main()
