"""CPU native-packet surfaces with the frozen common LiDAR numeric core."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import laspy
import numpy as np
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import box

from .camera_bridge import Pinhole
from .common_geometry import unique_rows, validate_source_sidecar
from .contracts import read_json, record_hash, safe_member, sha256, verify_sha, verify_source_snapshot
from .preflight import _load_reference_module
from .score_native_gcp import source_camera_mapping, write_csv, write_json


def native_surface_samples(packet, camera, mapping, stride=4, alpha_min=0.5):
    k = Pinhole(**camera["native_camera_array"])
    full_shape = (k.height, k.width)
    keys = ("alpha_normalized_expected_camera_z", "accumulated_alpha", "metric_depth_valid_mask")
    if any(packet[key].shape != full_shape for key in keys):
        raise ValueError("Surface packet/camera dimensions differ")
    depth, alpha, valid = (packet[key][::stride, ::stride] for key in keys)
    mask = valid.astype(bool) & np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha >= alpha_min)
    yy, xx = np.meshgrid(np.arange(0, k.height, stride, dtype=np.float64),
                         np.arange(0, k.width, stride, dtype=np.float64), indexing="ij")
    rays = np.c_[(xx[mask] - k.cx) / k.fx, (yy[mask] - k.cy) / k.fy, np.ones(int(mask.sum()))]
    camera_xyz = rays * depth[mask, None].astype(np.float64)
    # Invert the actual float32 renderer composition, not its approximate rotation transpose.
    source = np.linalg.solve(mapping[:3, :3], (camera_xyz - mapping[:3, 3]).T).T
    if not np.isfinite(source).all():
        raise ValueError("Nonfinite reconstructed surface")
    return source, {"sampled_pixels": depth.size, "supported_samples": int(mask.sum())}


def reference_surface(laz_path, metadata, scene, numeric, roi, origin):
    verify_sha(laz_path, scene["actual_roi_laz"]["sha256"])
    if metadata["vertical_shift_applied_m"] != 0.0 or metadata["normal_height_bridge_to_existing_benchmark_m"] != scene["normal_height_bridge_added_once_m"]:
        raise ValueError("LiDAR height bridge identity mismatch")
    if metadata["roi_bounds_xy_m"] != scene["roi_bounds_xy_m"]:
        raise ValueError("ROI metadata/binding mismatch")
    pending, decoded, count, inside_count = [], hashlib.sha256(), 0, 0
    with laspy.open(laz_path) as reader:
        if reader.header.parse_crs().to_epsg() != 32649 or reader.header.point_count != scene["actual_roi_laz"]["point_count"]:
            raise ValueError("LiDAR CRS/count mismatch")
        for points in reader.chunk_iterator(250000):
            decoded.update(points.array.tobytes())
            x, y = np.asarray(points.x, dtype=np.float64), np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64) + scene["normal_height_bridge_added_once_m"]
            mask = contains_xy(roi, x, y)
            pending.append(numeric.voxel_batch_ids(np.c_[x[mask], y[mask], z[mask]], scene["voxel_size_m"], origin))
            count += len(points)
            inside_count += int(mask.sum())
    if count != metadata["point_count"] or decoded.hexdigest() != metadata["decoded_selected_records_sha256"]:
        raise ValueError("Decoded LiDAR identity mismatch")
    ids = numeric.merge_voxel_id_batches(np.empty(0, dtype=np.uint64), pending)
    result = numeric.voxel_centers_local(ids, scene["voxel_size_m"])
    if len(result) < 1000:
        raise ValueError("Frozen LiDAR reference unexpectedly small")
    return result, {"raw_points": count, "strict_inside_points": inside_count,
                    "voxel_points": len(result), "decoded_records_sha256": decoded.hexdigest()}


def recompute_metrics(path, metrics, thresholds, epsilon):
    with np.load(path, allow_pickle=False) as f:
        a, c = f["reconstruction_to_reference_m"], f["reference_to_reconstruction_m"]
    if a.dtype != np.float64 or c.dtype != np.float64 or not a.size or not c.size or not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError("Invalid persisted bidirectional distances")
    values = {"reconstruction_points": a.size, "reference_points": c.size,
        "chamfer_l1_mean_m": (a.sum()/a.size + c.sum()/c.size)/2,
        "symmetric_rmse_m": np.sqrt((np.dot(a, a)/a.size + np.dot(c, c)/c.size)/2)}
    for prefix, distances in (("accuracy", a), ("completeness", c)):
        values.update({prefix+"_mean_m": distances.sum()/distances.size,
                       prefix+"_median_m": np.quantile(distances, .5),
                       prefix+"_p95_m": np.percentile(distances, 95)})
    for t in thresholds:
        label = str(int(round(t*100))) + "cm"
        p = np.count_nonzero(a <= t+epsilon)/a.size
        r = np.count_nonzero(c <= t+epsilon)/c.size
        values.update({"precision_"+label: p, "recall_"+label: r,
                       "fscore_"+label: 0 if p+r == 0 else 2*p*r/(p+r)})
    for key, value in values.items():
        if abs(float(value)-float(metrics[key])) > 1e-12:
            raise ValueError("Persisted-distance metric recomputation mismatch: " + key)
    return {"status": "PASS", "metrics": values, "distance_sha256": sha256(path)}


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    for path, expected in ((args.binding, args.binding_sha256), (args.surface_list, args.surface_list_sha256),
                           (args.sidecar, args.sidecar_sha256), (args.geometry, args.geometry_sha256),
                           (args.exports, args.exports_sha256)):
        verify_sha(path, expected)
    verify_source_snapshot(args.reference_root, args.reference_manifest_sha256)
    sys.path.insert(0, str(args.reference_root / "code/gcp"))
    numeric = _load_reference_module(args.reference_root, "evaluate_m3m_gcp_lidar_formal_v1")
    packet_reference = _load_reference_module(args.reference_root, "metric_depth_packet")
    protocol = _load_reference_module(args.reference_root, "m3m_native_quarter_protocol")
    binding, surface, sidecar, geometry = [read_json(p) for p in (args.binding, args.surface_list, args.sidecar, args.geometry)]
    for record in (binding, surface, geometry):
        if record_hash({k: v for k, v in record.items() if k != "records_root_sha256"}) != record["records_root_sha256"]:
            raise ValueError("Frozen binding record hash mismatch")
    source_views, _ = validate_source_sidecar(sidecar)
    scene = next(s for s in binding["scenes"] if s["scene"] == surface["scene"])
    if geometry["scene"] != surface["scene"] or sidecar["scene"] != surface["scene"] or geometry["transform"]["method_specific_fit"] or geometry["transform"]["lidar_used_in_fit"]:
        raise ValueError("Wrong shared source transform")
    if (scene["voxel_size_m"] != .05 or scene["boundary"] != "strict_inside_contains_xy"
            or scene["surface_list"]["sha256"] != args.surface_list_sha256
            or surface["sampling"] != {"alpha_min": .5, "array_start": [0, 0], "finite_positive_camera_z": True,
                                      "formula": "M1/A", "grid": "actual_native_no_resizing", "stride": 4,
                                      "tensor": "alpha_normalized_expected_camera_z"}
            or surface["distance_thresholds_m"] != [.05, .1, .2] or surface["distance_epsilon_m"] != 1e-9):
        raise ValueError("Unapproved surface protocol")
    selected = unique_rows(surface["views"], "image_name")
    if len(selected) != surface["counts"]["eval"] or not set(selected) <= set(source_views):
        raise ValueError("Surface view population mismatch")
    for name, item in selected.items():
        original = source_views[name]
        for key in ("image_id", "camera_id", "native_file", "native_sha256"):
            if item[key] != original[key]:
                raise ValueError("Surface/source view identity mismatch")
    transform = protocol.sim3_from_mapping(geometry)
    roi = box(*scene["roi_bounds_xy_m"])
    origin = np.asarray(scene["origin_m"], dtype=np.float64)
    np.testing.assert_array_equal(origin, numeric.freeze_local_origin(roi, .05))
    metadata_path = args.lidar_root / Path(scene["metadata"]["path"].replace("\\", "/")).name
    verify_sha(metadata_path, scene["metadata"]["sha256"])
    reference, reference_info = reference_surface(args.lidar_root / scene["actual_roi_laz"]["file"],
        read_json(metadata_path), scene, numeric, roi, origin)
    exports = read_json(args.exports)
    if len({e["method"] for e in exports}) != len(exports):
        raise ValueError("Duplicate method output")
    if any(Path(e["root"]).resolve() in args.output.resolve().parents for e in exports):
        raise ValueError("Cannot write into an export")
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "reference_voxel_centres.npz", points=reference, origin=origin)
    write_json(args.output / "reference_report.json", reference_info)
    transformer = Transformer.from_crs(4545, 32649, always_xy=True)
    reports = []
    for item in exports:
        root = Path(item["root"])
        verify_sha(root / "export_manifest.json", item["manifest_sha256"])
        manifest = read_json(root / "export_manifest.json")
        if (manifest["status"], manifest["packet_units"], manifest["packet_schema"], manifest["formula"], manifest["semantics"], manifest["primary_tensor"]) != (
                "EXPORTED_PENDING_SCIENTIFIC_SCORING", "source_model_not_survey_metres", "ms_gcp_metric_depth_packet_v2",
                "M1/A", "camera_z", "alpha_normalized_expected_camera_z"):
            raise ValueError("Unexpected depth export contract")
        if manifest["method"] != item["method"] or manifest["adapter_manifest_sha256"] != sidecar["adapter_manifest_sha256"]:
            raise ValueError("Method/adapter mismatch")
        verify_sha(root / "camera_records.json", manifest["camera_records_sha256"])
        cameras = unique_rows(read_json(root / "camera_records.json"), "image_name")
        packets = unique_rows(manifest["packet_views"], "image_name")
        if set(cameras) != set(packets) or set(cameras) != set(source_views):
            raise ValueError("Incomplete camera/packet population")
        pending, view_rows = [], []
        for name, frozen in selected.items():
            camera, entry = cameras[name], packets[name]
            if record_hash({k: v for k, v in camera.items() if k != "record_sha256"}) != camera["record_sha256"] or entry["camera_record_sha256"] != camera["record_sha256"]:
                raise ValueError("Camera identity/hash mismatch")
            if not camera["float32_camera_construction_exact"] or any(camera[k] != frozen[k] for k in ("image_id", "camera_id", "native_sha256")):
                raise ValueError("Runtime camera does not match selected surface view")
            original = np.asarray(camera["source_c2w_opengl_before_float32"]) @ np.diag([1., -1., -1., 1.])
            np.testing.assert_allclose(original, source_views[name]["source_c2w_opencv"], rtol=0, atol=1e-10)
            mapping = source_camera_mapping(camera)
            packet_path = safe_member(root, entry["packet"])
            verify_sha(packet_path, entry["sha256"])
            with np.load(packet_path, allow_pickle=False) as f:
                packet = {key: f[key] for key in f.files}
            if packet_reference.recompute_and_compare_packet(packet)["passed"] is not True:
                raise ValueError("Packet/reference inconsistency")
            source, row = native_surface_samples(packet, camera, mapping)
            target = transform.apply(source)
            e, n = transformer.transform(target[:, 0], target[:, 1])
            inside = contains_xy(roi, e, n)
            points = np.c_[e[inside], n[inside], target[inside, 2]]
            pending.append(numeric.voxel_batch_ids(points, .05, origin))
            view_rows.append({"image_name": name, **row, "roi_samples": int(inside.sum()),
                "packet_sha256": entry["sha256"], "camera_record_sha256": camera["record_sha256"]})
        ids = numeric.merge_voxel_id_batches(np.empty(0, dtype=np.uint64), pending)
        reconstruction = numeric.voxel_centers_local(ids, .05)
        out = args.output / item["method"]
        out.mkdir()
        write_csv(out / "surface_views.csv", view_rows)
        np.savez_compressed(out / "surface_voxel_centres.npz", points=reconstruction, origin=origin)
        if not len(reconstruction):
            report = {"method": item["method"], "status": "INCOMPLETE_UNRANKED", "reason": "no_surface_support_in_frozen_roi", "metrics": None}
        else:
            metrics, a, c = numeric.summarize_distances(reconstruction, reference, [.05, .1, .2], 100000, 1e-9)
            distance_path = out / "nearest_neighbor_distances.npz"
            np.savez_compressed(distance_path, reconstruction_to_reference_m=a, reference_to_reconstruction_m=c)
            independent = recompute_metrics(distance_path, metrics, [.05, .1, .2], 1e-9)
            write_json(out / "independent_recomputation.json", independent)
            report = {"method": item["method"], "status": "COMPLETE_RANKED", "metrics": metrics}
        report.update(schema="umgs_common_transform_lidar_result_v1", scene=surface["scene"], stage_review="PENDING",
                      whole_matrix_row_complete=False, view_count=len(view_rows), gpu_used=False,
                      common_transform=geometry["transform"], method_specific_transform_fitted=False,
                      reference_report=reference_info, surface_protocol=surface["sampling"],
                      roi_bounds_xy_m=scene["roi_bounds_xy_m"], local_origin_m=origin,
                      lidar_normal_height_bridge_applied_once_m=scene["normal_height_bridge_added_once_m"],
                      inputs={"export_manifest_sha256": item["manifest_sha256"], "binding_sha256": args.binding_sha256,
                              "geometry_sha256": args.geometry_sha256, "surface_list_sha256": args.surface_list_sha256,
                              "sidecar_sha256": args.sidecar_sha256, "reference_manifest_sha256": args.reference_manifest_sha256},
                      script_sha256=sha256(__file__))
        write_json(out / "summary.json", report)
        reports.append(report)
        print(json.dumps({"method": item["method"], "status": report["status"], "metrics": report["metrics"]}), flush=True)
    write_json(args.output / "execution_summary.json", {"scene": surface["scene"], "methods": [r["method"] for r in reports],
               "stage_review": "PENDING", "gpu_used": False, "wall_seconds": time.monotonic()-started})
    write_json(args.output / "output_hashes.json", [{"path": p.relative_to(args.output).as_posix(),
               "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(args.output.rglob("*")) if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binding", "surface-list", "sidecar", "geometry", "exports"):
        parser.add_argument("--"+name, required=True, type=Path)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--lidar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
