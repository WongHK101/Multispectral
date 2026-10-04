"""CPU source-camera geometry binding, shared by methods before scoring.

No method depth, LiDAR or method residual is an input. Callers authenticate the
release, camera sidecar and reference implementation before invoking this code.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

from .camera_bridge import Pinhole, finite_array, validate_pose
from .contracts import record_hash


def unique_rows(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if not key or key in result:
            raise ValueError(f"Missing or duplicate {field}: {key}")
        result[key] = row
    return result


def validate_source_sidecar(sidecar):
    if sidecar["schema"] != "umgs_old_camera_raw_click_sidecar_v1":
        raise ValueError("Unsupported source-camera sidecar")
    expected = record_hash({"views": sidecar["views"], "observations": sidecar["observations"]})
    if sidecar["records_root_sha256"] != expected:
        raise ValueError("Camera sidecar records root mismatch")
    views = unique_rows(sidecar["views"], "image_name")
    observations = unique_rows(sidecar["observations"], "observation_id")
    for view in views.values():
        if record_hash({k: v for k, v in view.items() if k != "mapping_sha256"}) != view["mapping_sha256"]:
            raise ValueError("Source view hash mismatch")
        a = validate_pose(view["source_w2c_opencv"], rigidity_atol=1e-10)
        b = validate_pose(view["source_c2w_opencv"], rigidity_atol=1e-10)
        if not np.allclose(a @ b, np.eye(4), rtol=0, atol=1e-10):
            raise ValueError("Source pose inverse mismatch")
    for row in observations.values():
        if type(row["formal_eligible"]) is not bool:
            raise ValueError("Formal eligibility must be boolean")
        view = views[row["image_name"]]
        if row["mapping_sha256"] != view["mapping_sha256"]:
            raise ValueError("Observation mapping mismatch")
        ray = finite_array(row["old_camera_ray_xy"], (2,), "raw source ray")
        pixel = finite_array(row["native_array_xy"], (2,), "native coordinate")
        k = Pinhole(**view["native_camera_array"])
        if np.max(np.abs(k.unproject([pixel])[0] - ray)) > 1e-12:
            raise ValueError("Native/source ray mismatch")
        bounds = bool(np.all(pixel >= 0) and np.all(pixel < [k.width, k.height]))
        if bounds != row["in_bounds"] or (row["formal_eligible"] and not bounds):
            raise ValueError("Formal bounds or cached bounds mismatch")
    return views, observations


def common_control_geometry(sidecar, roles, control_targets, *, triangulator, fitter, colmap):
    """Use authenticated DLT/Umeyama reference functions without changing them.

    Unit PINHOLE inputs carry the already distortion-inverted source rays. This
    avoids running a second distortion solver or inheriting the new SfM rays.
    Checkpoint survey coordinates cannot enter this API's fit.
    """
    views, rows = validate_source_sidecar(sidecar)
    if not roles or set(roles.values()) != {"control", "checkpoint"}:
        raise ValueError("Require frozen control and checkpoint roles")
    controls = sorted(k for k, v in roles.items() if v == "control")
    checkpoints = sorted(k for k, v in roles.items() if v == "checkpoint")
    if len(controls) < 4 or set(control_targets) != set(controls):
        raise ValueError("Require all and only frozen control targets")
    targets = np.vstack([finite_array(control_targets[n], (3,), "survey control") for n in controls])
    cameras, images = {}, {}
    for name, view in views.items():
        p = np.asarray(view["source_w2c_opencv"], dtype=np.float64)
        q = colmap.rotmat2qvec(p[:3, :3])
        if np.max(np.abs(colmap.qvec2rotmat(q) - p[:3, :3])) > 1e-12:
            raise ValueError("Pose quaternion conversion mismatch")
        cid = view["camera_id"]
        cameras[cid] = SimpleNamespace(model="PINHOLE", params=[1., 1., 0., 0.])
        images[name] = SimpleNamespace(name=name, camera_id=cid, qvec=q, tvec=p[:3, 3])
    grouped = defaultdict(list)
    formal = [r for r in rows.values() if r["formal_eligible"]]
    for row in formal:
        if (row["point_name"] not in roles or row["formal_role"] != roles[row["point_name"]]
                or row["annotation_quality"] != "good"):
            raise ValueError("Formal row disagrees with frozen role/quality")
        grouped[row["point_name"]].append(row)
    if set(grouped) != set(roles):
        raise ValueError("Frozen point population incomplete")
    points, source_rows = {}, []
    for name in sorted(grouped):
        obs = sorted(grouped[name], key=lambda r: r["image_name"])
        if len(obs) < 2 or len({r["image_name"] for r in obs}) != len(obs):
            raise ValueError("Need unique multi-view observations")
        normalized = [{"image_name": r["image_name"], "u_px": r["old_camera_ray_xy"][0],
                       "v_px": r["old_camera_ray_xy"][1]} for r in obs]
        xyz = finite_array(triangulator(normalized, cameras, images), (3,), "triangulated point")
        for row in obs:
            p = np.asarray(views[row["image_name"]]["source_w2c_opencv"])
            if (p[:3, :3] @ xyz + p[:3, 3])[2] <= 1e-9:
                raise ValueError("Triangulated point behind source camera")
        points[name] = xyz
        source_rows.append({"point_name": name, "role": roles[name], "source_model_xyz": xyz.tolist(),
                            "observation_ids": [r["observation_id"] for r in obs]})
    source = np.vstack([points[n] for n in controls])
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("Collinear source controls")
    scale, rotation, translation = fitter(source, targets, estimate_scale=True)
    rotation = finite_array(rotation, (3, 3), "common rotation")
    translation = finite_array(translation, (3,), "common translation")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Nonpositive/nonfinite common scale")
    transform = {"scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist(),
                 "definition": "survey_xyz = scale * rotation @ source_model_xyz + translation",
                 "survey_fields": ["cgcs2000_gk_cm108_e_m", "cgcs2000_gk_cm108_n_m", "cgcs2000_normal_height_m"],
                 "fit_controls": controls, "excluded_checkpoints": checkpoints,
                 "method_specific_fit": False, "lidar_used_in_fit": False}
    residual = (scale * (rotation @ source.T)).T + translation - targets
    geometry_rows = []
    for row in formal:
        view = views[row["image_name"]]
        c2w = np.asarray(view["source_c2w_opencv"])
        centre = c2w[:3, 3]
        axis = rotation @ c2w[:3, 2]
        axis /= np.linalg.norm(axis)
        off = math.degrees(math.acos(float(np.clip(axis @ [0., 0., -1.], -1., 1.))))
        delta = centre - points[row["point_name"]]
        azimuth = (math.degrees(math.atan2(float(delta[0]), float(delta[1]))) + 360.) % 360.
        geometry_rows.append({"observation_id": row["observation_id"], "point_name": row["point_name"],
            "image_name": row["image_name"], "role": row["formal_role"],
            "mapping_sha256": row["mapping_sha256"], "native_array_xy": row["native_array_xy"],
            "source_camera_ray_xy": row["old_camera_ray_xy"],
            "view_class": "nadir" if off <= 5. else "oblique", "off_nadir_deg": off,
            "camera_position_azimuth_deg": azimuth, "azimuth_bin_45deg": int(((azimuth + 22.5) % 360.) // 45.)})
    result = {"schema": "umgs_common_source_geometry_v1", "scene": sidecar["scene"],
        "status": "PASS_SOURCE_GEOMETRY_NOT_METHOD_METRICS", "transform": transform,
        "control_calibration_residuals": [{"point_name": n, "delta_xyz_m": r.tolist()}
                                           for n, r in zip(controls, residual)],
        "triangulated_source_points": source_rows, "observations": geometry_rows,
        "view_group_rule": {"nadir_max_deg": 5., "azimuth_frame": "source_model_XY",
                            "azimuth_formula": "atan2(camera_x-point_x,camera_y-point_y)",
                            "bin_width_deg": 45., "bin_offset_deg": 22.5},
        "formal_observation_count": len(formal), "control_count": len(controls),
        "checkpoint_count": len(checkpoints), "method_depth_read": False,
        "method_metrics_generated": False, "gpu_used": False}
    result["records_root_sha256"] = record_hash(result)
    return result


def heldout_surface_views(sidecar, adapter):
    """Bind actual UMGS eval captures; never borrow another project's holdout."""
    views, _ = validate_source_sidecar(sidecar)
    groups = unique_rows(adapter["groups"], "image_name")
    if set(groups) != set(views):
        raise ValueError("Adapter/sidecar image set mismatch")
    counts = Counter(g["split"] for g in groups.values())
    if set(counts) != {"train", "eval"}:
        raise ValueError("Unexpected or absent split")
    rows = []
    for name in sorted(groups):
        g, v = groups[name], views[name]
        rgb = g["output_images"]["D"]
        if (rgb["sha256"] != v["native_sha256"] or rgb["relative_path"] != v["native_file"]
                or [rgb["width"], rgb["height"]] != [v["native_camera_array"][k] for k in ("width", "height")]
                or g["image_id"] != v["image_id"] or g["camera_id"] != v["camera_id"]):
            raise ValueError("Surface image/camera identity mismatch")
        if g["split"] == "eval":
            rows.append({"image_name": name, "image_id": v["image_id"], "camera_id": v["camera_id"],
                         "native_file": v["native_file"], "native_sha256": v["native_sha256"],
                         "native_camera_array": v["native_camera_array"], "mapping_sha256": v["mapping_sha256"]})
    result = {"schema": "umgs_heldout_surface_view_list_v1", "scene": sidecar["scene"],
        "group": "umgs_existing_eval_captures", "counts": dict(counts), "views": rows,
        "sampling": {"tensor": "alpha_normalized_expected_camera_z", "formula": "M1/A",
                     "alpha_min": .5, "finite_positive_camera_z": True,
                     "stride": 4, "array_start": [0, 0], "grid": "actual_native_no_resizing"},
        "voxel_size_m": .05, "distance_thresholds_m": [.05, .10, .20], "distance_epsilon_m": 1e-9,
        "other_project_view_lists_used": False}
    result["records_root_sha256"] = record_hash(result)
    return result
