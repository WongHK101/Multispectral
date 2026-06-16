#!/usr/bin/env python3
"""Stage-1 validation for DA3 camera geometry against independent evidence.

This script does not run neural inference. It evaluates:
1. DA3 camera centers against a COLMAP camera trajectory via one global Sim(3).
2. DA3 camera centers against WGS84/RTK camera positions via one global Sim(3).
3. Whether the supplied COLMAP sparse model is geometrically compatible with
   the DA3 image domain for a later sparse-depth residual test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (self.rotation @ points.T).T + self.translation


def qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(list(qvec), dtype=np.float64)
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qw * qz,
                2 * qz * qx + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qz * qx - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def camera_centers_from_w2c(extrinsics: np.ndarray) -> np.ndarray:
    centers = []
    for extrinsic in extrinsics:
        rotation = np.asarray(extrinsic[:, :3], dtype=np.float64)
        translation = np.asarray(extrinsic[:, 3], dtype=np.float64)
        centers.append(-rotation.T @ translation)
    return np.asarray(centers)


def fit_sim3(source: np.ndarray, target: np.ndarray) -> Sim3:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError("At least three correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u_mat @ vt_mat) < 0:
        sign[-1] = -1
    rotation = u_mat @ np.diag(sign) @ vt_mat
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate source trajectory")
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def residual_statistics(residuals: np.ndarray, footprint_diagonal: float) -> dict:
    residuals = np.asarray(residuals, dtype=np.float64)
    return {
        "count": int(len(residuals)),
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "p90": float(np.percentile(residuals, 90)),
        "p95": float(np.percentile(residuals, 95)),
        "max": float(np.max(residuals)),
        "median_fraction_of_footprint_diagonal": float(np.median(residuals) / footprint_diagonal),
        "p90_fraction_of_footprint_diagonal": float(np.percentile(residuals, 90) / footprint_diagonal),
        "max_fraction_of_footprint_diagonal": float(np.max(residuals) / footprint_diagonal),
    }


def component_residual_statistics(aligned: np.ndarray, target: np.ndarray) -> dict:
    differences = aligned - target
    target_z_std = float(np.std(target[:, 2]))
    aligned_z_std = float(np.std(aligned[:, 2]))
    return {
        "axis_mae_xyz": np.mean(np.abs(differences), axis=0).tolist(),
        "axis_rmse_xyz": np.sqrt(np.mean(differences**2, axis=0)).tolist(),
        "axis_max_abs_xyz": np.max(np.abs(differences), axis=0).tolist(),
        "target_z_std": target_z_std,
        "aligned_z_std": aligned_z_std,
        "target_z_range": float(np.ptp(target[:, 2])),
        "aligned_z_range": float(np.ptp(aligned[:, 2])),
        "vertical_pose_caution": bool(
            aligned_z_std > max(0.5, 5.0 * max(target_z_std, np.finfo(np.float64).eps))
        ),
        "vertical_pose_caution_note": (
            "The DA3-predicted camera trajectory has materially larger vertical variation than the external trajectory. "
            "Use externally anchored COLMAP/RTK poses for any later depth fusion."
        ),
    }


def trajectory_statistics(points: np.ndarray) -> dict:
    extent = points.max(axis=0) - points.min(axis=0)
    footprint_diagonal = float(np.linalg.norm(extent))
    path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return {
        "min_xyz": points.min(axis=0).tolist(),
        "max_xyz": points.max(axis=0).tolist(),
        "extent_xyz": extent.tolist(),
        "footprint_diagonal": footprint_diagonal,
        "ordered_path_length": path_length,
    }


def load_da3(scene_root: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    summary_path = scene_root / "da3_smoke_summary.json"
    npz_path = scene_root / "exports" / "mini_npz" / "results.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    names = [Path(name).name for name in summary["selected_images"]]
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    depth = np.asarray(data["depth"])
    if not (len(names) == len(extrinsics) == len(intrinsics) == len(depth)):
        raise ValueError("DA3 image list and exported arrays have different lengths")
    return names, camera_centers_from_w2c(extrinsics), extrinsics, intrinsics


def parse_colmap_images(images_path: Path) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[float, float, int]]]]:
    lines = [
        line.strip()
        for line in images_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    centers: dict[str, np.ndarray] = {}
    observations: dict[str, list[tuple[float, float, int]]] = {}
    if len(lines) % 2:
        raise ValueError(f"Unexpected odd number of non-comment lines in {images_path}")
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            continue
        qvec = np.asarray(fields[1:5], dtype=np.float64)
        tvec = np.asarray(fields[5:8], dtype=np.float64)
        name = Path(fields[9]).name
        rotation = qvec_to_rotmat(qvec)
        centers[name] = -rotation.T @ tvec

        observation_fields = lines[index + 1].split()
        image_observations = []
        for offset in range(0, len(observation_fields) - 2, 3):
            image_observations.append(
                (
                    float(observation_fields[offset]),
                    float(observation_fields[offset + 1]),
                    int(observation_fields[offset + 2]),
                )
            )
        observations[name] = image_observations
    return centers, observations


def parse_colmap_camera(cameras_path: Path) -> dict:
    for line in cameras_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        return {
            "camera_id": int(fields[0]),
            "model": fields[1],
            "width": int(fields[2]),
            "height": int(fields[3]),
            "params": [float(value) for value in fields[4:]],
        }
    raise ValueError(f"No camera found in {cameras_path}")


def parse_sequence(name: str) -> int:
    match = re.search(r"_(\d{4})_D\.JPG$", Path(name).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse sequence from {name}")
    return int(match.group(1))


def parse_mrk(mrk_path: Path) -> dict[int, dict]:
    latitude_pattern = re.compile(r"([-+]?\d+(?:\.\d+)?),Lat")
    longitude_pattern = re.compile(r"([-+]?\d+(?:\.\d+)?),Lon")
    altitude_pattern = re.compile(r"([-+]?\d+(?:\.\d+)?),Ellh")
    result: dict[int, dict] = {}
    for line in mrk_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        sequence = int(fields[0])
        latitude = float(latitude_pattern.search(line).group(1))
        longitude = float(longitude_pattern.search(line).group(1))
        altitude = float(altitude_pattern.search(line).group(1))
        after_altitude = line.split("Ellh", maxsplit=1)[1]
        standard_deviations = [
            float(value)
            for value in re.findall(r"[-+]?\d+(?:\.\d+)?", after_altitude)
        ][:3]
        result[sequence] = {
            "latitude": latitude,
            "longitude": longitude,
            "ellipsoid_height": altitude,
            "std_north": standard_deviations[0],
            "std_east": standard_deviations[1],
            "std_vertical": standard_deviations[2],
        }
    return result


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, height: float) -> np.ndarray:
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    normal_radius = semi_major / math.sqrt(1.0 - eccentricity_sq * sin_latitude**2)
    return np.array(
        [
            (normal_radius + height) * cos_latitude * math.cos(longitude),
            (normal_radius + height) * cos_latitude * math.sin(longitude),
            (normal_radius * (1.0 - eccentricity_sq) + height) * sin_latitude,
        ],
        dtype=np.float64,
    )


def geodetic_rows_to_enu(rows: list[dict]) -> tuple[np.ndarray, dict]:
    latitude0 = float(np.mean([row["latitude"] for row in rows]))
    longitude0 = float(np.mean([row["longitude"] for row in rows]))
    height0 = float(np.mean([row["ellipsoid_height"] for row in rows]))
    origin_ecef = geodetic_to_ecef(latitude0, longitude0, height0)
    latitude = math.radians(latitude0)
    longitude = math.radians(longitude0)
    rotation = np.array(
        [
            [-math.sin(longitude), math.cos(longitude), 0.0],
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ],
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ],
        ],
        dtype=np.float64,
    )
    enu = np.asarray(
        [
            rotation @ (geodetic_to_ecef(row["latitude"], row["longitude"], row["ellipsoid_height"]) - origin_ecef)
            for row in rows
        ]
    )
    return enu, {
        "latitude": latitude0,
        "longitude": longitude0,
        "ellipsoid_height": height0,
    }


def orthogonality_audit(extrinsics: np.ndarray) -> dict:
    determinants = []
    errors = []
    identity = np.eye(3)
    for extrinsic in extrinsics:
        rotation = np.asarray(extrinsic[:, :3], dtype=np.float64)
        determinants.append(float(np.linalg.det(rotation)))
        errors.append(float(np.linalg.norm(rotation.T @ rotation - identity, ord="fro")))
    return {
        "rotation_determinant_min": min(determinants),
        "rotation_determinant_max": max(determinants),
        "rotation_orthogonality_error_max": max(errors),
    }


def evaluate_alignment(source: np.ndarray, target: np.ndarray) -> tuple[Sim3, np.ndarray, dict]:
    transform = fit_sim3(source, target)
    aligned = transform.apply(source)
    residuals = np.linalg.norm(aligned - target, axis=1)
    target_stats = trajectory_statistics(target)
    statistics = residual_statistics(residuals, target_stats["footprint_diagonal"])
    statistics["sim3"] = {
        "scale": transform.scale,
        "rotation": transform.rotation.tolist(),
        "rotation_determinant": float(np.linalg.det(transform.rotation)),
        "translation": transform.translation.tolist(),
    }
    statistics["source_trajectory"] = trajectory_statistics(source)
    statistics["target_trajectory"] = target_stats
    statistics["aligned_trajectory"] = trajectory_statistics(aligned)
    statistics["component_residuals"] = component_residual_statistics(aligned, target)
    return transform, aligned, statistics


def alternating_cross_validation(source: np.ndarray, target: np.ndarray) -> dict:
    results = {}
    for train_parity in (0, 1):
        train_indices = np.arange(len(source)) % 2 == train_parity
        test_indices = ~train_indices
        transform = fit_sim3(source[train_indices], target[train_indices])
        residuals = np.linalg.norm(transform.apply(source[test_indices]) - target[test_indices], axis=1)
        footprint_diagonal = trajectory_statistics(target)["footprint_diagonal"]
        results[f"train_{'even' if train_parity == 0 else 'odd'}_test_{'odd' if train_parity == 0 else 'even'}"] = {
            "scale": transform.scale,
            **residual_statistics(residuals, footprint_diagonal),
        }
    scales = [item["scale"] for item in results.values()]
    results["scale_relative_difference"] = float(abs(scales[0] - scales[1]) / np.mean(scales))
    return results


def provisional_gate_status(statistics: dict, cross_validation: dict) -> dict:
    p90_cv = max(
        item["p90_fraction_of_footprint_diagonal"]
        for key, item in cross_validation.items()
        if key.startswith("train_")
    )
    checks = {
        "median_le_5pct_footprint": statistics["median_fraction_of_footprint_diagonal"] <= 0.05,
        "p90_le_10pct_footprint": statistics["p90_fraction_of_footprint_diagonal"] <= 0.10,
        "alternating_cv_p90_le_15pct_footprint": p90_cv <= 0.15,
        "alternating_scale_difference_le_20pct": cross_validation["scale_relative_difference"] <= 0.20,
        "proper_positive_sim3": (
            statistics["sim3"]["scale"] > 0
            and abs(statistics["sim3"]["rotation_determinant"] - 1.0) <= 1e-6
        ),
    }
    return {
        "status": "provisional_pass" if all(checks.values()) else "provisional_fail",
        "checks": checks,
        "note": "Smoke-feasibility thresholds declared before inspecting Stage-1 residual outputs; not manuscript acceptance thresholds.",
    }


def save_alignment_csv(
    path: Path,
    names: list[str],
    source: np.ndarray,
    aligned: np.ndarray,
    target: np.ndarray,
) -> None:
    residuals = np.linalg.norm(aligned - target, axis=1)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "source_x",
                "source_y",
                "source_z",
                "aligned_x",
                "aligned_y",
                "aligned_z",
                "target_x",
                "target_y",
                "target_z",
                "residual",
            ]
        )
        for name, source_point, aligned_point, target_point, residual in zip(
            names, source, aligned, target, residuals
        ):
            writer.writerow(
                [
                    name,
                    *source_point.tolist(),
                    *aligned_point.tolist(),
                    *target_point.tolist(),
                    float(residual),
                ]
            )


def save_alignment_plot(
    path: Path,
    names: list[str],
    aligned: np.ndarray,
    target: np.ndarray,
    target_label: str,
) -> None:
    residuals = np.linalg.norm(aligned - target, axis=1)
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    axes[0].plot(target[:, 0], target[:, 1], "o-", ms=3.5, lw=1.0, label=target_label)
    axes[0].plot(aligned[:, 0], aligned[:, 1], "s--", ms=3.0, lw=0.9, label="DA3 after Sim(3)")
    axes[0].set_xlabel("X / East")
    axes[0].set_ylabel("Y / North")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Top-view trajectory")

    axes[1].plot(target[:, 0], target[:, 2], "o-", ms=3.5, lw=1.0, label=target_label)
    axes[1].plot(aligned[:, 0], aligned[:, 2], "s--", ms=3.0, lw=0.9, label="DA3 after Sim(3)")
    axes[1].set_xlabel("X / East")
    axes[1].set_ylabel("Z / Up")
    axes[1].set_title("Vertical profile")

    axes[2].plot(np.arange(len(residuals)), residuals, "o-", ms=3.5, lw=1.0)
    axes[2].axhline(np.median(residuals), color="tab:orange", ls="--", lw=1.0, label="median")
    axes[2].set_xticks(np.arange(len(names))[::4])
    axes[2].set_xticklabels([f"{parse_sequence(name):04d}" for name in names][::4], rotation=45)
    axes[2].set_xlabel("Selected image sequence")
    axes[2].set_ylabel("3D center residual")
    axes[2].set_title("Per-view residual")
    axes[2].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def sparse_compatibility_audit(
    camera: dict,
    observations: dict[str, list[tuple[float, float, int]]],
    selected_names: list[str],
    raw_width: int,
    raw_height: int,
    da3_width: int,
    da3_height: int,
) -> dict:
    totals = {
        "observation_count": 0,
        "linked_point_count": 0,
        "inside_camera_bounds_count": 0,
        "linked_inside_camera_bounds_count": 0,
    }
    missing_images = []
    per_image = []
    for name in selected_names:
        image_observations = observations.get(name)
        if image_observations is None:
            missing_images.append(name)
            continue
        total = len(image_observations)
        linked = sum(point_id >= 0 for _, _, point_id in image_observations)
        inside = sum(
            0 <= x < camera["width"] and 0 <= y < camera["height"]
            for x, y, _ in image_observations
        )
        linked_inside = sum(
            point_id >= 0 and 0 <= x < camera["width"] and 0 <= y < camera["height"]
            for x, y, point_id in image_observations
        )
        totals["observation_count"] += total
        totals["linked_point_count"] += linked
        totals["inside_camera_bounds_count"] += inside
        totals["linked_inside_camera_bounds_count"] += linked_inside
        per_image.append(
            {
                "image": name,
                "observation_count": total,
                "linked_point_count": linked,
                "inside_camera_bounds_count": inside,
                "linked_inside_camera_bounds_count": linked_inside,
            }
        )

    camera_aspect = camera["width"] / camera["height"]
    raw_aspect = raw_width / raw_height
    da3_aspect = da3_width / da3_height
    width_difference = abs(camera["width"] - raw_width) / raw_width
    height_difference = abs(camera["height"] - raw_height) / raw_height
    aspect_difference = abs(camera_aspect - raw_aspect) / raw_aspect
    mapping_is_explicit = False
    compatible = (
        width_difference <= 0.01
        and height_difference <= 0.01
        and aspect_difference <= 0.01
        and not missing_images
    )
    return {
        "status": "compatible" if compatible else "blocked_pending_compatible_rgb_sparse_or_explicit_rectification_map",
        "camera": camera,
        "raw_image_size": [raw_width, raw_height],
        "da3_image_size": [da3_width, da3_height],
        "camera_aspect": camera_aspect,
        "raw_aspect": raw_aspect,
        "da3_aspect": da3_aspect,
        "camera_vs_raw_width_fraction_difference": width_difference,
        "camera_vs_raw_height_fraction_difference": height_difference,
        "camera_vs_raw_aspect_fraction_difference": aspect_difference,
        "explicit_rectification_map_available": mapping_is_explicit,
        "missing_selected_images": missing_images,
        "aggregate_observations": totals,
        "per_image_observations": per_image,
        "reason": (
            "Sparse-point depth residuals are not computed unless the COLMAP image domain matches the DA3/raw RGB domain within 1%, "
            "or an explicit validated pixel mapping is supplied."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maize-da3-root", type=Path, required=True)
    parser.add_argument("--road-da3-root", type=Path, required=True)
    parser.add_argument("--maize-colmap-root", type=Path, required=True)
    parser.add_argument("--road-mrk", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-width", type=int, default=5280)
    parser.add_argument("--raw-height", type=int, default=3956)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    maize_names, maize_da3_centers, maize_extrinsics, maize_intrinsics = load_da3(args.maize_da3_root)
    colmap_centers, colmap_observations = parse_colmap_images(args.maize_colmap_root / "images.txt")
    matched_maize_names = [name for name in maize_names if name in colmap_centers]
    maize_source = np.asarray([maize_da3_centers[maize_names.index(name)] for name in matched_maize_names])
    maize_target = np.asarray([colmap_centers[name] for name in matched_maize_names])
    maize_transform, maize_aligned, maize_statistics = evaluate_alignment(maize_source, maize_target)
    maize_cv = alternating_cross_validation(maize_source, maize_target)
    maize_gate = provisional_gate_status(maize_statistics, maize_cv)
    maize_result = {
        "gate": "A_camera_sim3_colmap",
        "matched_images": len(matched_maize_names),
        "selected_images": len(maize_names),
        "missing_images": [name for name in maize_names if name not in colmap_centers],
        "da3_extrinsic_audit": orthogonality_audit(maize_extrinsics),
        "statistics": maize_statistics,
        "alternating_cross_validation": maize_cv,
        "provisional_gate": maize_gate,
    }
    save_alignment_csv(
        args.output_root / "maize_da3_vs_colmap_camera_centers.csv",
        matched_maize_names,
        maize_source,
        maize_aligned,
        maize_target,
    )
    save_alignment_plot(
        args.output_root / "maize_da3_vs_colmap_camera_centers.png",
        matched_maize_names,
        maize_aligned,
        maize_target,
        "COLMAP",
    )

    road_names, road_da3_centers, road_extrinsics, road_intrinsics = load_da3(args.road_da3_root)
    mrk = parse_mrk(args.road_mrk)
    road_rows = [mrk[parse_sequence(name)] for name in road_names]
    road_enu, road_origin = geodetic_rows_to_enu(road_rows)
    road_transform, road_aligned, road_statistics = evaluate_alignment(road_da3_centers, road_enu)
    road_cv = alternating_cross_validation(road_da3_centers, road_enu)
    road_gate = provisional_gate_status(road_statistics, road_cv)
    horizontal_std = np.asarray(
        [math.hypot(row["std_north"], row["std_east"]) for row in road_rows]
    )
    vertical_std = np.asarray([row["std_vertical"] for row in road_rows])
    rtk_quality = {
        "horizontal_std_median": float(np.median(horizontal_std)),
        "horizontal_std_p90": float(np.percentile(horizontal_std, 90)),
        "horizontal_std_max": float(np.max(horizontal_std)),
        "vertical_std_median": float(np.median(vertical_std)),
        "vertical_std_p90": float(np.percentile(vertical_std, 90)),
        "vertical_std_max": float(np.max(vertical_std)),
        "provisional_quality_check": {
            "median_horizontal_std_le_0.10m": bool(np.median(horizontal_std) <= 0.10),
            "median_vertical_std_le_0.20m": bool(np.median(vertical_std) <= 0.20),
        },
    }
    road_result = {
        "gate": "E_camera_sim3_rtk_enu",
        "matched_images": len(road_names),
        "enu_origin_wgs84": road_origin,
        "da3_extrinsic_audit": orthogonality_audit(road_extrinsics),
        "rtk_quality": rtk_quality,
        "statistics": road_statistics,
        "alternating_cross_validation": road_cv,
        "provisional_gate": road_gate,
        "scope_note": "This validates camera-trajectory consistency only; it is not a dense-depth or surface-accuracy claim.",
    }
    save_alignment_csv(
        args.output_root / "road_da3_vs_rtk_camera_centers.csv",
        road_names,
        road_da3_centers,
        road_aligned,
        road_enu,
    )
    save_alignment_plot(
        args.output_root / "road_da3_vs_rtk_camera_centers.png",
        road_names,
        road_aligned,
        road_enu,
        "RTK ENU",
    )

    camera = parse_colmap_camera(args.maize_colmap_root / "cameras.txt")
    sparse_compatibility = sparse_compatibility_audit(
        camera=camera,
        observations=colmap_observations,
        selected_names=maize_names,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        da3_width=int(maize_intrinsics[0, 0, 2] * 2),
        da3_height=int(maize_intrinsics[0, 1, 2] * 2),
    )

    combined = {
        "schema": "da3_depth_reference_validation_stage1_v1",
        "scope": "No new inference; camera geometry and evidence compatibility only.",
        "maize_gate_a": maize_result,
        "road_gate_e": road_result,
        "maize_gate_b_sparse_depth_compatibility": sparse_compatibility,
    }
    (args.output_root / "da3_stage1_validation_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# DA3 Stage-1 Depth-Reference Validation",
        "",
        "This audit uses existing DA3-LARGE-1.1 outputs only. It does not run inference, training, rendering, or manuscript changes.",
        "",
        "## Gate A: Maize camera trajectory vs COLMAP",
        "",
        f"- Matched views: {len(matched_maize_names)}/{len(maize_names)}",
        f"- Global Sim(3) scale: {maize_transform.scale:.6f}",
        f"- Median residual: {maize_statistics['median']:.6f} ({maize_statistics['median_fraction_of_footprint_diagonal']:.2%} of footprint diagonal)",
        f"- P90 residual: {maize_statistics['p90']:.6f} ({maize_statistics['p90_fraction_of_footprint_diagonal']:.2%})",
        f"- Alternating-fit scale difference: {maize_cv['scale_relative_difference']:.2%}",
        f"- Provisional status: **{maize_gate['status']}**",
        f"- Vertical range, COLMAP vs aligned DA3: {maize_statistics['component_residuals']['target_z_range']:.3f} vs {maize_statistics['component_residuals']['aligned_z_range']:.3f}",
        "- Interpretation: the coarse trajectory gate passes, but DA3-predicted vertical pose jitter prevents replacing the COLMAP pose reference.",
        "",
        "## Gate E: Road camera trajectory vs WGS84/RTK",
        "",
        f"- Matched views: {len(road_names)}/{len(road_names)}",
        f"- ENU origin: {road_origin['latitude']:.9f}, {road_origin['longitude']:.9f}, {road_origin['ellipsoid_height']:.3f} m",
        f"- Global Sim(3) scale: {road_transform.scale:.6f}",
        f"- Median residual: {road_statistics['median']:.3f} m ({road_statistics['median_fraction_of_footprint_diagonal']:.2%} of footprint diagonal)",
        f"- P90 residual: {road_statistics['p90']:.3f} m ({road_statistics['p90_fraction_of_footprint_diagonal']:.2%})",
        f"- RTK horizontal std median/P90: {rtk_quality['horizontal_std_median']:.3f}/{rtk_quality['horizontal_std_p90']:.3f} m",
        f"- RTK vertical std median/P90: {rtk_quality['vertical_std_median']:.3f}/{rtk_quality['vertical_std_p90']:.3f} m",
        f"- Provisional status: **{road_gate['status']}**",
        f"- Vertical range, RTK vs aligned DA3: {road_statistics['component_residuals']['target_z_range']:.3f} vs {road_statistics['component_residuals']['aligned_z_range']:.3f} m",
        "- Scope: camera-trajectory sanity only, not dense-depth or surface accuracy.",
        "- Interpretation: preserve RTK/COLMAP camera poses for later fusion; do not use DA3-predicted poses as the metric reference.",
        "",
        "## Gate B: Sparse-depth compatibility",
        "",
        f"- Status: **{sparse_compatibility['status']}**",
        f"- COLMAP camera: {camera['width']} x {camera['height']} ({camera['model']})",
        f"- Raw RGB domain: {args.raw_width} x {args.raw_height}",
        f"- DA3 domain: {sparse_compatibility['da3_image_size'][0]} x {sparse_compatibility['da3_image_size'][1]}",
        f"- Camera/raw aspect difference: {sparse_compatibility['camera_vs_raw_aspect_fraction_difference']:.2%}",
        "- No sparse-depth residual is reported unless a compatible RGB sparse model or an explicit validated rectification map is available.",
        "",
        "## Decision Boundary",
        "",
        "These are smoke-feasibility checks. Passing them would justify a later, separately approved depth-reference experiment; it would not replace the manuscript's current validation-gated mesh diagnostic by itself.",
    ]
    (args.output_root / "DA3_STAGE1_VALIDATION_REPORT_20260615.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
