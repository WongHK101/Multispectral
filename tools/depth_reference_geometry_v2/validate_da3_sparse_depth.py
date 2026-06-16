#!/usr/bin/env python3
"""Compare existing DA3 depths with matched COLMAP sparse point depths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validate_da3_stage1 import (
    camera_centers_from_w2c,
    fit_sim3,
    load_da3,
    parse_colmap_camera,
    qvec_to_rotmat,
)


def parse_images_with_poses(path: Path) -> dict[str, dict]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) % 2:
        raise ValueError(f"Unexpected images.txt structure: {path}")
    images = {}
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            continue
        name = Path(fields[9]).name
        rotation = qvec_to_rotmat([float(value) for value in fields[1:5]])
        translation = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        observation_fields = lines[index + 1].split()
        observations = []
        for offset in range(0, len(observation_fields) - 2, 3):
            observations.append(
                (
                    float(observation_fields[offset]),
                    float(observation_fields[offset + 1]),
                    int(observation_fields[offset + 2]),
                )
            )
        images[name] = {
            "rotation": rotation,
            "translation": translation,
            "center": -rotation.T @ translation,
            "observations": observations,
        }
    return images


def parse_points3d(path: Path) -> dict[int, dict]:
    points = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        point_id = int(fields[0])
        points[point_id] = {
            "xyz": np.asarray([float(value) for value in fields[1:4]], dtype=np.float64),
            "error": float(fields[7]),
            "track_length": (len(fields) - 8) // 2,
        }
    return points


def bilinear_sample(array: np.ndarray, x: float, y: float) -> float:
    height, width = array.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return float("nan")
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return float(
        (1 - wx) * (1 - wy) * array[y0, x0]
        + wx * (1 - wy) * array[y0, x1]
        + (1 - wx) * wy * array[y1, x0]
        + wx * wy * array[y1, x1]
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def compute_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    epsilon = 1e-8
    absolute_relative = np.abs(prediction - reference) / np.maximum(reference, epsilon)
    ratios = np.maximum(
        prediction / np.maximum(reference, epsilon),
        reference / np.maximum(prediction, epsilon),
    )
    log_residual = np.abs(
        np.log(np.maximum(prediction, epsilon)) - np.log(np.maximum(reference, epsilon))
    )
    return {
        "count": int(len(reference)),
        "reference_median": float(np.median(reference)),
        "prediction_median": float(np.median(prediction)),
        "prediction_reference_ratio_median": float(np.median(prediction / reference)),
        "absrel_mean": float(np.mean(absolute_relative)),
        "absrel_median": float(np.median(absolute_relative)),
        "absrel_p90": float(np.percentile(absolute_relative, 90)),
        "rmse": float(np.sqrt(np.mean((prediction - reference) ** 2))),
        "log_error_median": float(np.median(log_residual)),
        "delta1": float(np.mean(ratios < 1.25)),
        "delta2": float(np.mean(ratios < 1.25**2)),
        "pearson": correlation(reference, prediction),
        "spearman": correlation(rankdata(reference), rankdata(prediction)),
    }


def compute_within_view_structure_metrics(
    samples: np.ndarray, prediction_field: str
) -> dict:
    per_view_correlations = []
    pooled_reference = []
    pooled_prediction = []
    represented_views = 0
    for view_index in np.unique(samples["view_index"]):
        view_samples = samples[samples["view_index"] == view_index]
        reference = view_samples["reference_depth"]
        prediction = view_samples[prediction_field]
        if len(view_samples) < 20 or np.std(reference) <= 1e-8 or np.std(prediction) <= 1e-8:
            continue
        represented_views += 1
        per_view_correlations.append(correlation(reference, prediction))
        pooled_reference.append((reference - np.median(reference)) / np.std(reference))
        pooled_prediction.append((prediction - np.median(prediction)) / np.std(prediction))
    return {
        "represented_views": represented_views,
        "per_view_pearson_median": float(np.median(per_view_correlations)),
        "per_view_pearson_p10": float(np.percentile(per_view_correlations, 10)),
        "pooled_within_view_zscore_pearson": correlation(
            np.concatenate(pooled_reference), np.concatenate(pooled_prediction)
        ),
    }


def evaluate_scene(da3_root: Path, sparse_root: Path, output_root: Path) -> dict:
    names, da3_centers, da3_extrinsics, _ = load_da3(da3_root)
    npz = np.load(da3_root / "exports" / "mini_npz" / "results.npz")
    depths = np.asarray(npz["depth"], dtype=np.float64)
    confidences = np.asarray(npz["conf"], dtype=np.float64)
    camera = parse_colmap_camera(sparse_root / "cameras.txt")
    if camera["width"] != 5280 or camera["height"] != 3956:
        raise ValueError(f"Expected raw RGB domain 5280x3956, got {camera}")
    images = parse_images_with_poses(sparse_root / "images.txt")
    points = parse_points3d(sparse_root / "points3D.txt")
    missing = [name for name in names if name not in images]
    if missing:
        raise ValueError(f"Missing selected images in COLMAP model: {missing}")

    colmap_centers = np.asarray([images[name]["center"] for name in names])
    transform = fit_sim3(da3_centers, colmap_centers)
    depth_height, depth_width = depths.shape[-2:]
    scale_x = depth_width / camera["width"]
    scale_y = depth_height / camera["height"]
    samples = []

    for view_index, name in enumerate(names):
        image = images[name]
        rotation = image["rotation"]
        translation = image["translation"]
        for raw_x, raw_y, point_id in image["observations"]:
            point = points.get(point_id)
            if point is None:
                continue
            camera_xyz = rotation @ point["xyz"] + translation
            reference_depth = float(camera_xyz[2])
            if not np.isfinite(reference_depth) or reference_depth <= 0:
                continue
            da3_x = (raw_x + 0.5) * scale_x - 0.5
            da3_y = (raw_y + 0.5) * scale_y - 0.5
            raw_depth = bilinear_sample(depths[view_index], da3_x, da3_y)
            confidence = bilinear_sample(confidences[view_index], da3_x, da3_y)
            next_view_depth = bilinear_sample(
                depths[(view_index + 1) % len(depths)], da3_x, da3_y
            )
            mirrored_depth = bilinear_sample(
                depths[view_index], depth_width - 1 - da3_x, da3_y
            )
            if not np.isfinite(raw_depth) or raw_depth <= 0 or not np.isfinite(confidence):
                continue
            samples.append(
                (
                    view_index,
                    name,
                    point_id,
                    raw_x,
                    raw_y,
                    da3_x,
                    da3_y,
                    reference_depth,
                    raw_depth,
                    next_view_depth,
                    mirrored_depth,
                    confidence,
                    point["error"],
                    point["track_length"],
                )
            )

    dtype = [
        ("view_index", "i4"),
        ("image", "U128"),
        ("point_id", "i8"),
        ("raw_x", "f8"),
        ("raw_y", "f8"),
        ("da3_x", "f8"),
        ("da3_y", "f8"),
        ("reference_depth", "f8"),
        ("da3_depth_raw", "f8"),
        ("da3_depth_next_view", "f8"),
        ("da3_depth_mirrored_x", "f8"),
        ("confidence", "f8"),
        ("colmap_error", "f8"),
        ("track_length", "i4"),
    ]
    sample_array = np.asarray(samples, dtype=dtype)
    confidence_threshold = float(np.median(sample_array["confidence"]))
    quality_mask = (
        (sample_array["colmap_error"] <= 2.0)
        & (sample_array["track_length"] >= 3)
        & (sample_array["confidence"] >= confidence_threshold)
    )
    quality = sample_array[quality_mask]
    if len(quality) == 0:
        raise ValueError("No primary quality samples remain")

    camera_scale = transform.scale
    depth_optimal_scale = float(
        np.median(quality["reference_depth"] / quality["da3_depth_raw"])
    )
    scale_difference = float(
        abs(camera_scale - depth_optimal_scale)
        / (0.5 * (camera_scale + depth_optimal_scale))
    )

    all_prediction = sample_array["da3_depth_raw"] * camera_scale
    quality_prediction = quality["da3_depth_raw"] * camera_scale
    quality_optimal_prediction = quality["da3_depth_raw"] * depth_optimal_scale
    next_view_metrics = compute_metrics(
        quality["reference_depth"], quality["da3_depth_next_view"] * camera_scale
    )
    mirrored_metrics = compute_metrics(
        quality["reference_depth"], quality["da3_depth_mirrored_x"] * camera_scale
    )
    within_view_matched = compute_within_view_structure_metrics(
        quality, "da3_depth_raw"
    )
    within_view_next = compute_within_view_structure_metrics(
        quality, "da3_depth_next_view"
    )
    within_view_mirror = compute_within_view_structure_metrics(
        quality, "da3_depth_mirrored_x"
    )
    all_metrics = compute_metrics(sample_array["reference_depth"], all_prediction)
    quality_metrics = compute_metrics(quality["reference_depth"], quality_prediction)
    optimal_scale_metrics = compute_metrics(
        quality["reference_depth"], quality_optimal_prediction
    )

    per_view_rows = []
    for view_index, name in enumerate(names):
        view_samples = quality[quality["view_index"] == view_index]
        if len(view_samples) == 0:
            continue
        metrics = compute_metrics(
            view_samples["reference_depth"],
            view_samples["da3_depth_raw"] * camera_scale,
        )
        metrics["view_index"] = view_index
        metrics["image"] = name
        per_view_rows.append(metrics)

    checks = {
        "primary_samples_ge_2000": len(quality) >= 2000,
        "represented_views_ge_20": len(per_view_rows) >= 20,
        "camera_vs_depth_scale_difference_le_25pct": scale_difference <= 0.25,
        "absrel_median_le_0_20": quality_metrics["absrel_median"] <= 0.20,
        "absrel_p90_le_0_40": quality_metrics["absrel_p90"] <= 0.40,
        "delta1_ge_0_65": quality_metrics["delta1"] >= 0.65,
        "spearman_ge_0_50": quality_metrics["spearman"] >= 0.50,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "sparse_depth_samples.npz",
        samples=sample_array,
        quality_mask=quality_mask,
    )
    with (output_root / "sparse_depth_per_view.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "view_index",
            "image",
            "count",
            "reference_median",
            "prediction_median",
            "prediction_reference_ratio_median",
            "absrel_mean",
            "absrel_median",
            "absrel_p90",
            "rmse",
            "log_error_median",
            "delta1",
            "delta2",
            "pearson",
            "spearman",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_view_rows)

    rng = np.random.default_rng(42)
    plot_samples = quality
    if len(plot_samples) > 10000:
        plot_samples = plot_samples[rng.choice(len(plot_samples), 10000, replace=False)]
    plot_prediction = plot_samples["da3_depth_raw"] * camera_scale
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    axes[0].scatter(
        plot_samples["reference_depth"],
        plot_prediction,
        s=4,
        alpha=0.25,
        rasterized=True,
    )
    limits = [
        float(min(plot_samples["reference_depth"].min(), plot_prediction.min())),
        float(max(plot_samples["reference_depth"].max(), plot_prediction.max())),
    ]
    axes[0].plot(limits, limits, "--", color="black", lw=1)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("COLMAP sparse depth")
    axes[0].set_ylabel("DA3 depth x camera scale")
    axes[0].set_title("Sparse-depth agreement")

    view_sequences = [int(Path(row["image"]).stem.split("_")[-2]) for row in per_view_rows]
    axes[1].plot(
        view_sequences,
        [row["absrel_median"] for row in per_view_rows],
        "o-",
        ms=3.5,
        lw=1,
    )
    axes[1].axhline(0.20, color="tab:red", ls="--", lw=1, label="gate threshold")
    axes[1].set_xlabel("Image sequence")
    axes[1].set_ylabel("Median AbsRel")
    axes[1].set_title("Per-view depth residual")
    axes[1].legend(fontsize=7)

    axes[2].plot(
        view_sequences,
        [row["prediction_reference_ratio_median"] for row in per_view_rows],
        "o-",
        ms=3.5,
        lw=1,
    )
    axes[2].axhline(1.0, color="black", ls="--", lw=1)
    axes[2].set_xlabel("Image sequence")
    axes[2].set_ylabel("Median predicted/reference")
    axes[2].set_title("Per-view scale consistency")
    figure.tight_layout()
    figure.savefig(output_root / "sparse_depth_diagnostic.png", dpi=220)
    plt.close(figure)

    result = {
        "scene": da3_root.name,
        "selected_views": len(names),
        "colmap_registered_images": len(images),
        "colmap_points": len(points),
        "pixel_domain": {
            "raw_width": camera["width"],
            "raw_height": camera["height"],
            "da3_width": depth_width,
            "da3_height": depth_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "mapping": "(raw + 0.5) * scale - 0.5",
        },
        "sample_counts": {
            "all_geometrically_valid": int(len(sample_array)),
            "primary_quality": int(len(quality)),
            "represented_views": int(len(per_view_rows)),
            "confidence_median_threshold": confidence_threshold,
        },
        "scales": {
            "camera_sim3_scale": camera_scale,
            "sparse_depth_optimal_global_scale": depth_optimal_scale,
            "symmetric_relative_difference": scale_difference,
        },
        "all_samples_camera_scale_metrics": all_metrics,
        "primary_quality_camera_scale_metrics": quality_metrics,
        "primary_quality_optimal_scale_metrics_secondary": optimal_scale_metrics,
        "negative_controls": {
            "next_view_cyclic_mismatch_camera_scale_metrics": next_view_metrics,
            "same_view_horizontal_mirror_camera_scale_metrics": mirrored_metrics,
            "matched_vs_next_view_absrel_median_ratio": float(
                next_view_metrics["absrel_median"]
                / max(quality_metrics["absrel_median"], 1e-8)
            ),
            "matched_vs_mirror_absrel_median_ratio": float(
                mirrored_metrics["absrel_median"]
                / max(quality_metrics["absrel_median"], 1e-8)
            ),
        },
        "within_view_structure": {
            "matched": within_view_matched,
            "next_view_cyclic_mismatch": within_view_next,
            "same_view_horizontal_mirror": within_view_mirror,
            "interpretation": (
                "Per-view centering and standardization remove the dominant scene/camera depth offset. "
                "Matched values should exceed negative controls if the maps carry spatial depth structure."
            ),
        },
        "provisional_gate": {
            "status": "provisional_pass" if all(checks.values()) else "provisional_fail",
            "checks": checks,
            "note": "Thresholds were declared in DA3_GATE_B_SPARSE_DEPTH_PROTOCOL_20260615.md before this evaluation.",
        },
    }
    (output_root / "sparse_depth_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maize-da3-root", type=Path, required=True)
    parser.add_argument("--maize-sparse-root", type=Path, required=True)
    parser.add_argument("--road-da3-root", type=Path, required=True)
    parser.add_argument("--road-sparse-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    scenes = {
        "maize_02_20260526_1658": (args.maize_da3_root, args.maize_sparse_root),
        "road_01_20260602_1648_40m": (args.road_da3_root, args.road_sparse_root),
    }
    results = {}
    for scene, (da3_root, sparse_root) in scenes.items():
        results[scene] = evaluate_scene(
            da3_root=da3_root,
            sparse_root=sparse_root,
            output_root=args.output_root / scene,
        )

    (args.output_root / "da3_gate_b_sparse_depth_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    lines = [
        "# DA3 Gate-B Sparse-Depth Validation",
        "",
        "Existing DA3-LARGE-1.1 outputs were evaluated against matched COLMAP sparse depths in the original distorted RGB pixel domain.",
        "",
    ]
    for scene, result in results.items():
        metrics = result["primary_quality_camera_scale_metrics"]
        scales = result["scales"]
        gate = result["provisional_gate"]
        counts = result["sample_counts"]
        controls = result["negative_controls"]
        structure = result["within_view_structure"]
        lines.extend(
            [
                f"## {scene}",
                "",
                f"- Primary samples/views: {counts['primary_quality']}/{counts['represented_views']}",
                f"- Camera Sim(3) scale: {scales['camera_sim3_scale']:.6f}",
                f"- Sparse-depth optimal scale: {scales['sparse_depth_optimal_global_scale']:.6f}",
                f"- Scale difference: {scales['symmetric_relative_difference']:.2%}",
                f"- Median/P90 AbsRel: {metrics['absrel_median']:.4f}/{metrics['absrel_p90']:.4f}",
                f"- Delta-1 / Delta-2: {metrics['delta1']:.3f}/{metrics['delta2']:.3f}",
                f"- Pearson / Spearman: {metrics['pearson']:.3f}/{metrics['spearman']:.3f}",
                f"- Negative-control median AbsRel, next-view / mirror: {controls['next_view_cyclic_mismatch_camera_scale_metrics']['absrel_median']:.4f}/{controls['same_view_horizontal_mirror_camera_scale_metrics']['absrel_median']:.4f}",
                f"- Negative-control degradation ratio: {controls['matched_vs_next_view_absrel_median_ratio']:.1f}x/{controls['matched_vs_mirror_absrel_median_ratio']:.1f}x",
                f"- Within-view z-score correlation, matched / next-view / mirror: {structure['matched']['pooled_within_view_zscore_pearson']:.3f}/{structure['next_view_cyclic_mismatch']['pooled_within_view_zscore_pearson']:.3f}/{structure['same_view_horizontal_mirror']['pooled_within_view_zscore_pearson']:.3f}",
                f"- Provisional status: **{gate['status']}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "This is a sparse feature-depth feasibility check. It does not establish dense ground truth or authorize replacement of the current manuscript diagnostic.",
        ]
    )
    (args.output_root / "DA3_GATE_B_SPARSE_DEPTH_REPORT_20260615.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
