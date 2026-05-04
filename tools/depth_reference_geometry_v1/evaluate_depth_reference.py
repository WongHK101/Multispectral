from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from depth_reference_common import load_json, save_json, write_simple_csv


def _format_threshold(value: float) -> str:
    return f"{float(value):.6g}"


def _raw_depth_to_metric_camera_z(raw_depth: np.ndarray, depth_semantics: str) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float64)
    if depth_semantics in {
        "metric_camera_z_from_renderer",
        "metric_camera_z_from_point_splat_centers",
    }:
        return raw_depth
    if depth_semantics == "inverse_camera_z_from_renderer":
        metric = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(raw_depth) & (raw_depth > 0.0)
        metric[positive] = 1.0 / raw_depth[positive]
        return metric
    raise ValueError(f"Unsupported depth semantics: {depth_semantics!r}")


def _make_model_valid_mask(metric_depth: np.ndarray, opacity: np.ndarray, depth_min: float, opacity_threshold: float) -> np.ndarray:
    return (
        np.isfinite(metric_depth)
        & np.isfinite(opacity)
        & (metric_depth > float(depth_min))
        & (opacity >= float(opacity_threshold))
    )


def _load_runtime_flags(out_dir: Path) -> Dict[str, Any]:
    for candidate_dir in [out_dir] + list(out_dir.parents):
        flags_path = candidate_dir / "depth_reference_runtime_flags.json"
        if not flags_path.exists():
            continue
        with flags_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Runtime flags file must contain a JSON object: {flags_path}")
        return data
    return {}


def _parse_thresholds(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if str(x).strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    if any(v <= 0 for v in values):
        raise ValueError(f"Thresholds must be positive, got {values!r}")
    return values


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a rendered model depth bundle against a training-only reference depth bundle")
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--model_manifest", required=True)
    parser.add_argument("--adapter_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--enable_agreement_metrics",
        action="store_true",
        help="Also compute symmetric depth-agreement statistics; default OFF for backward compatibility.",
    )
    parser.add_argument(
        "--error_mode",
        default="absolute_depth",
        choices=("absolute_depth", "relative_depth"),
        help=(
            "absolute_depth compares D_model - D_ref in the manifest's distance unit. "
            "relative_depth compares (D_model - D_ref) / D_ref and treats thresholds as ratios."
        ),
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional comma-separated threshold override. For relative_depth, values are ratios such as 0.01 for 1%%.",
    )
    parser.add_argument(
        "--relative_depth_min",
        type=float,
        default=1e-6,
        help="Minimum positive reference depth required for relative_depth evaluation.",
    )
    parser.add_argument(
        "--camera_intrinsic_tolerance",
        type=float,
        default=1e-4,
        help="Fail if a matched model/reference view differs by more than this in fx/fy/cx/cy.",
    )
    parser.add_argument(
        "--camera_pose_tolerance",
        type=float,
        default=1e-5,
        help="Fail if a matched model/reference view differs by more than this in camera_to_world.",
    )
    return parser


def _max_abs_camera_delta(ref_view: Dict[str, Any], model_view: Dict[str, Any]) -> Dict[str, float]:
    intrinsic_delta = 0.0
    for key in ("fx", "fy", "cx", "cy"):
        intrinsic_delta = max(intrinsic_delta, abs(float(ref_view[key]) - float(model_view[key])))
    ref_pose = np.asarray(ref_view["camera_to_world"], dtype=np.float64)
    model_pose = np.asarray(model_view["camera_to_world"], dtype=np.float64)
    if ref_pose.shape != model_pose.shape:
        pose_delta = float("inf")
    else:
        pose_delta = float(np.max(np.abs(ref_pose - model_pose)))
    width_delta = abs(int(ref_view["width"]) - int(model_view["width"]))
    height_delta = abs(int(ref_view["height"]) - int(model_view["height"]))
    return {
        "intrinsic": intrinsic_delta,
        "pose": pose_delta,
        "width": float(width_delta),
        "height": float(height_delta),
    }


def main() -> None:
    args = _argparser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_manifest_path = Path(args.reference_manifest).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    adapter_manifest_path = Path(args.adapter_manifest).resolve()

    ref_manifest = load_json(ref_manifest_path)
    model_manifest = load_json(model_manifest_path)
    adapter_manifest = load_json(adapter_manifest_path)
    runtime_flags = _load_runtime_flags(out_dir)
    enable_agreement_metrics = bool(args.enable_agreement_metrics or runtime_flags.get("enable_agreement_metrics", False))

    error_mode = str(args.error_mode)
    thresholds_m = _parse_thresholds(args.thresholds) if args.thresholds else [float(x) for x in ref_manifest["thresholds_m"]]
    distance_unit = str(ref_manifest.get("distance_unit", "meters"))
    scale_mode = str(ref_manifest.get("scale_mode", "metric_verified"))
    if error_mode == "relative_depth":
        distance_unit = "relative_depth_ratio"
        scale_mode = "relative_depth"
    adapter_semantics = str(adapter_manifest["depth_semantics"])
    validity_rule = adapter_manifest["validity_rule"]
    depth_min = float(validity_rule.get("depth_min", 1e-6))
    opacity_threshold = float(validity_rule.get("opacity_threshold", 0.5))
    relative_depth_min = float(args.relative_depth_min)
    if relative_depth_min <= 0.0:
        raise ValueError(f"--relative_depth_min must be positive, got {relative_depth_min}")

    ref_by_name = {str(v["image_name"]): v for v in ref_manifest["views"]}
    model_by_name = {str(v["image_name"]): v for v in model_manifest["views"]}
    missing_in_model = sorted(set(ref_by_name) - set(model_by_name))
    extra_in_model = sorted(set(model_by_name) - set(ref_by_name))
    if missing_in_model:
        sample = ", ".join(missing_in_model[:8])
        raise ValueError(f"Model bundle is missing reference views: {sample}")
    if extra_in_model:
        sample = ", ".join(extra_in_model[:8])
        raise ValueError(f"Model bundle has extra views not in reference: {sample}")

    max_intrinsic_delta = 0.0
    max_pose_delta = 0.0
    for image_name in sorted(ref_by_name):
        ref_view = ref_by_name[image_name]
        model_view = model_by_name[image_name]
        delta = _max_abs_camera_delta(ref_view, model_view)
        max_intrinsic_delta = max(max_intrinsic_delta, delta["intrinsic"])
        max_pose_delta = max(max_pose_delta, delta["pose"])
        if int(delta["width"]) != 0 or int(delta["height"]) != 0:
            raise ValueError(
                f"Camera resolution mismatch for {image_name}: "
                f"ref=({ref_view['width']},{ref_view['height']}) "
                f"model=({model_view['width']},{model_view['height']})"
            )
        if delta["intrinsic"] > float(args.camera_intrinsic_tolerance):
            raise ValueError(
                f"Camera intrinsic mismatch for {image_name}: "
                f"max_abs_delta={delta['intrinsic']:.12g} "
                f"> tolerance={float(args.camera_intrinsic_tolerance):.12g}"
            )
        if delta["pose"] > float(args.camera_pose_tolerance):
            raise ValueError(
                f"Camera pose mismatch for {image_name}: "
                f"max_abs_delta={delta['pose']:.12g} "
                f"> tolerance={float(args.camera_pose_tolerance):.12g}"
            )

    total_ref_valid = 0
    total_model_valid_on_ref = 0
    total_missing = 0
    intrusion_counts = {d: 0 for d in thresholds_m}
    too_deep_counts = {d: 0 for d in thresholds_m}
    agreement_counts = {d: 0 for d in thresholds_m}
    intrusion_sum = {d: 0.0 for d in thresholds_m}
    abs_errors: List[np.ndarray] = []
    signed_errors: List[np.ndarray] = []
    per_view_rows: List[List[Any]] = []
    ignored_reference_due_to_relative_depth_min = 0

    for image_name in sorted(ref_by_name):
        ref_view = ref_by_name[image_name]
        model_view = model_by_name[image_name]
        ref_npz = np.load(ref_manifest_path.parent / ref_view["npz_file"])
        model_npz = np.load(model_manifest_path.parent / model_view["npz_file"])

        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        raw_model_depth = np.asarray(model_npz["depth"], dtype=np.float64)
        model_opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
        model_depth = _raw_depth_to_metric_camera_z(raw_model_depth, depth_semantics=adapter_semantics)
        model_valid = _make_model_valid_mask(model_depth, model_opacity, depth_min=depth_min, opacity_threshold=opacity_threshold)

        if ref_depth.shape != model_depth.shape:
            raise ValueError(f"Shape mismatch for {image_name}: ref {ref_depth.shape} vs model {model_depth.shape}")
        if ref_depth.shape != model_opacity.shape:
            raise ValueError(f"Opacity shape mismatch for {image_name}: ref {ref_depth.shape} vs opacity {model_opacity.shape}")

        eval_mask = ref_valid
        if error_mode == "relative_depth":
            ref_depth_positive = np.isfinite(ref_depth) & (ref_depth > relative_depth_min)
            ignored_reference_due_to_relative_depth_min += int(np.count_nonzero(eval_mask & (~ref_depth_positive)))
            eval_mask = eval_mask & ref_depth_positive
        valid_joint = eval_mask & model_valid
        total_ref_valid += int(np.count_nonzero(eval_mask))
        total_model_valid_on_ref += int(np.count_nonzero(valid_joint))
        missing_mask = eval_mask & (~model_valid)
        total_missing += int(np.count_nonzero(missing_mask))

        signed_metric = np.full(ref_depth.shape, np.nan, dtype=np.float64)
        if error_mode == "absolute_depth":
            signed_metric[valid_joint] = model_depth[valid_joint] - ref_depth[valid_joint]
        elif error_mode == "relative_depth":
            signed_metric[valid_joint] = (model_depth[valid_joint] - ref_depth[valid_joint]) / ref_depth[valid_joint]
        else:
            raise ValueError(f"Unsupported error_mode: {error_mode!r}")

        if np.any(valid_joint):
            signed = signed_metric[valid_joint]
            signed_errors.append(np.asarray(signed, dtype=np.float64))
            abs_errors.append(np.asarray(np.abs(signed), dtype=np.float64))
        else:
            signed = np.zeros((0,), dtype=np.float64)

        row: List[Any] = [
            image_name,
            int(np.count_nonzero(eval_mask)),
            int(np.count_nonzero(valid_joint)),
            int(np.count_nonzero(missing_mask)),
        ]

        for delta in thresholds_m:
            intrusion = valid_joint & (signed_metric < -delta)
            too_deep = valid_joint & (signed_metric > delta)
            intrusion_count = int(np.count_nonzero(intrusion))
            too_deep_count = int(np.count_nonzero(too_deep))
            if enable_agreement_metrics:
                agreement = valid_joint & (np.abs(signed_metric) <= delta)
                agreement_count = int(np.count_nonzero(agreement))
                agreement_counts[delta] += agreement_count
            intrusion_counts[delta] += intrusion_count
            too_deep_counts[delta] += too_deep_count
            if intrusion_count > 0:
                intrusion_sum[delta] += float(np.sum(-signed_metric[intrusion]))
            row.extend([intrusion_count, too_deep_count])
            if enable_agreement_metrics:
                row.append(agreement_count)
        per_view_rows.append(row)

    if total_ref_valid <= 0:
        raise ValueError("Reference valid set is empty")

    if abs_errors:
        abs_concat = np.concatenate(abs_errors, axis=0)
        signed_concat = np.concatenate(signed_errors, axis=0)
        abs_mean = float(np.mean(abs_concat))
        abs_median = float(np.median(abs_concat))
        signed_mean = float(np.mean(signed_concat))
    else:
        abs_mean = float("nan")
        abs_median = float("nan")
        signed_mean = float("nan")

    curve_rows: List[List[Any]] = []
    summary_rows: List[List[Any]] = []
    for delta in thresholds_m:
        intrusion_count = intrusion_counts[delta]
        too_deep_count = too_deep_counts[delta]
        intrusion_rate = float(intrusion_count) / float(total_ref_valid)
        too_deep_rate = float(too_deep_count) / float(total_ref_valid)
        agreement_rate = float(agreement_counts[delta]) / float(total_ref_valid) if enable_agreement_metrics else None
        intrusion_magnitude = float(intrusion_sum[delta] / intrusion_count) if intrusion_count > 0 else 0.0
        curve_row = [
            adapter_manifest["method_name"],
            _format_threshold(delta),
            f"{intrusion_rate:.12f}",
            f"{intrusion_magnitude:.12f}",
            f"{too_deep_rate:.12f}",
        ]
        if enable_agreement_metrics:
            curve_row.append(f"{agreement_rate:.12f}")
        curve_rows.append(curve_row)
        summary_row = [
            adapter_manifest["method_name"],
            _format_threshold(delta),
            f"{intrusion_rate:.12f}",
            f"{intrusion_magnitude:.12f}",
            f"{too_deep_rate:.12f}",
            f"{(float(total_missing) / float(total_ref_valid)):.12f}",
            f"{abs_mean:.12f}" if np.isfinite(abs_mean) else "nan",
            f"{abs_median:.12f}" if np.isfinite(abs_median) else "nan",
            f"{signed_mean:.12f}" if np.isfinite(signed_mean) else "nan",
        ]
        if enable_agreement_metrics:
            summary_row.append(f"{agreement_rate:.12f}")
        summary_rows.append(summary_row)

    summary_payload = {
        "protocol_name": "reference-depth-based-geometric-evaluation-v1",
        "scene_name": ref_manifest["scene_name"],
        "method_name": adapter_manifest["method_name"],
        "reference_manifest": str(ref_manifest_path),
        "model_manifest": str(model_manifest_path),
        "adapter_manifest": str(adapter_manifest_path),
        "depth_semantics": adapter_semantics,
        "distance_unit": distance_unit,
        "scale_mode": scale_mode,
        "error_mode": error_mode,
        "threshold_unit": "relative_depth_ratio" if error_mode == "relative_depth" else distance_unit,
        "validity_rule": validity_rule,
        "evaluation_options": {
            "enable_agreement_metrics": enable_agreement_metrics,
            "relative_depth_min": relative_depth_min if error_mode == "relative_depth" else None,
            "thresholds_source": "override" if args.thresholds else "reference_manifest",
            "camera_intrinsic_tolerance": float(args.camera_intrinsic_tolerance),
            "camera_pose_tolerance": float(args.camera_pose_tolerance),
            "max_camera_intrinsic_abs_delta": float(max_intrinsic_delta),
            "max_camera_pose_abs_delta": float(max_pose_delta),
        },
        "counts": {
            "reference_valid_pixels": int(total_ref_valid),
            "model_valid_on_reference_pixels": int(total_model_valid_on_ref),
            "missing_pixels": int(total_missing),
            "ignored_reference_pixels_due_to_relative_depth_min": int(ignored_reference_due_to_relative_depth_min),
        },
        "secondary_metrics": {
            "ModelValidOnReferenceRate": float(total_model_valid_on_ref) / float(total_ref_valid),
            "MissingRate": float(total_missing) / float(total_ref_valid),
            "AbsDepthError_Mean": abs_mean,
            "AbsDepthError_Median": abs_median,
            "SignedDepthBias_Mean": signed_mean,
        },
        "threshold_metrics": [
            {
                "threshold_m": float(delta),
                "threshold": float(delta),
                "threshold_unit": "relative_depth_ratio" if error_mode == "relative_depth" else distance_unit,
                "FrontIntrusionRate": float(intrusion_counts[delta]) / float(total_ref_valid),
                "FrontIntrusionMagnitude": float(intrusion_sum[delta] / intrusion_counts[delta]) if intrusion_counts[delta] > 0 else 0.0,
                "TooDeepRate": float(too_deep_counts[delta]) / float(total_ref_valid),
                **(
                    {"DepthAgreementRate": float(agreement_counts[delta]) / float(total_ref_valid)}
                    if enable_agreement_metrics
                    else {}
                ),
            }
            for delta in thresholds_m
        ],
    }
    save_json(out_dir / "metrics_summary.json", summary_payload)
    write_simple_csv(
        out_dir / "metrics_summary.csv",
        [
            "method_name",
            "threshold_m",
            "FrontIntrusionRate",
            "FrontIntrusionMagnitude",
            "TooDeepRate",
            "MissingRate",
            "AbsDepthError_Mean",
            "AbsDepthError_Median",
            "SignedDepthBias_Mean",
            *(
                ["DepthAgreementRate"]
                if enable_agreement_metrics
                else []
            ),
        ],
        summary_rows,
    )
    write_simple_csv(
        out_dir / "front_intrusion_curve.csv",
        [
            "method_name",
            "threshold_m",
            "FrontIntrusionRate",
            "FrontIntrusionMagnitude",
            "TooDeepRate",
            *(
                ["DepthAgreementRate"]
                if enable_agreement_metrics
                else []
            ),
        ],
        curve_rows,
    )
    per_view_header = ["image_name", "reference_valid_pixels", "model_valid_pixels", "missing_pixels"]
    for delta in thresholds_m:
        threshold_label = _format_threshold(delta)
        per_view_header.extend([f"FrontIntrusionCount@{threshold_label}", f"TooDeepCount@{threshold_label}"])
        if enable_agreement_metrics:
            per_view_header.append(f"DepthAgreementCount@{threshold_label}")
    write_simple_csv(out_dir / "per_view_counts.csv", per_view_header, per_view_rows)
    print(f"DEPTH_REFERENCE_METRICS {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
