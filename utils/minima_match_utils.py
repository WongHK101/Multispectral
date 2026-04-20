from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np

HOMOGRAPHY_PARAM_NAMES = ("h00", "h01", "h02_tx", "h10", "h11", "h12_ty", "h20", "h21")


def normalize_homography_matrix(homography: np.ndarray) -> np.ndarray:
    H = np.asarray(homography, dtype=np.float64)
    denom = H[2, 2] if abs(float(H[2, 2])) > 1e-12 else 1.0
    return H / denom


def _homography_param_stack(homographies: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack([normalize_homography_matrix(h) for h in homographies], axis=0)
    return np.stack(
        [
            stack[:, 0, 0],
            stack[:, 0, 1],
            stack[:, 0, 2],
            stack[:, 1, 0],
            stack[:, 1, 1],
            stack[:, 1, 2],
            stack[:, 2, 0],
            stack[:, 2, 1],
        ],
        axis=1,
    )


def _summarize_numeric(values: Sequence[float]) -> Dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "p05": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "std": None,
            "mad": None,
        }
    median = float(np.median(arr))
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "mean": float(np.mean(arr)),
        "median": median,
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "mad": float(np.median(np.abs(arr - median))),
    }


def _source_grid_points(source_shape: Sequence[int], grid_size: int) -> np.ndarray:
    height = int(source_shape[0])
    width = int(source_shape[1])
    grid = int(max(grid_size, 2))
    xs = np.linspace(0.0, float(max(width - 1, 0)), num=grid)
    ys = np.linspace(0.0, float(max(height - 1, 0)), num=grid)
    return np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, normalize_homography_matrix(homography))
    return projected.reshape(-1, 2)


def homography_pair_displacement_summary(
    reference_homography: np.ndarray,
    candidate_homography: np.ndarray,
    source_shape: Sequence[int],
    grid_size: int = 5,
) -> Dict[str, object]:
    points = _source_grid_points(source_shape, grid_size=grid_size)
    reference_points = _project_points(reference_homography, points)
    candidate_points = _project_points(candidate_homography, points)
    displacement = np.linalg.norm(candidate_points - reference_points, axis=1)
    source_long_edge = float(max(int(source_shape[0]), int(source_shape[1]), 1))
    return {
        "source_shape": [int(source_shape[0]), int(source_shape[1])],
        "source_long_edge_px": source_long_edge,
        "grid_size": int(max(grid_size, 2)),
        "num_points": int(points.shape[0]),
        "mean_px": float(np.mean(displacement)),
        "median_px": float(np.median(displacement)),
        "max_px": float(np.max(displacement)),
        "mean_rel": float(np.mean(displacement) / source_long_edge),
        "median_rel": float(np.median(displacement) / source_long_edge),
        "max_rel": float(np.max(displacement) / source_long_edge),
        "summary_px": _summarize_numeric(displacement),
        "summary_rel": _summarize_numeric(displacement / source_long_edge),
    }


def homography_stability_diagnostics(
    homographies: Sequence[np.ndarray],
    weights: Sequence[float] | None,
    aggregate_homography: np.ndarray,
    source_shape: Sequence[int],
    grid_size: int = 5,
) -> Dict[str, object]:
    matrices = [normalize_homography_matrix(h) for h in homographies]
    if not matrices:
        source_long_edge = float(max(int(source_shape[0]), int(source_shape[1]), 1))
        return {
            "num_homographies": 0,
            "source_shape": [int(source_shape[0]), int(source_shape[1])],
            "source_long_edge_px": source_long_edge,
            "robust_keep_count": 0,
            "robust_reject_count": 0,
            "per_homography": [],
        }

    params = _homography_param_stack(matrices)
    linear_dets = [float(np.linalg.det(H[:2, :2])) for H in matrices]
    scale_x = [float(np.linalg.norm(H[:2, 0])) for H in matrices]
    scale_y = [float(np.linalg.norm(H[:2, 1])) for H in matrices]
    center = np.median(params, axis=0)
    dist = np.linalg.norm(params - center[None, :], axis=1)
    if len(matrices) == 1:
        robust_keep = np.ones((1,), dtype=bool)
    else:
        med = float(np.median(dist))
        mad = float(np.median(np.abs(dist - med))) + 1e-9
        robust_keep = dist <= (med + 2.5 * mad)
        if not np.any(robust_keep):
            robust_keep = np.ones_like(dist, dtype=bool)

    if weights is None:
        raw_weights = np.ones((len(matrices),), dtype=np.float64)
    else:
        raw_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if raw_weights.size != len(matrices):
            raw_weights = np.ones((len(matrices),), dtype=np.float64)
    raw_weights = np.clip(raw_weights, 1e-6, None)
    aggregation_weights = np.zeros_like(raw_weights, dtype=np.float64)
    if np.any(robust_keep):
        kept_weights = raw_weights[robust_keep]
        aggregation_weights[robust_keep] = kept_weights / float(np.sum(kept_weights))

    points = _source_grid_points(source_shape, grid_size=grid_size)
    aggregate_points = _project_points(aggregate_homography, points)
    source_long_edge = float(max(int(source_shape[0]), int(source_shape[1]), 1))
    per_homography = []
    mean_displacements = []
    median_displacements = []
    max_displacements = []
    for idx, H in enumerate(matrices):
        projected = _project_points(H, points)
        displacement = np.linalg.norm(projected - aggregate_points, axis=1)
        mean_px = float(np.mean(displacement))
        median_px = float(np.median(displacement))
        max_px = float(np.max(displacement))
        mean_displacements.append(mean_px)
        median_displacements.append(median_px)
        max_displacements.append(max_px)
        per_homography.append(
            {
                "index": int(idx),
                "param_distance_to_median": float(dist[idx]),
                "robust_keep": bool(robust_keep[idx]),
                "raw_weight": float(raw_weights[idx]),
                "aggregation_weight": float(aggregation_weights[idx]),
                "disp_vs_aggregate_grid_mean_px": mean_px,
                "disp_vs_aggregate_grid_median_px": median_px,
                "disp_vs_aggregate_grid_max_px": max_px,
                "disp_vs_aggregate_grid_mean_rel": float(mean_px / source_long_edge),
                "disp_vs_aggregate_grid_median_rel": float(median_px / source_long_edge),
                "disp_vs_aggregate_grid_max_rel": float(max_px / source_long_edge),
            }
        )

    pairwise_mean = []
    pairwise_max = []
    projected_points = [_project_points(H, points) for H in matrices]
    for i in range(len(projected_points)):
        for j in range(i + 1, len(projected_points)):
            displacement = np.linalg.norm(projected_points[i] - projected_points[j], axis=1)
            pairwise_mean.append(float(np.mean(displacement)))
            pairwise_max.append(float(np.max(displacement)))

    return {
        "num_homographies": int(len(matrices)),
        "source_shape": [int(source_shape[0]), int(source_shape[1])],
        "source_long_edge_px": source_long_edge,
        "grid_size": int(max(grid_size, 2)),
        "num_grid_points": int(points.shape[0]),
        "param_names": list(HOMOGRAPHY_PARAM_NAMES),
        "param_summary": {
            name: _summarize_numeric(params[:, idx])
            for idx, name in enumerate(HOMOGRAPHY_PARAM_NAMES)
        },
        "linear_det_summary": _summarize_numeric(linear_dets),
        "scale_x_summary": _summarize_numeric(scale_x),
        "scale_y_summary": _summarize_numeric(scale_y),
        "param_distance_to_median_summary": _summarize_numeric(dist),
        "robust_keep_count": int(np.sum(robust_keep)),
        "robust_reject_count": int(np.sum(~robust_keep)),
        "robust_reject_ratio": float(np.mean(~robust_keep)),
        "disp_vs_aggregate_grid_mean_px": _summarize_numeric(mean_displacements),
        "disp_vs_aggregate_grid_median_px": _summarize_numeric(median_displacements),
        "disp_vs_aggregate_grid_max_px": _summarize_numeric(max_displacements),
        "disp_vs_aggregate_grid_mean_rel": _summarize_numeric(np.asarray(mean_displacements, dtype=np.float64) / source_long_edge),
        "disp_vs_aggregate_grid_median_rel": _summarize_numeric(np.asarray(median_displacements, dtype=np.float64) / source_long_edge),
        "disp_vs_aggregate_grid_max_rel": _summarize_numeric(np.asarray(max_displacements, dtype=np.float64) / source_long_edge),
        "pairwise_grid_mean_px": _summarize_numeric(pairwise_mean),
        "pairwise_grid_max_px": _summarize_numeric(pairwise_max),
        "pairwise_grid_mean_rel": _summarize_numeric(np.asarray(pairwise_mean, dtype=np.float64) / source_long_edge),
        "pairwise_grid_max_rel": _summarize_numeric(np.asarray(pairwise_max, dtype=np.float64) / source_long_edge),
        "per_homography": per_homography,
    }


def filter_matches_by_confidence(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    mconf: np.ndarray,
    conf_thresh: float,
) -> Dict[str, np.ndarray]:
    pts0 = np.asarray(mkpts0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(mkpts1, dtype=np.float32).reshape(-1, 2)
    conf = np.asarray(mconf, dtype=np.float32).reshape(-1)
    if pts0.shape[0] != pts1.shape[0] or pts0.shape[0] != conf.shape[0]:
        raise ValueError(
            f"Match array shape mismatch: mkpts0={pts0.shape}, mkpts1={pts1.shape}, mconf={conf.shape}"
        )
    keep = conf >= float(conf_thresh)
    return {
        "mkpts0": pts0[keep],
        "mkpts1": pts1[keep],
        "mconf": conf[keep],
    }


def estimate_homography_ransac(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    method: str = "usac_magsac",
    ransac_thresh: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 10000,
) -> Dict[str, object]:
    pts0 = np.asarray(mkpts0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(mkpts1, dtype=np.float32).reshape(-1, 2)
    num_matches = int(pts0.shape[0])
    if num_matches < 4:
        return {
            "success": False,
            "H": np.eye(3, dtype=np.float64),
            "inlier_mask": np.zeros((num_matches,), dtype=bool),
            "num_matches": num_matches,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": float("inf"),
            "ransac_method": method,
        }

    if str(method).lower() == "usac_magsac" and hasattr(cv2, "USAC_MAGSAC"):
        cv_method = int(cv2.USAC_MAGSAC)
    else:
        cv_method = int(cv2.RANSAC)

    # We estimate a band->RGB mapping, so src is band points and dst is RGB points.
    H, inliers = cv2.findHomography(
        pts1,
        pts0,
        method=cv_method,
        ransacReprojThreshold=float(ransac_thresh),
        confidence=float(confidence),
        maxIters=int(max_iters),
    )
    if H is None or inliers is None:
        return {
            "success": False,
            "H": np.eye(3, dtype=np.float64),
            "inlier_mask": np.zeros((num_matches,), dtype=bool),
            "num_matches": num_matches,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": float("inf"),
            "ransac_method": method,
        }

    inlier_mask = inliers.reshape(-1).astype(bool)
    num_inliers = int(np.sum(inlier_mask))
    inlier_ratio = float(num_inliers) / float(max(num_matches, 1))

    reproj_error = float("inf")
    if num_inliers >= 4:
        src_in = pts1[inlier_mask].reshape(-1, 1, 2)
        dst_in = pts0[inlier_mask]
        proj = cv2.perspectiveTransform(src_in, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst_in, axis=1)
        if err.size > 0:
            reproj_error = float(np.mean(err))

    return {
        "success": True,
        "H": np.asarray(H, dtype=np.float64),
        "inlier_mask": inlier_mask,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "reproj_error": reproj_error,
        "ransac_method": method,
    }


def compute_match_spatial_coverage(
    points: np.ndarray,
    image_shape: Sequence[int],
    grid_size: int = 4,
) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return 0.0
    height = int(image_shape[0])
    width = int(image_shape[1])
    grid = int(max(grid_size, 1))
    if height <= 1 or width <= 1:
        return 0.0
    x = np.clip(pts[:, 0], 0.0, float(width - 1))
    y = np.clip(pts[:, 1], 0.0, float(height - 1))
    gx = np.clip((x / float(width) * grid).astype(np.int32), 0, grid - 1)
    gy = np.clip((y / float(height) * grid).astype(np.int32), 0, grid - 1)
    occupied = set((int(ix), int(iy)) for ix, iy in zip(gx, gy))
    return float(len(occupied)) / float(grid * grid)


def score_frame_alignment_quality(
    match_stats: Dict[str, object],
    coverage: float,
    conf_stats: Dict[str, float],
) -> float:
    inlier_ratio = float(match_stats.get("inlier_ratio", 0.0))
    reproj = float(match_stats.get("reproj_error", float("inf")))
    num_inliers = float(match_stats.get("num_inliers", 0.0))
    conf_mean = float(conf_stats.get("mean", 0.0))
    reproj_term = 1.0 / (1.0 + max(reproj, 0.0))
    return (
        0.42 * inlier_ratio
        + 0.28 * float(coverage)
        + 0.15 * reproj_term
        + 0.10 * conf_mean
        + 0.05 * np.tanh(num_inliers / 300.0)
    )


def accept_frame_for_global_pool(
    match_stats: Dict[str, object],
    coverage: float,
    cfg: Dict[str, object],
) -> bool:
    return (
        bool(match_stats.get("success", False))
        and int(match_stats.get("num_matches", 0)) >= int(cfg.get("minima_min_matches", 80))
        and float(match_stats.get("inlier_ratio", 0.0)) >= float(cfg.get("minima_min_inlier_ratio", 0.30))
        and float(match_stats.get("reproj_error", float("inf"))) <= float(cfg.get("minima_max_reproj_error", 4.0))
        and float(coverage) >= float(cfg.get("minima_min_coverage", 0.25))
    )


def robust_aggregate_homographies_weighted(
    homographies: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    matrices = [np.asarray(h, dtype=np.float64) for h in homographies]
    if not matrices:
        raise ValueError("No homographies to aggregate.")
    if len(matrices) == 1:
        return normalize_homography_matrix(matrices[0])

    params = _homography_param_stack(matrices)
    center = np.median(params, axis=0)
    dist = np.linalg.norm(params - center[None, :], axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) + 1e-9
    inlier_keep = dist <= (med + 2.5 * mad)
    if not np.any(inlier_keep):
        inlier_keep = np.ones_like(dist, dtype=bool)

    params_in = params[inlier_keep]
    if weights:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != len(matrices):
            w = np.ones((len(matrices),), dtype=np.float64)
        w_in = w[inlier_keep]
        w_in = np.clip(w_in, 1e-6, None)
        w_in = w_in / float(np.sum(w_in))
        agg = np.sum(params_in * w_in[:, None], axis=0)
    else:
        agg = np.mean(params_in, axis=0)

    H = np.array(
        [
            [agg[0], agg[1], agg[2]],
            [agg[3], agg[4], agg[5]],
            [agg[6], agg[7], 1.0],
        ],
        dtype=np.float64,
    )
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)


def build_candidate_pool_schedule(
    total_frames: int,
    base_frames: int,
    initial_ratio: float,
    ratio_step: float,
    max_ratio: float,
    include_all: bool = True,
) -> List[int]:
    total = int(max(total_frames, 0))
    if total <= 0:
        return []
    base = int(max(1, base_frames))
    current = int(max(base, round(total * float(initial_ratio))))
    cap = int(max(current, round(total * float(max_ratio))))
    step = int(max(1, round(total * float(ratio_step))))

    counts = []
    while current < cap:
        counts.append(int(min(current, total)))
        current += step
    counts.append(int(min(cap, total)))
    if include_all and total not in counts:
        counts.append(total)
    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for c in counts:
        c = int(max(1, min(c, total)))
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup

