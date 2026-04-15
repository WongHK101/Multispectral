from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np


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
        H = matrices[0]
        return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)

    stack = np.stack([h / (h[2, 2] if abs(h[2, 2]) > 1e-12 else 1.0) for h in matrices], axis=0)
    params = np.stack(
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

