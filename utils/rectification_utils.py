from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.spectral_image_utils import load_image_preserve_dtype, normalize_scalar_band_image


LEGACY_ECC_NOTE = "legacy_debug_only"


def load_rgb_plane_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image:
        image.load()
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr)


def load_raw_band_image(path: str | Path,
                        dynamic_range: str = "uint16",
                        radiometric_mode: str = "exposure_normalized") -> np.ndarray:
    loaded = load_image_preserve_dtype(path)
    raw = np.asarray(loaded.array)

    # Scalar path (multispectral single-band TIFF, etc.).
    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1):
        arr = normalize_scalar_band_image(
            image=loaded,
            metadata=loaded.metadata,
            mode=radiometric_mode,
            dynamic_range=dynamic_range,
        )
        return np.ascontiguousarray(arr.astype(np.float32))

    # RGB/false-color thermal path: convert to robust grayscale proxy for matching.
    if raw.ndim == 3 and raw.shape[2] >= 3:
        arr = raw.astype(np.float32, copy=False)
        if np.issubdtype(raw.dtype, np.integer):
            if str(dynamic_range).lower() == "uint8":
                denom = 255.0
            elif str(dynamic_range).lower() == "uint16":
                denom = 65535.0
            else:
                denom = float(np.iinfo(raw.dtype).max)
            arr = arr / max(denom, 1.0)
        # Luma-style projection keeps edge structure stable for cross-modal matching.
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
        gray = np.clip(gray, 0.0, 1.0)
        return np.ascontiguousarray(gray.astype(np.float32))

    raise ValueError(f"Unsupported band image shape for rectification loading: {raw.shape}")


def normalize_homography(homography: np.ndarray) -> np.ndarray:
    H = np.asarray(homography, dtype=np.float64)
    denom = float(H[2, 2]) if abs(float(H[2, 2])) > 1e-12 else 1.0
    return H / denom


def _scale_matrix(scale_x: float, scale_y: float) -> np.ndarray:
    return np.array(
        [
            [float(scale_x), 0.0, 0.0],
            [0.0, float(scale_y), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _first_metadata_value(metadata: Optional[dict], *keys: str) -> Optional[float]:
    if not isinstance(metadata, dict):
        return None
    lowered = {str(key).lower(): value for key, value in metadata.items()}
    for key in keys:
        if key.lower() in lowered:
            value = _safe_float(lowered[key.lower()])
            if value is not None:
                return value
    return None


def build_naive_h0(rgb_shape, band_shape) -> np.ndarray:
    rgb_h, rgb_w = int(rgb_shape[0]), int(rgb_shape[1])
    band_h, band_w = int(band_shape[0]), int(band_shape[1])
    sx = float(rgb_w) / float(max(band_w, 1))
    sy = float(rgb_h) / float(max(band_h, 1))
    tx = 0.5 * (float(rgb_w) - sx * float(band_w))
    ty = 0.5 * (float(rgb_h) - sy * float(band_h))
    H = np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_homography(H)


def metadata_has_alignment_prior(rgb_meta: Optional[dict], band_meta: Optional[dict]) -> bool:
    keys = (
        "alignment_offset_x", "alignment_offset_y",
        "band_offset_x", "band_offset_y",
        "principal_point_offset_x", "principal_point_offset_y",
        "center_offset_x", "center_offset_y",
        "relative_center_offset_x", "relative_center_offset_y",
        "scale_adjust_x", "scale_adjust_y",
    )
    return any(_first_metadata_value(band_meta, key) is not None for key in keys) or any(
        _first_metadata_value(rgb_meta, key) is not None for key in keys
    )


def build_metadata_assisted_h0(rgb_meta, band_meta, rgb_shape, band_shape) -> np.ndarray:
    H0 = build_naive_h0(rgb_shape, band_shape)
    tx = _first_metadata_value(
        band_meta,
        "alignment_offset_x",
        "band_offset_x",
        "principal_point_offset_x",
        "center_offset_x",
        "relative_center_offset_x",
    )
    ty = _first_metadata_value(
        band_meta,
        "alignment_offset_y",
        "band_offset_y",
        "principal_point_offset_y",
        "center_offset_y",
        "relative_center_offset_y",
    )
    sx_mul = _first_metadata_value(band_meta, "scale_adjust_x", "band_scale_x")
    sy_mul = _first_metadata_value(band_meta, "scale_adjust_y", "band_scale_y")

    if tx is None and ty is None and sx_mul is None and sy_mul is None:
        return H0

    delta = np.eye(3, dtype=np.float64)
    if sx_mul is not None and sx_mul > 0:
        delta[0, 0] = float(sx_mul)
    if sy_mul is not None and sy_mul > 0:
        delta[1, 1] = float(sy_mul)
    if tx is not None:
        delta[0, 2] = float(tx)
    if ty is not None:
        delta[1, 2] = float(ty)
    return normalize_homography(delta @ H0)


def evenly_sample_items(items: Sequence[str], count: int) -> List[str]:
    if count <= 0 or len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=count)
    sampled = [items[int(round(pos))] for pos in positions]
    deduped: List[str] = []
    seen = set()
    for item in sampled:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    for item in items:
        if len(deduped) >= count:
            break
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def resize_with_scale(image: np.ndarray, scale: float) -> np.ndarray:
    scale = float(scale)
    if scale >= 0.999:
        return np.ascontiguousarray(np.asarray(image))
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    out_w = max(int(round(w * scale)), 8)
    out_h = max(int(round(h * scale)), 8)
    resized = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if hi <= lo + 1e-8:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    return np.ascontiguousarray(arr.astype(np.float32))


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    max_val = float(np.max(mag)) if mag.size > 0 else 0.0
    if max_val > 1e-8:
        mag = mag / max_val
    return np.ascontiguousarray(mag.astype(np.float32))


def compute_binary_edge_map(gradient_magnitude: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    grad = np.asarray(gradient_magnitude, dtype=np.float32)
    threshold = float(np.percentile(grad, percentile))
    edges = (grad >= threshold).astype(np.uint8)
    return np.ascontiguousarray(edges)


def compute_edge_map(image: np.ndarray) -> np.ndarray:
    return compute_gradient_magnitude(image)


def prepare_alignment_images(rgb_img, band_img, mode: str = "gradient") -> dict:
    rgb_gray = _normalize_grayscale(rgb_img)
    band_gray = _normalize_grayscale(band_img)
    rgb_grad = compute_gradient_magnitude(rgb_gray)
    band_grad = compute_gradient_magnitude(band_gray)
    rgb_edges = compute_binary_edge_map(rgb_grad)
    band_edges = compute_binary_edge_map(band_grad)
    return {
        "rgb_gray": rgb_gray,
        "band_gray": band_gray,
        "rgb_grad": rgb_grad,
        "band_grad": band_grad,
        "rgb_edges": rgb_edges,
        "band_edges": band_edges,
        "mode": mode,
    }


def compute_structure_score(image: np.ndarray, max_dim: int = 1024) -> float:
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        arr = resize_with_scale(arr, scale=scale)
    grad = compute_gradient_magnitude(_normalize_grayscale(arr))
    return float(np.mean(grad))


def select_representative_frames(frame_records, max_frames, min_structure_score=None) -> list:
    if not frame_records:
        return []
    needs_structure_score = min_structure_score is not None
    scored_records = []
    for index, record in enumerate(frame_records):
        item = dict(record)
        score = item.get("structure_score", None if needs_structure_score else 0.0)
        if needs_structure_score and score is None:
            rgb_path = item.get("rgb_path", None)
            if rgb_path is None:
                score = 0.0
            else:
                score = compute_structure_score(load_rgb_plane_image(rgb_path))
        item["structure_score"] = float(score)
        item["order_index"] = index
        scored_records.append(item)

    if min_structure_score is not None:
        filtered = [item for item in scored_records if item["structure_score"] >= float(min_structure_score)]
        if filtered:
            scored_records = filtered

    scored_records = sorted(scored_records, key=lambda item: item["order_index"])
    if len(scored_records) <= max_frames:
        return scored_records

    names = [item["frame_id"] for item in scored_records]
    sampled = set(evenly_sample_items(names, max_frames))
    selected = [item for item in scored_records if item["frame_id"] in sampled]
    if len(selected) > max_frames:
        selected = selected[:max_frames]
    return selected


def warp_with_homography(image: np.ndarray,
                         H: np.ndarray,
                         out_size: Tuple[int, int],
                         interpolation=cv2.INTER_LINEAR,
                         border_value=0) -> np.ndarray:
    out_w, out_h = int(out_size[0]), int(out_size[1])
    warped = cv2.warpPerspective(
        np.asarray(image),
        np.asarray(H, dtype=np.float64),
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    return np.ascontiguousarray(warped)


def warp_mask_with_homography(mask, H, out_size) -> np.ndarray:
    warped = warp_with_homography(mask, H, out_size, interpolation=cv2.INTER_NEAREST, border_value=0)
    return np.ascontiguousarray((np.asarray(warped) > 0.5).astype(np.float32))


def warp_band_to_rgb_plane(source_image: np.ndarray,
                           homography: np.ndarray,
                           target_size: Tuple[int, int],
                           interpolation=cv2.INTER_LINEAR,
                           border_value: float = 0.0) -> np.ndarray:
    return warp_with_homography(source_image, homography, target_size, interpolation=interpolation, border_value=border_value)


def build_validity_mask_from_warp(source_shape: Sequence[int],
                                  homography: np.ndarray,
                                  target_size: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    ones = np.ones((src_h, src_w), dtype=np.float32)
    return warp_mask_with_homography(ones, homography, target_size)


def params_to_affine_residual(theta) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.size != 6:
        raise ValueError(f"Affine residual expects 6 params, got {theta.size}")
    tx, ty, log_sx, log_sy, rot, shear = theta.tolist()
    sx = float(np.exp(log_sx))
    sy = float(np.exp(log_sy))
    c = float(np.cos(rot))
    s = float(np.sin(rot))
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    Sh = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    A = R @ Sh @ S
    delta = np.array(
        [
            [A[0, 0], A[0, 1], tx],
            [A[1, 0], A[1, 1], ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_homography(delta)


def compose_global_transform(H0, theta, mode: str = "affine_residual_over_h0") -> np.ndarray:
    H0 = normalize_homography(H0)
    if mode == "affine_residual_over_h0":
        delta = params_to_affine_residual(theta)
        return normalize_homography(delta @ H0)
    if mode == "projective_residual_over_h0":
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != 8:
            raise ValueError(f"Projective residual expects 8 params, got {theta.size}")
        delta = np.array(
            [
                [1.0 + theta[0], theta[1], theta[2]],
                [theta[3], 1.0 + theta[4], theta[5]],
                [theta[6], theta[7], 1.0],
            ],
            dtype=np.float64,
        )
        return normalize_homography(delta @ H0)
    raise ValueError(f"Unsupported global transform mode: {mode}")


def _masked_arrays(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(mask, dtype=np.float32) > 0.5
    return np.asarray(a, dtype=np.float32)[valid], np.asarray(b, dtype=np.float32)[valid]


def score_edge_overlap_f1(rgb_edges, band_edges_warped, mask, dilate_radius: int = 1) -> float:
    ref = (np.asarray(rgb_edges) > 0).astype(np.uint8)
    cand = (np.asarray(band_edges_warped) > 0).astype(np.uint8)
    valid = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
    ref = ref * valid
    cand = cand * valid
    if dilate_radius > 0:
        kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=np.uint8)
        ref_d = cv2.dilate(ref, kernel, iterations=1)
        cand_d = cv2.dilate(cand, kernel, iterations=1)
    else:
        ref_d = ref
        cand_d = cand
    ref_count = int(ref.sum())
    cand_count = int(cand.sum())
    if ref_count == 0 and cand_count == 0:
        return 1.0
    if ref_count == 0 or cand_count == 0:
        return 0.0
    precision = float((cand * ref_d).sum()) / float(max(cand_count, 1))
    recall = float((ref * cand_d).sum()) / float(max(ref_count, 1))
    denom = precision + recall
    if denom <= 1e-8:
        return 0.0
    return 2.0 * precision * recall / denom


def score_gradient_ncc(rgb_grad, band_grad_warped, mask, eps: float = 1e-6) -> float:
    ref_vals, cand_vals = _masked_arrays(rgb_grad, band_grad_warped, mask)
    if ref_vals.size == 0 or cand_vals.size == 0:
        return float("-inf")
    ref_centered = ref_vals - ref_vals.mean()
    cand_centered = cand_vals - cand_vals.mean()
    denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered) + eps)
    return float(np.sum(ref_centered * cand_centered) / denom)


def compute_alignment_score(reference_image, candidate_image, validity_mask=None, eps: float = 1e-8) -> float:
    mask = np.ones_like(reference_image, dtype=np.float32) if validity_mask is None else validity_mask
    return score_gradient_ncc(reference_image, candidate_image, mask, eps=eps)


def joint_objective(theta, H0, frame_batch, config) -> float:
    transform = compose_global_transform(H0, theta, mode=str(config.get("rectification_global_mode", "affine_residual_over_h0")))
    evaluation = evaluate_transform_on_frames(
        transform,
        frame_batch,
        config,
        baseline_mode=str(config.get("baseline_mode", "naive_resize")),
    )
    w_edge = float(config.get("w_edge", 0.5))
    w_grad = float(config.get("w_grad", 0.5))
    reg = float(config.get("rectification_residual_reg", 1e-3)) * float(np.dot(theta, theta))
    # Align the optimizer target with QA gate semantics:
    # we care about improvement over naive resize, not only absolute score.
    score = w_edge * evaluation["delta_edge_f1"] + w_grad * evaluation["delta_grad_ncc"] - reg
    return -score


def _theta_dim_for_mode(mode: str) -> int:
    return 8 if str(mode) == "projective_residual_over_h0" else 6


def _default_step_floor(mode: str) -> np.ndarray:
    if str(mode) == "projective_residual_over_h0":
        return np.array([1e-3, 1e-3, 0.25, 1e-3, 1e-3, 0.25, 1e-6, 1e-6], dtype=np.float64)
    return np.array([0.25, 0.25, 1e-3, 1e-3, np.deg2rad(0.05), 5e-4], dtype=np.float64)


def _default_search_steps(frame_batch: Sequence[dict], config: dict) -> np.ndarray:
    sample = frame_batch[0]
    out_w, out_h = sample["target_size"]
    max_dim = float(max(out_w, out_h))
    mode = str(config.get("rectification_global_mode", "affine_residual_over_h0"))
    if mode == "projective_residual_over_h0":
        return np.array([
            0.01,                 # h11 residual
            0.01,                 # h12 residual
            0.008 * max_dim,      # tx
            0.01,                 # h21 residual
            0.01,                 # h22 residual
            0.008 * max_dim,      # ty
            1e-5,                 # h31
            1e-5,                 # h32
        ], dtype=np.float64)
    return np.array([
        0.008 * max_dim,
        0.008 * max_dim,
        0.02,
        0.02,
        np.deg2rad(0.8),
        0.01,
    ], dtype=np.float64)


def optimize_global_transform_opencv_search(H0, frame_batch, config) -> dict:
    mode = str(config.get("rectification_global_mode", "affine_residual_over_h0"))
    theta_dim = _theta_dim_for_mode(mode)
    restarts = int(config.get("rectification_search_restarts", 5))
    max_rounds = int(config.get("rectification_search_steps", 50))
    rng = np.random.default_rng(int(config.get("seed", 0)))
    step_floor = _default_step_floor(mode)
    base_steps = _default_search_steps(frame_batch, config)
    if base_steps.size != theta_dim:
        raise RuntimeError(f"Search step dimensionality mismatch for mode={mode}: steps={base_steps.size}, theta_dim={theta_dim}")
    best_theta = np.zeros(theta_dim, dtype=np.float64)
    best_score = -joint_objective(best_theta, H0, frame_batch, config)
    best_history = []

    for restart in range(max(restarts, 1)):
        theta = np.zeros(theta_dim, dtype=np.float64)
        if restart > 0:
            theta += rng.normal(0.0, 1.0, size=theta_dim) * (0.35 * base_steps)
        steps = base_steps.copy()
        history = []
        current_score = -joint_objective(theta, H0, frame_batch, config)

        for _ in range(max_rounds):
            improved = False
            for idx in range(theta.size):
                local_best_theta = theta
                local_best_score = current_score
                for direction in (-1.0, 1.0):
                    candidate = theta.copy()
                    candidate[idx] += direction * steps[idx]
                    candidate_score = -joint_objective(candidate, H0, frame_batch, config)
                    if candidate_score > local_best_score:
                        local_best_theta = candidate
                        local_best_score = candidate_score
                theta = local_best_theta
                if local_best_score > current_score:
                    current_score = local_best_score
                    improved = True
            history.append(float(current_score))
            if not improved:
                steps *= 0.5
                if np.all(steps <= step_floor):
                    break

        if current_score > best_score:
            best_score = current_score
            best_theta = theta.copy()
            best_history = history[:]

    best_transform = compose_global_transform(H0, best_theta, mode=str(config.get("rectification_global_mode", "affine_residual_over_h0")))
    return {
        "best_theta": best_theta.tolist(),
        "best_T": best_transform.tolist(),
        "best_score": float(best_score),
        "history": best_history,
        "backend": "opencv_search",
        "num_restarts": restarts,
        "num_iterations": max_rounds,
    }


def optimize_global_transform_scipy(H0, frame_batch, config) -> dict:
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise RuntimeError("SciPy backend requested but scipy is unavailable.") from exc

    mode = str(config.get("rectification_global_mode", "affine_residual_over_h0"))
    theta0 = np.zeros(_theta_dim_for_mode(mode), dtype=np.float64)
    result = minimize(
        lambda t: joint_objective(t, H0, frame_batch, config),
        theta0,
        method="Powell",
        options={"maxiter": int(config.get("rectification_search_steps", 50)), "disp": False},
    )
    best_theta = np.asarray(result.x, dtype=np.float64)
    best_transform = compose_global_transform(H0, best_theta, mode=str(config.get("rectification_global_mode", "affine_residual_over_h0")))
    return {
        "best_theta": best_theta.tolist(),
        "best_T": best_transform.tolist(),
        "best_score": float(-result.fun),
        "history": [],
        "backend": "scipy_minimize",
        "num_restarts": 1,
        "num_iterations": int(config.get("rectification_search_steps", 50)),
    }


def evaluate_transform_on_frames(T, frame_batch, config, baseline_mode: str = "naive_resize") -> dict:
    T = normalize_homography(T)
    edge_dilate_radius = int(config.get("rectification_edge_dilate_radius", 1))
    per_frame = []
    baseline_edge = []
    baseline_grad = []
    rect_edge = []
    rect_grad = []
    improved_edge = 0
    improved_grad = 0
    improved_either = 0
    severe_outliers = 0
    tau_edge = float(config.get("severe_misalignment_tau_edge", 0.01))
    tau_grad = float(config.get("severe_misalignment_tau_grad", 0.01))

    for frame in frame_batch:
        out_size = frame["target_size"]
        rect_mask = warp_mask_with_homography(frame["ones"], T, out_size)
        rect_edges = warp_with_homography(frame["band_edges"], T, out_size, interpolation=cv2.INTER_NEAREST, border_value=0)
        rect_gradients = warp_with_homography(frame["band_grad"], T, out_size, interpolation=cv2.INTER_LINEAR, border_value=0.0)

        baseline_edges = cv2.resize(frame["band_edges"], out_size, interpolation=cv2.INTER_NEAREST)
        baseline_gradients = cv2.resize(frame["band_grad"], out_size, interpolation=cv2.INTER_LINEAR)

        edge_baseline = score_edge_overlap_f1(frame["rgb_edges"], baseline_edges, rect_mask, dilate_radius=edge_dilate_radius)
        grad_baseline = score_gradient_ncc(frame["rgb_grad"], baseline_gradients, rect_mask)
        edge_rectified = score_edge_overlap_f1(frame["rgb_edges"], rect_edges, rect_mask, dilate_radius=edge_dilate_radius)
        grad_rectified = score_gradient_ncc(frame["rgb_grad"], rect_gradients, rect_mask)

        delta_edge = float(edge_rectified - edge_baseline)
        delta_grad = float(grad_rectified - grad_baseline)
        validity_ratio = float(np.mean(rect_mask))
        if delta_edge > 0:
            improved_edge += 1
        if delta_grad > 0:
            improved_grad += 1
        if delta_edge > 0 or delta_grad > 0:
            improved_either += 1
        if delta_edge < -tau_edge and delta_grad < -tau_grad:
            severe_outliers += 1

        per_frame.append(
            {
                "frame_id": frame["frame_id"],
                "image_name": frame["image_name"],
                "baseline_edge_f1": float(edge_baseline),
                "rectified_edge_f1": float(edge_rectified),
                "baseline_grad_ncc": float(grad_baseline),
                "rectified_grad_ncc": float(grad_rectified),
                "delta_edge_f1": delta_edge,
                "delta_grad_ncc": delta_grad,
                "validity_ratio": validity_ratio,
                "severe_misalignment": bool(delta_edge < -tau_edge and delta_grad < -tau_grad),
            }
        )
        baseline_edge.append(float(edge_baseline))
        baseline_grad.append(float(grad_baseline))
        rect_edge.append(float(edge_rectified))
        rect_grad.append(float(grad_rectified))

    num_frames = len(per_frame)
    summary = {
        "baseline_mean_edge_f1": float(np.mean(baseline_edge)) if baseline_edge else float("nan"),
        "baseline_mean_grad_ncc": float(np.mean(baseline_grad)) if baseline_grad else float("nan"),
        "rectified_mean_edge_f1": float(np.mean(rect_edge)) if rect_edge else float("nan"),
        "rectified_mean_grad_ncc": float(np.mean(rect_grad)) if rect_grad else float("nan"),
        "delta_edge_f1": (float(np.mean(rect_edge)) - float(np.mean(baseline_edge))) if baseline_edge else float("nan"),
        "delta_grad_ncc": (float(np.mean(rect_grad)) - float(np.mean(baseline_grad))) if baseline_grad else float("nan"),
        "num_frames": num_frames,
        "num_frames_improved_edge": improved_edge,
        "num_frames_improved_grad": improved_grad,
        "num_frames_improved_either": improved_either,
        "severe_outlier_count": severe_outliers,
        "representative_frame_ids": [str(item.get("frame_id", "")) for item in per_frame],
        "representative_image_names": [str(item.get("image_name", "")) for item in per_frame],
        "per_frame": per_frame,
    }
    return summary


def determine_pass_from_summary(summary: dict,
                                min_improved_ratio: float = 0.6,
                                max_severe_outliers: int = 0) -> bool:
    num_frames = int(summary.get("num_frames", 0))
    if num_frames <= 0:
        return False
    delta_edge = float(summary.get("delta_edge_f1", float("-inf")))
    delta_grad = float(summary.get("delta_grad_ncc", float("-inf")))
    improved_ratio = float(summary.get("num_frames_improved_either", 0)) / float(max(num_frames, 1))
    severe_outliers = int(summary.get("severe_outlier_count", 0))
    return (
        delta_edge > 0.0
        and delta_grad > 0.0
        and improved_ratio >= float(min_improved_ratio)
        and severe_outliers <= int(max_severe_outliers)
    )


def _summary_stat(summary: dict, key: str, default: float) -> float:
    value = summary.get(key, default) if isinstance(summary, dict) else default
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_band_set(value, default: str = "NIR") -> set[str]:
    if value is None:
        value = default
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    return {str(item).strip().upper() for item in items if str(item).strip()}


def determine_qa_status_from_summary(
    summary: dict,
    *,
    legacy_pass: Optional[bool] = None,
    band_name: str = "",
    h0_source: str = "",
    match_summary: Optional[dict] = None,
    homography_stability: Optional[dict] = None,
    config: Optional[dict] = None,
    min_improved_ratio: float = 0.6,
    max_severe_outliers: int = 0,
) -> dict:
    """Baseline-aware rectification QA.

    The legacy hard gate remains a strict pass path. A weak H0 baseline can only
    be downgraded to pass_with_warning when matching quality, H stability,
    baseline-relative QA outliers, and gradient improvement are all acceptable
    for a configured high-modality band.
    """
    cfg = dict(config or {})
    match_summary = match_summary or {}
    homography_stability = homography_stability or {}
    if legacy_pass is None:
        legacy_pass = determine_pass_from_summary(
            summary,
            min_improved_ratio=min_improved_ratio,
            max_severe_outliers=max_severe_outliers,
        )

    h0_source_norm = str(h0_source or "").strip().lower()
    baseline_is_weak = h0_source_norm == "naive_fallback"
    band_upper = str(band_name or "").strip().upper()
    high_modality_bands = _parse_band_set(cfg.get("rectification_high_modality_bands", "NIR"))
    high_modality_gap = band_upper in high_modality_bands

    delta_edge = _summary_stat(summary, "delta_edge_f1", float("nan"))
    delta_grad = _summary_stat(summary, "delta_grad_ncc", float("nan"))
    num_frames = int(summary.get("num_frames", 0) or 0)
    improved_grad_ratio = float(summary.get("num_frames_improved_grad", 0) or 0) / float(max(num_frames, 1))
    severe_outliers = int(summary.get("severe_outlier_count", 0) or 0)
    severe_outlier_ratio = float(severe_outliers) / float(max(num_frames, 1))
    max_severe = int(cfg.get("rectification_max_severe_outliers", max_severe_outliers))
    min_warning_accepted_ratio = float(cfg.get("rectification_warning_min_accepted_ratio", 0.80))
    warning_min_delta_edge = float(cfg.get("rectification_warning_min_delta_edge_f1", -0.05))
    warning_min_grad_improved_ratio = float(cfg.get("rectification_warning_min_grad_improved_ratio", 0.60))
    warning_max_severe_outlier_ratio = float(cfg.get("rectification_warning_max_severe_outlier_ratio", 0.10))
    warning_max_severe_outlier_count = int(cfg.get("rectification_warning_max_severe_outlier_count", 1))

    accepted_stats = match_summary.get("accepted", {}) if isinstance(match_summary, dict) else {}
    accepted_ratio = _summary_stat(match_summary, "accepted_ratio", 0.0)
    mean_inlier = _summary_stat(accepted_stats.get("inlier_ratio", {}), "mean", 0.0)
    mean_reproj = _summary_stat(accepted_stats.get("reproj_error", {}), "mean", float("inf"))
    mean_coverage = _summary_stat(accepted_stats.get("coverage", {}), "mean", 0.0)

    median_disp = _summary_stat(
        homography_stability.get("disp_vs_aggregate_grid_mean_px", {}),
        "median",
        float("inf"),
    )
    robust_reject_ratio = _summary_stat(homography_stability, "robust_reject_ratio", 1.0)
    stability_warn_px = float(cfg.get("rectification_stability_warn_mean_px", 25.0))
    max_reject_ratio = float(cfg.get("rectification_stability_max_reject_ratio", 0.25))

    min_inlier = float(cfg.get("minima_min_inlier_ratio", 0.30))
    max_reproj = float(cfg.get("minima_max_reproj_error", 4.0))
    min_coverage = float(cfg.get("minima_min_coverage", 0.25))

    warnings: List[str] = []
    if baseline_is_weak:
        warnings.append("weak_h0_baseline")
    if high_modality_gap:
        warnings.append("high_modality_gap")
    if not (np.isfinite(delta_edge) and delta_edge > 0.0):
        warnings.append("edge_not_improved_against_h0")
    if not (np.isfinite(delta_edge) and delta_edge >= warning_min_delta_edge):
        warnings.append("edge_below_warning_floor")
    if not (np.isfinite(delta_grad) and delta_grad > 0.0):
        warnings.append("grad_not_improved_against_h0")
    if improved_grad_ratio < warning_min_grad_improved_ratio:
        warnings.append("low_grad_improved_ratio")
    if severe_outliers > max_severe:
        warnings.append("baseline_relative_qa_outliers")
    if not match_summary:
        warnings.append("missing_match_summary")
    if accepted_ratio < min_warning_accepted_ratio:
        warnings.append("low_accepted_ratio")
    if mean_inlier < min_inlier:
        warnings.append("low_mean_inlier_ratio")
    if mean_reproj > max_reproj:
        warnings.append("high_mean_reproj_error")
    if mean_coverage < min_coverage:
        warnings.append("low_mean_coverage")
    if not homography_stability:
        warnings.append("missing_homography_stability")
    if median_disp > stability_warn_px:
        warnings.append("homography_dispersion_warning")
    if robust_reject_ratio > max_reject_ratio:
        warnings.append("high_robust_reject_ratio")

    match_ok = (
        bool(match_summary)
        and accepted_ratio >= min_warning_accepted_ratio
        and mean_inlier >= min_inlier
        and mean_reproj <= max_reproj
        and mean_coverage >= min_coverage
    )
    stability_ok = (
        bool(homography_stability)
        and median_disp <= stability_warn_px
        and robust_reject_ratio <= max_reject_ratio
    )
    qa_outlier_ok = (
        severe_outliers <= warning_max_severe_outlier_count
        and severe_outlier_ratio <= warning_max_severe_outlier_ratio
    )
    if not qa_outlier_ok:
        warnings.append("qa_outliers_exceed_warning_limit")
    geometry_ok = stability_ok
    grad_ok = np.isfinite(delta_grad) and delta_grad > 0.0
    edge_floor_ok = np.isfinite(delta_edge) and delta_edge >= warning_min_delta_edge
    improved_grad_ratio_ok = improved_grad_ratio >= warning_min_grad_improved_ratio
    warning_path_ok = (
        baseline_is_weak
        and high_modality_gap
        and match_ok
        and stability_ok
        and qa_outlier_ok
        and grad_ok
        and edge_floor_ok
        and improved_grad_ratio_ok
    )

    if legacy_pass:
        status = "pass"
        warning_path_reason = ""
    elif warning_path_ok:
        status = "pass_with_warning"
        warning_path_reason = "weak baseline + high modality gap + stable H + acceptable QA outliers + grad improvement"
        warnings.append("legacy_hard_gate_failed")
    else:
        status = "fail"
        warning_path_reason = ""
        if baseline_is_weak and not high_modality_gap:
            warnings.append("weak_h0_without_high_modality_policy")

    return {
        "qa_status": status,
        "qa_pass": status in {"pass", "pass_with_warning"},
        "qa_warning_codes": sorted(set(warnings)),
        "legacy_pass": bool(legacy_pass),
        "warning_path_reason": warning_path_reason,
        "decision_inputs": {
            "h0_source": str(h0_source or "unknown"),
            "baseline_is_weak": bool(baseline_is_weak),
            "high_modality_gap": bool(high_modality_gap),
            "match_ok": bool(match_ok),
            "geometry_ok": bool(geometry_ok),
            "stability_ok": bool(stability_ok),
            "qa_outlier_ok": bool(qa_outlier_ok),
            "grad_ok": bool(grad_ok),
            "edge_floor_ok": bool(edge_floor_ok),
            "improved_grad_ratio_ok": bool(improved_grad_ratio_ok),
            "warning_path_ok": bool(warning_path_ok),
            "accepted_ratio": float(accepted_ratio),
            "mean_inlier_ratio": float(mean_inlier),
            "mean_reproj_error": float(mean_reproj),
            "mean_coverage": float(mean_coverage),
            "median_disp_to_aggregate_mean_px": float(median_disp),
            "median_disp_to_aggregate_mean_rel": _summary_stat(
                homography_stability.get("disp_vs_aggregate_grid_mean_rel", {}),
                "median",
                float("inf"),
            ),
            "robust_reject_ratio": float(robust_reject_ratio),
            "severe_outlier_count": int(severe_outliers),
            "severe_outlier_ratio": float(severe_outlier_ratio),
            "delta_edge_f1": float(delta_edge),
            "delta_grad_ncc": float(delta_grad),
            "improved_grad_ratio": float(improved_grad_ratio),
            "warning_min_delta_edge_f1": float(warning_min_delta_edge),
            "warning_min_grad_improved_ratio": float(warning_min_grad_improved_ratio),
            "warning_max_severe_outlier_ratio": float(warning_max_severe_outlier_ratio),
            "warning_max_severe_outlier_count": int(warning_max_severe_outlier_count),
            "min_warning_accepted_ratio": float(min_warning_accepted_ratio),
            "stability_warn_mean_px": float(stability_warn_px),
            "max_robust_reject_ratio": float(max_reject_ratio),
        },
        "notes": {
            "policy": "baseline_aware_v2",
            "pass_bool_compatibility": "pass follows qa_status; legacy_pass preserves the old edge+grad hard gate.",
            "h0_source": str(h0_source or "unknown"),
            "warning_path_reason": warning_path_reason,
            "baseline_is_weak": bool(baseline_is_weak),
            "high_modality_gap": bool(high_modality_gap),
            "high_modality_bands": sorted(high_modality_bands),
            "min_warning_accepted_ratio": float(min_warning_accepted_ratio),
            "warning_min_delta_edge_f1": float(warning_min_delta_edge),
            "warning_min_grad_improved_ratio": float(warning_min_grad_improved_ratio),
            "warning_max_severe_outlier_ratio": float(warning_max_severe_outlier_ratio),
            "warning_max_severe_outlier_count": int(warning_max_severe_outlier_count),
            "stability_warn_mean_px": float(stability_warn_px),
            "max_robust_reject_ratio": float(max_reject_ratio),
        },
    }


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return np.ascontiguousarray((arr * 255.0).astype(np.uint8))


def overlay_red_green(reference_image: np.ndarray, candidate_image: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_image, dtype=np.float32)
    cand = np.asarray(candidate_image, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"Overlay expects same-sized arrays, got {ref.shape} vs {cand.shape}")
    zeros = np.zeros_like(ref)
    stacked = np.stack([np.clip(ref, 0.0, 1.0), np.clip(cand, 0.0, 1.0), zeros], axis=-1)
    return np.ascontiguousarray((stacked * 255.0).astype(np.uint8))


def export_rectification_debug_panel(rgb_img, raw_band_img, naive_resized, rectified_img, mask, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = _to_uint8(rgb_img)
    naive_u8 = _to_uint8(naive_resized)
    rect_u8 = _to_uint8(rectified_img)
    mask_u8 = _to_uint8(mask)
    overlay_naive = overlay_red_green(rgb_img, naive_resized)
    overlay_rect = overlay_red_green(rgb_img, rectified_img)
    row1 = np.hstack([cv2.cvtColor(rgb_u8, cv2.COLOR_GRAY2BGR), cv2.cvtColor(naive_u8, cv2.COLOR_GRAY2BGR), cv2.cvtColor(rect_u8, cv2.COLOR_GRAY2BGR)])
    row2 = np.hstack([overlay_naive, overlay_rect, cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)])
    panel = np.vstack([row1, row2])
    Image.fromarray(panel).save(out_path)


def write_rectification_diagnostics_json(diag, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")


def build_scale_adapters(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    alignment_scale: float,
    alignment_max_dim: int = 0,
) -> Dict[str, np.ndarray]:
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    tgt_h, tgt_w = int(target_shape[0]), int(target_shape[1])
    scale = float(alignment_scale)
    max_dim = int(alignment_max_dim)
    if max_dim > 0:
        scale = min(scale, float(max_dim) / float(max(src_h, src_w, tgt_h, tgt_w, 1)))
    scaled_src_w = max(int(round(src_w * scale)), 8)
    scaled_src_h = max(int(round(src_h * scale)), 8)
    scaled_tgt_w = max(int(round(tgt_w * scale)), 8)
    scaled_tgt_h = max(int(round(tgt_h * scale)), 8)
    src_to_align = _scale_matrix(scaled_src_w / float(max(src_w, 1)), scaled_src_h / float(max(src_h, 1)))
    tgt_to_align = _scale_matrix(scaled_tgt_w / float(max(tgt_w, 1)), scaled_tgt_h / float(max(tgt_h, 1)))
    align_to_src = np.linalg.inv(src_to_align)
    align_to_tgt = np.linalg.inv(tgt_to_align)
    return {
        "source_to_align": src_to_align,
        "target_to_align": tgt_to_align,
        "align_to_source": align_to_src,
        "align_to_target": align_to_tgt,
        "source_shape_align": (scaled_src_h, scaled_src_w),
        "target_shape_align": (scaled_tgt_h, scaled_tgt_w),
        "alignment_scale": scale,
        "alignment_max_dim": max_dim,
    }


def scale_transform_to_alignment(H_full: np.ndarray, scale_adapters: dict) -> np.ndarray:
    return normalize_homography(scale_adapters["target_to_align"] @ H_full @ scale_adapters["align_to_source"])


def scale_transform_from_alignment(H_align: np.ndarray, scale_adapters: dict) -> np.ndarray:
    return normalize_homography(scale_adapters["align_to_target"] @ H_align @ scale_adapters["source_to_align"])


def prepare_frame_batch(frame_records: Sequence[dict],
                        input_dynamic_range: str,
                        radiometric_mode: str,
                        alignment_scale: float,
                        alignment_max_dim: int = 0) -> List[dict]:
    frame_batch: List[dict] = []
    for record in frame_records:
        rgb_full = load_rgb_plane_image(record["rgb_path"])
        band_full = load_raw_band_image(
            record["band_path"],
            dynamic_range=input_dynamic_range,
            radiometric_mode=radiometric_mode,
        )
        scale_adapters = build_scale_adapters(
            band_full.shape,
            rgb_full.shape,
            alignment_scale=alignment_scale,
            alignment_max_dim=alignment_max_dim,
        )
        rgb_align = cv2.resize(
            rgb_full,
            (int(scale_adapters["target_shape_align"][1]), int(scale_adapters["target_shape_align"][0])),
            interpolation=cv2.INTER_AREA,
        )
        band_align = cv2.resize(
            band_full,
            (int(scale_adapters["source_shape_align"][1]), int(scale_adapters["source_shape_align"][0])),
            interpolation=cv2.INTER_AREA,
        )
        prepared = prepare_alignment_images(rgb_align, band_align)
        frame_batch.append(
            {
                "frame_id": record["frame_id"],
                "image_name": record["image_name"],
                "rgb_path": record["rgb_path"],
                "band_path": record["band_path"],
                "rgb_meta": record.get("rgb_meta", {}),
                "band_meta": record.get("band_meta", {}),
                "target_size": (prepared["rgb_gray"].shape[1], prepared["rgb_gray"].shape[0]),
                "target_size_full": (rgb_full.shape[1], rgb_full.shape[0]),
                "source_shape_full": band_full.shape,
                "rgb_gray": prepared["rgb_gray"],
                "band_gray": prepared["band_gray"],
                "rgb_grad": prepared["rgb_grad"],
                "band_grad": prepared["band_grad"],
                "rgb_edges": prepared["rgb_edges"],
                "band_edges": prepared["band_edges"],
                "ones": np.ones(prepared["band_gray"].shape, dtype=np.float32),
                "scale_adapters": scale_adapters,
                "rgb_full": rgb_full,
                "band_full": band_full,
                "structure_score": float(record.get("structure_score", compute_structure_score(rgb_full))),
            }
        )
    return frame_batch


def save_scalar_tiff_with_sidecar(path: str | Path, image_array: np.ndarray, metadata: Optional[dict] = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image_array)
    Image.fromarray(arr).save(out_path, format="TIFF")
    if metadata is not None:
        sidecar_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        sidecar_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def estimate_homography_ecc(template_image: np.ndarray,
                            input_image: np.ndarray,
                            initial_h: Optional[np.ndarray] = None,
                            iterations: int = 200,
                            termination_eps: float = 1e-6,
                            gauss_filter_size: int = 5) -> Tuple[np.ndarray, float, bool]:
    template = np.ascontiguousarray(np.asarray(template_image, dtype=np.float32))
    moving = np.ascontiguousarray(np.asarray(input_image, dtype=np.float32))
    warp_matrix = np.eye(3, dtype=np.float32) if initial_h is None else np.asarray(initial_h, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(iterations),
        float(termination_eps),
    )
    try:
        cc, warp = cv2.findTransformECC(
            template,
            moving,
            warp_matrix,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,
            gauss_filter_size,
        )
        return normalize_homography(warp.astype(np.float64)), float(cc), True
    except cv2.error:
        return normalize_homography(warp_matrix.astype(np.float64)), float("nan"), False


def robust_average_homographies(homographies: Iterable[np.ndarray]) -> np.ndarray:
    matrices: List[np.ndarray] = [normalize_homography(h) for h in homographies]
    if not matrices:
        raise ValueError("robust_average_homographies received no matrices.")
    stack = np.stack(matrices, axis=0)
    return normalize_homography(np.median(stack, axis=0))
