"""Explicit CPU camera/normalization algebra; no fitting and no metric computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label}")
    return result


@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int
                or min(self.width, self.height) <= 0
                or not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all()
                or min(self.fx, self.fy) <= 0):
            raise ValueError("Invalid PINHOLE intrinsics")

    def project(self, rays):
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 2 or not np.isfinite(rays).all():
            raise ValueError("Expected finite normalized xy rays")
        with np.errstate(over="ignore", invalid="ignore"):
            pixels = rays * [self.fx, self.fy] + [self.cx, self.cy]
        if not np.isfinite(pixels).all():
            raise ValueError("Nonfinite projected pixels")
        return pixels

    def unproject(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Expected finite pixel coordinates")
        with np.errstate(over="ignore", invalid="ignore"):
            rays = (pixels - [self.cx, self.cy]) / [self.fx, self.fy]
        if not np.isfinite(rays).all():
            raise ValueError("Nonfinite recovered rays")
        return rays


def projection_to_pinhole(projection_row, width, height, *, convention):
    """Convert a row-vector projection with explicit NDC->pixel convention.

    Graphdeco ndc2Pix is ((ndc + 1) * size - 1) / 2. The old mesh proxy
    uses (ndc + 1) * size / 2. These are different coordinate systems.
    """
    if convention not in {"graphdeco_ndc_index_centers_v1", "corner_origin_centers_plus_half_v1"}:
        raise ValueError("Unknown projection pixel convention")
    p = finite_array(projection_row, (4, 4), "row-vector projection")
    if (not np.array_equal(p[:, 3], [0, 0, 1, 0])
            or p[1, 0] != 0 or p[0, 1] != 0 or p[3, 0] != 0 or p[3, 1] != 0):
        raise ValueError("Unsupported skew/oblique/non-PINHOLE projection")
    offset = -0.5 if convention == "graphdeco_ndc_index_centers_v1" else 0.0
    return Pinhole(width, height, p[0, 0] * width / 2, p[1, 1] * height / 2,
                   (p[2, 0] + 1) * width / 2 + offset,
                   (p[2, 1] + 1) * height / 2 + offset)


def map_rays(source, target, source_pixels, *, coordinate_atol=1e-12, angle_atol=1e-7):
    """Pure intrinsic mapping. Caller must independently prove image/pose identity."""
    if min(coordinate_atol, angle_atol) <= 0 or not np.isfinite([coordinate_atol, angle_atol]).all():
        raise ValueError("Invalid ray tolerances")
    rays = source.unproject(source_pixels)
    target_pixels = target.project(rays)
    recovered = target.unproject(target_pixels)
    a = np.column_stack([rays, np.ones(len(rays))])
    b = np.column_stack([recovered, np.ones(len(rays))])
    with np.errstate(over="ignore", invalid="ignore"):
        angles = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))
    errors = np.max(np.abs(recovered - rays), axis=1)
    if (not np.isfinite(errors).all() or not np.isfinite(angles).all()
            or np.any(errors > coordinate_atol) or np.any(angles > angle_atol)):
        raise ValueError("Normalized ray equivalence failed")
    bounds = ((target_pixels >= 0).all(axis=1)
              & (target_pixels < [target.width, target.height]).all(axis=1))
    return {"pixels": target_pixels, "rays": recovered, "coordinate_error": errors,
            "angular_error_rad": angles, "in_bounds": bounds}


def validate_pose(w2c, *, rigidity_atol):
    m = finite_array(w2c, (4, 4), "world-to-camera pose")
    if not np.array_equal(m[3], [0, 0, 0, 1]):
        raise ValueError("Non-affine pose")
    if not 0 < rigidity_atol <= 1e-6:
        raise ValueError("Invalid rigidity tolerance")
    r = m[:3, :3]
    if (np.max(np.abs(r @ r.T - np.eye(3))) > rigidity_atol
            or abs(np.linalg.det(r) - 1) > rigidity_atol):
        raise ValueError("Pose contains scale/reflection/shear")
    return m


def compare_poses(first, second, *, center_atol, angle_atol, rigidity_atol=1e-10):
    a = validate_pose(first, rigidity_atol=rigidity_atol)
    b = validate_pose(second, rigidity_atol=rigidity_atol)
    if not np.isfinite([center_atol, angle_atol]).all() or min(center_atol, angle_atol) <= 0:
        raise ValueError("Invalid pose comparison tolerances")
    ca = -np.linalg.solve(a[:3, :3], a[:3, 3])
    cb = -np.linalg.solve(b[:3, :3], b[:3, 3])
    delta = a[:3, :3] @ b[:3, :3].T
    # Exact equal arrays are equal even when float serialization perturbs trace.
    angle = 0.0 if np.array_equal(a[:3, :3], b[:3, :3]) else float(
        np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
    center = float(np.linalg.norm(ca - cb))
    if center > center_atol or angle > angle_atol:
        raise ValueError(f"Non-equivalent poses: center={center}, angle={angle}")
    return {"center_difference": center, "angular_difference_rad": angle}


def inverse_dataparser_points(points, transform, scale, *, rigidity_atol=1e-7):
    """Undo recorded X_model = s * (R * X_source + t); never fit R/t/s."""
    transform = validate_pose(transform, rigidity_atol=rigidity_atol)
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid model points or normalization scale")
    with np.errstate(over="ignore", invalid="ignore"):
        recovered = np.linalg.solve(transform[:3, :3], (points / scale - transform[:3, 3]).T).T
    if not np.isfinite(recovered).all():
        raise ValueError("Nonfinite inverse-normalized points")
    return recovered


def source_unit_accumulators(accumulators, scale):
    """Pure-array unit conversion; inverse-depth H cannot be inferred from M1."""
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Require positive finite scalar scale")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        inverse_scale = np.float64(1) / scale
        factors = {"accumulated_alpha": 1.0, "weighted_camera_z_sum": inverse_scale,
                   "weighted_camera_z_second_moment": inverse_scale * inverse_scale,
                   "weighted_inverse_camera_z_sum": scale}
    if (not all(np.isfinite(f) and f > 0 for f in factors.values())
            or set(accumulators) != set(factors)):
        raise ValueError("Require positive scale and all four raw accumulators")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in accumulators.items()}
    if len({value.shape for value in arrays.values()}) != 1 or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Accumulator shape/nonfinite mismatch")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = {key: arrays[key] * factor for key, factor in factors.items()}
    if not all(np.isfinite(v).all() for v in converted.values()):
        raise ValueError("Nonfinite unit-converted accumulators")
    return converted
