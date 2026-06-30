#!/usr/bin/env python3
"""CPU-only DA3 / MapAnything / MVS tri-proxy comparison.

This script consumes existing Papaya target-0049 proxy outputs and produces a
review package. It does not run inference, train, rerender UMGS, or modify any
manuscript/benchmark files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr, spearmanr


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--da3_mvs_npz", required=True, type=Path)
    p.add_argument("--mapanything_target_npz", required=True, type=Path)
    p.add_argument("--robust_scale_csv", required=True, type=Path)
    p.add_argument("--mapanything_manifest", required=True, type=Path)
    p.add_argument("--source_only_manifest", required=True, type=Path)
    p.add_argument("--target_source_consistency_csv", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--package_name", default="TRI_PROXY_PAPAYA_0049_COMPARISON_REVIEW_PACKAGE_20260623.zip")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def resize_float(arr: np.ndarray, shape: tuple[int, int], resample: int = Image.Resampling.BILINEAR) -> np.ndarray:
    if arr.shape == shape:
        return arr.astype(np.float64, copy=False)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = img.resize((shape[1], shape[0]), resample=resample)
    return np.asarray(resized, dtype=np.float64)


def resize_masked_float(
    arr: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
    *,
    min_weight: float = 0.5,
    resample: int = Image.Resampling.BILINEAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize a masked raster without bleeding invalid values across edges.

    The value resize follows normalized convolution:

        resize(arr * mask) / resize(mask)

    Pixels whose resized mask weight is below ``min_weight`` are marked invalid.
    """

    mask_f = mask.astype(np.float32)
    arr_f = np.where(mask, arr, 0).astype(np.float32)
    if arr.shape == shape:
        weight = mask_f.astype(np.float64)
        valid = weight >= min_weight
        return arr.astype(np.float64, copy=False), valid, weight
    num = resize_float(arr_f * mask_f, shape, resample=resample)
    den = resize_float(mask_f, shape, resample=resample)
    out = np.full(shape, np.nan, dtype=np.float64)
    valid = den >= min_weight
    out[valid] = num[valid] / np.maximum(den[valid], EPS)
    return out, valid, den


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = img.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def finite_positive(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr > 0)


def safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan"), float("nan")
    pear = float(pearsonr(a, b).statistic)
    spear = float(spearmanr(a, b).statistic)
    return pear, spear


def relief(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 95) - np.percentile(arr, 5))


def gradients(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(arr.astype(np.float64))
    return gx, gy


def eroded_local_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(bool)
    return binary_erosion(mask.astype(bool), structure=np.ones((3, 3), dtype=bool), iterations=iterations, border_value=0)


def gradient_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    *,
    highgrad_quantile: float = 75.0,
) -> dict[str, float | int]:
    gx_c, gy_c = gradients(candidate)
    gx_r, gy_r = gradients(reference)
    gm = eroded_local_mask(mask) & np.isfinite(gx_c) & np.isfinite(gy_c) & np.isfinite(gx_r) & np.isfinite(gy_r)
    out: dict[str, float | int] = {
        "gradient_local_valid_count": int(gm.sum()),
        "gradient_highgrad_count": 0,
        "gradient_highgrad_quantile": float(highgrad_quantile),
        "gradient_highgrad_threshold": float("nan"),
        "gradient_primary_cosine_highgrad_median": float("nan"),
        "gradient_primary_magnitude_absrel_highgrad_median": float("nan"),
        "gradient_sensitivity_cosine_allvalid_median": float("nan"),
        "gradient_sensitivity_magnitude_absrel_allvalid_median": float("nan"),
    }
    if gm.sum() < 10:
        return out
    dot = gx_c[gm] * gx_r[gm] + gy_c[gm] * gy_r[gm]
    mag_c = np.sqrt(gx_c[gm] ** 2 + gy_c[gm] ** 2)
    mag_r = np.sqrt(gx_r[gm] ** 2 + gy_r[gm] ** 2)
    grad_cos = dot / np.maximum(mag_c * mag_r, EPS)
    # Magnitude ratio uses a floor tied to the reference gradient distribution,
    # so tiny textureless gradients do not dominate AbsRel-style values.
    mag_floor = max(float(np.percentile(mag_r, 10)), EPS)
    grad_mag_err = np.abs(mag_c - mag_r) / np.maximum(mag_r, mag_floor)
    out["gradient_sensitivity_cosine_allvalid_median"] = float(np.median(grad_cos))
    out["gradient_sensitivity_magnitude_absrel_allvalid_median"] = float(np.median(grad_mag_err))

    threshold = float(np.percentile(mag_r, highgrad_quantile))
    high = mag_r > threshold
    if high.sum() < 10:
        high = mag_r >= threshold
    out["gradient_highgrad_threshold"] = threshold
    out["gradient_highgrad_count"] = int(high.sum())
    if high.sum() >= 10:
        out["gradient_primary_cosine_highgrad_median"] = float(np.median(grad_cos[high]))
        out["gradient_primary_magnitude_absrel_highgrad_median"] = float(np.median(grad_mag_err[high]))
    return out


def spatial_block_bootstrap_ci(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    *,
    block: int = 16,
    reps: int = 200,
    seed: int = 230623,
) -> dict[str, float | int]:
    """Spatial-block bootstrap CIs for actual global metrics.

    Blocks are sampled with replacement. Pixels from sampled blocks are
    concatenated, and the global median AbsRel and Spearman are recomputed for
    each replicate. This is intentionally not a block-summary bootstrap.
    """

    rng = np.random.default_rng(seed)
    h, w = mask.shape
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            sl = (slice(y, min(y + block, h)), slice(x, min(x + block, w)))
            m = mask[sl] & finite_positive(candidate[sl]) & finite_positive(reference[sl])
            if m.sum() < 8:
                continue
            c = candidate[sl][m].astype(np.float64)
            r = reference[sl][m].astype(np.float64)
            blocks.append((c, r))
    if len(blocks) < 4:
        return {
            "spatial_block_count": len(blocks),
            "spatial_block_bootstrap_absrel_median_ci_low": float("nan"),
            "spatial_block_bootstrap_absrel_median_ci_high": float("nan"),
            "spatial_block_bootstrap_spearman_ci_low": float("nan"),
            "spatial_block_bootstrap_spearman_ci_high": float("nan"),
        }
    absrel_samples = []
    spear_samples = []
    n = len(blocks)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        c = np.concatenate([blocks[i][0] for i in idx])
        r = np.concatenate([blocks[i][1] for i in idx])
        absrel = np.abs(c - r) / np.maximum(np.abs(r), EPS)
        absrel_samples.append(float(np.median(absrel)))
        if np.std(c) < EPS or np.std(r) < EPS:
            spear_samples.append(float("nan"))
        else:
            spear_samples.append(float(spearmanr(c, r).statistic))
    spear_arr = np.asarray(spear_samples, dtype=np.float64)
    spear_arr = spear_arr[np.isfinite(spear_arr)]
    return {
        "spatial_block_count": n,
        "spatial_block_bootstrap_absrel_median_ci_low": float(np.percentile(absrel_samples, 2.5)),
        "spatial_block_bootstrap_absrel_median_ci_high": float(np.percentile(absrel_samples, 97.5)),
        "spatial_block_bootstrap_spearman_ci_low": float(np.percentile(spear_arr, 2.5)) if spear_arr.size else float("nan"),
        "spatial_block_bootstrap_spearman_ci_high": float(np.percentile(spear_arr, 97.5)) if spear_arr.size else float("nan"),
    }


def metric_row(
    comparison: str,
    domain: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    *,
    source_scale: float | None = None,
    scale_mode: str = "raw",
    negative_control: str = "none",
) -> dict[str, Any]:
    m = mask & finite_positive(candidate) & finite_positive(reference)
    total = int(mask.size)
    count = int(m.sum())
    row: dict[str, Any] = {
        "comparison": comparison,
        "domain": domain,
        "scale_mode": scale_mode,
        "negative_control": negative_control,
        "valid_count": count,
        "total_pixels": total,
        "coverage_full_raster": count / total if total else float("nan"),
    }
    if count < 10:
        row.update(
            {
                "absrel_median": float("nan"),
                "absrel_p90": float("nan"),
                "absrel_mean": float("nan"),
                "rmse": float("nan"),
                "pearson": float("nan"),
                "spearman": float("nan"),
                "candidate_relief_p95_p5": float("nan"),
                "reference_relief_p95_p5": float("nan"),
                "relative_relief_error": float("nan"),
                "gradient_local_valid_count": 0,
                "gradient_highgrad_count": 0,
                "gradient_highgrad_quantile": float("nan"),
                "gradient_highgrad_threshold": float("nan"),
                "gradient_primary_cosine_highgrad_median": float("nan"),
                "gradient_primary_magnitude_absrel_highgrad_median": float("nan"),
                "gradient_sensitivity_cosine_allvalid_median": float("nan"),
                "gradient_sensitivity_magnitude_absrel_allvalid_median": float("nan"),
                "scale_aligned_factor_sensitivity": float("nan"),
                "scale_aligned_absrel_median_sensitivity": float("nan"),
                "spatial_block_count": 0,
                "spatial_block_bootstrap_absrel_median_ci_low": float("nan"),
                "spatial_block_bootstrap_absrel_median_ci_high": float("nan"),
                "spatial_block_bootstrap_spearman_ci_low": float("nan"),
                "spatial_block_bootstrap_spearman_ci_high": float("nan"),
            }
        )
        return row

    c = candidate[m].astype(np.float64)
    r = reference[m].astype(np.float64)
    absrel = np.abs(c - r) / np.maximum(np.abs(r), EPS)
    pear, spear = safe_corr(c, r)
    c_relief = relief(c)
    r_relief = relief(r)
    rel_relief_err = abs(c_relief - r_relief) / max(abs(r_relief), EPS)

    grad = gradient_metrics(candidate, reference, m)
    boot = spatial_block_bootstrap_ci(candidate, reference, m)

    # Sensitivity only: fit one median scale on the evaluation mask.
    scale_aligned = float(np.median(r / np.maximum(c, EPS)))
    c_sa = c * scale_aligned
    absrel_sa = np.abs(c_sa - r) / np.maximum(np.abs(r), EPS)

    row.update(
        {
            "source_scale_used": source_scale if source_scale is not None else "",
            "absrel_median": float(np.median(absrel)),
            "absrel_p90": float(np.percentile(absrel, 90)),
            "absrel_mean": float(np.mean(absrel)),
            "rmse": float(np.sqrt(np.mean((c - r) ** 2))),
            "pearson": pear,
            "spearman": spear,
            "candidate_relief_p95_p5": c_relief,
            "reference_relief_p95_p5": r_relief,
            "relative_relief_error": float(rel_relief_err),
            "scale_aligned_factor_sensitivity": scale_aligned,
            "scale_aligned_absrel_median_sensitivity": float(np.median(absrel_sa)),
            "scale_aligned_absrel_p90_sensitivity": float(np.percentile(absrel_sa, 90)),
        }
    )
    row.update(grad)
    row.update(boot)
    return row


def shuffled(arr: np.ndarray, mask: np.ndarray, seed: int = 230623) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = arr.copy()
    vals = out[mask].copy()
    rng.shuffle(vals)
    out[mask] = vals
    return out


def mirrored(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.fliplr(arr), np.fliplr(mask.astype(bool))


def save_map(path: Path, arr: np.ndarray, title: str, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_contact_sheet(out: Path, panels: list[tuple[str, np.ndarray, str]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8))
    axes = axes.ravel()
    for ax, (title, arr, cmap) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    residual_dir = out / "residual_maps"
    contact_dir = out / "contact_sheets"
    residual_dir.mkdir(exist_ok=True)
    contact_dir.mkdir(exist_ok=True)

    da = np.load(args.da3_mvs_npz)
    ma = np.load(args.mapanything_target_npz)
    robust_rows = read_rows(args.robust_scale_csv)
    vb = next(r for r in robust_rows if r["estimator"] == "C_two_level_view_balanced_median")
    map_source_scale = float(vb["derived_original_colmap_mvs_frame_scale"])

    da3 = da["target_conditioned_da3_depth"].astype(np.float64)
    da3_conf = da["target_da3_confidence"].astype(np.float64)
    mvs = da["mvs_reference_depth"].astype(np.float64)
    mvs_mask = da["mvs_valid_mask"].astype(bool)
    da3_mask = da["target_proxy_valid_mask"].astype(bool)
    da3_mvs_common = da["common_mask"].astype(bool)
    source_proj = da["source_projected_da3_depth"].astype(np.float64)
    source_proj_mask = da["source_projection_common_mask"].astype(bool)

    target_shape = mvs.shape
    map_native_mask = ma["mask"].astype(bool) & finite_positive(ma["depth_z"].astype(np.float64))
    map_raw, map_mask, map_resize_weight = resize_masked_float(
        ma["depth_z"].astype(np.float64),
        map_native_mask,
        target_shape,
        min_weight=0.5,
    )
    map_conf, map_conf_mask, _ = resize_masked_float(
        ma["confidence"].astype(np.float64),
        ma["mask"].astype(bool),
        target_shape,
        min_weight=0.5,
    )
    map_mask = map_mask & map_conf_mask
    map_scaled = map_raw * map_source_scale

    masks = {
        "native_da3_mvs": da3_mvs_common & finite_positive(da3) & finite_positive(mvs),
        "native_map_mvs": mvs_mask & map_mask & finite_positive(map_raw) & finite_positive(mvs),
        "native_da3_map": da3_mask & map_mask & finite_positive(da3) & finite_positive(map_raw),
    }
    masks["triple_common"] = masks["native_da3_mvs"] & map_mask & finite_positive(map_raw)

    metric_rows: list[dict[str, Any]] = []
    comparisons = [
        ("DA3_vs_MVS", da3, mvs, None, masks["native_da3_mvs"]),
        ("MapAnything_raw_vs_MVS", map_raw, mvs, None, masks["native_map_mvs"]),
        ("MapAnything_source_scaled_vs_MVS", map_scaled, mvs, map_source_scale, masks["native_map_mvs"]),
        ("MapAnything_raw_vs_DA3", map_raw, da3, None, masks["native_da3_map"]),
        ("MapAnything_source_scaled_vs_DA3", map_scaled, da3, map_source_scale, masks["native_da3_map"]),
        ("DA3_target_vs_DA3_source_projection", da3, source_proj, None, source_proj_mask),
    ]
    for comp, cand, ref, scale, native_mask in comparisons:
        metric_rows.append(metric_row(comp, "native_common", cand, ref, native_mask, source_scale=scale, scale_mode="source_scaled" if scale else "raw"))
        if comp != "DA3_target_vs_DA3_source_projection":
            metric_rows.append(metric_row(comp, "triple_common", cand, ref, masks["triple_common"], source_scale=scale, scale_mode="source_scaled" if scale else "raw"))

    # Confidence-gated sensitivity on triple-common where applicable.
    sensitivity_rows: list[dict[str, Any]] = []
    conf_specs = [
        ("none", masks["triple_common"]),
        ("map_q20", masks["triple_common"] & (map_conf >= np.percentile(map_conf[masks["triple_common"]], 20))),
        ("map_q50", masks["triple_common"] & (map_conf >= np.percentile(map_conf[masks["triple_common"]], 50))),
        ("da3_q20", masks["triple_common"] & (da3_conf >= np.percentile(da3_conf[masks["triple_common"]], 20))),
        ("da3_q50", masks["triple_common"] & (da3_conf >= np.percentile(da3_conf[masks["triple_common"]], 50))),
        (
            "map_q20_and_da3_q20",
            masks["triple_common"]
            & (map_conf >= np.percentile(map_conf[masks["triple_common"]], 20))
            & (da3_conf >= np.percentile(da3_conf[masks["triple_common"]], 20)),
        ),
    ]
    for label, cmask in conf_specs:
        for comp, cand, ref, scale in [
            ("DA3_vs_MVS", da3, mvs, None),
            ("MapAnything_source_scaled_vs_MVS", map_scaled, mvs, map_source_scale),
            ("MapAnything_source_scaled_vs_DA3", map_scaled, da3, map_source_scale),
        ]:
            row = metric_row(comp, f"triple_common_conf_{label}", cand, ref, cmask, source_scale=scale, scale_mode="source_scaled" if scale else "raw")
            row["confidence_filter"] = label
            sensitivity_rows.append(row)

    # Negative controls on triple-common.
    neg_rows: list[dict[str, Any]] = []
    map_mirror, map_mirror_mask = mirrored(map_scaled, map_mask)
    da3_mirror, da3_mirror_mask = mirrored(da3, da3_mask)
    neg_pairs = [
        ("DA3_shuffle_vs_MVS", shuffled(da3, masks["triple_common"]), mvs, None, masks["triple_common"]),
        ("MapAnything_scaled_shuffle_vs_MVS", shuffled(map_scaled, masks["triple_common"], seed=230624), mvs, map_source_scale, masks["triple_common"]),
        ("MapAnything_scaled_mirror_vs_MVS", map_mirror, mvs, map_source_scale, map_mirror_mask & mvs_mask & finite_positive(map_mirror) & finite_positive(mvs)),
        ("DA3_mirror_vs_MVS", da3_mirror, mvs, None, da3_mirror_mask & mvs_mask & finite_positive(da3_mirror) & finite_positive(mvs)),
        ("MapAnything_scaled_shuffle_vs_DA3", shuffled(map_scaled, masks["triple_common"], seed=230625), da3, map_source_scale, masks["triple_common"]),
    ]
    for comp, cand, ref, scale, neg_mask in neg_pairs:
        kind = "mirror" if "mirror" in comp else "spatial_shuffle"
        neg_rows.append(metric_row(comp, "negative_control_common", cand, ref, neg_mask, source_scale=scale, scale_mode="source_scaled" if scale else "raw", negative_control=kind))

    write_rows(out / "tri_proxy_pairwise_metrics.csv", metric_rows)
    write_rows(out / "tri_proxy_confidence_sensitivity.csv", sensitivity_rows)
    write_rows(out / "tri_proxy_negative_controls.csv", neg_rows)

    # Visuals.
    abs_map_mvs = np.full(target_shape, np.nan)
    m = masks["triple_common"]
    abs_map_mvs[m] = np.abs(map_scaled[m] - mvs[m]) / np.maximum(mvs[m], EPS)
    abs_da3_mvs = np.full(target_shape, np.nan)
    abs_da3_mvs[m] = np.abs(da3[m] - mvs[m]) / np.maximum(mvs[m], EPS)
    abs_map_da3 = np.full(target_shape, np.nan)
    abs_map_da3[m] = np.abs(map_scaled[m] - da3[m]) / np.maximum(da3[m], EPS)
    save_map(residual_dir / "mapanything_scaled_vs_mvs_absrel.png", abs_map_mvs, "MapAnything scaled vs MVS AbsRel", "magma", 0, np.nanpercentile(abs_map_mvs, 95))
    save_map(residual_dir / "da3_vs_mvs_absrel.png", abs_da3_mvs, "DA3 vs MVS AbsRel", "magma", 0, np.nanpercentile(abs_da3_mvs, 95))
    save_map(residual_dir / "mapanything_scaled_vs_da3_absrel.png", abs_map_da3, "MapAnything scaled vs DA3 AbsRel", "magma", 0, np.nanpercentile(abs_map_da3, 95))
    make_contact_sheet(
        contact_dir / "tri_proxy_depth_and_residual_contact_sheet.png",
        [
            ("MVS proxy depth", mvs, "viridis"),
            ("DA3 target depth", da3, "viridis"),
            ("MapAnything source-scaled depth", map_scaled, "viridis"),
            ("DA3 vs MVS AbsRel", abs_da3_mvs, "magma"),
            ("MapAnything vs MVS AbsRel", abs_map_mvs, "magma"),
            ("MapAnything vs DA3 AbsRel", abs_map_da3, "magma"),
        ],
    )

    pair_lookup = {(r["comparison"], r["domain"]): r for r in metric_rows}
    neg_lookup = {(r["comparison"], r["domain"]): r for r in neg_rows}
    da3_mvs = pair_lookup[("DA3_vs_MVS", "triple_common")]
    map_mvs = pair_lookup[("MapAnything_source_scaled_vs_MVS", "triple_common")]
    map_da3 = pair_lookup[("MapAnything_source_scaled_vs_DA3", "triple_common")]
    da3_internal = pair_lookup[("DA3_target_vs_DA3_source_projection", "native_common")]
    # Use pre-existing MapAnything target-source CSV for internal consistency summary if available.
    map_internal_rows = read_rows(args.target_source_consistency_csv)
    map_internal = next((r for r in map_internal_rows if r.get("count")), map_internal_rows[0] if map_internal_rows else {})

    decision_md = f"""# DA3--MapAnything--MVS Decision Table

This table is a CPU-only comparison on Papaya target 0049. None of the three proxies is treated as ground truth.

| Criterion | DA3 | MapAnything | MVS proxy |
|---|---|---|---|
| coverage | triple-common coverage {float(da3_mvs['coverage_full_raster']):.3f} | triple-common coverage {float(map_mvs['coverage_full_raster']):.3f}; target mask high after resize | MVS mask is dense for this target but remains a proxy |
| internal multi-view consistency | DA3 target/source median AbsRel {float(da3_internal['absrel_median']):.4f} on DA3 source projection mask | MapAnything target/source median AbsRel {float(map_internal.get('absrel_median', float('nan'))):.4f} in internal scale | not applicable here; MVS is existing mesh-rendered proxy |
| agreement with sparse anchors | sparse anchors are biased local checks; not used as truth | source-side scale from sparse anchors is stable but does not prove dense geometry | sparse anchors also underlie COLMAP/MVS pipeline |
| agreement with MVS | low median AbsRel {float(da3_mvs['absrel_median']):.4f}, but low Spearman {float(da3_mvs['spearman']):.3f} and negative controls show AbsRel is weak on this narrow-depth target | source-scaled median AbsRel {float(map_mvs['absrel_median']):.4f}; Spearman {float(map_mvs['spearman']):.3f}; not MVS-equivalent | reference proxy in this comparison, not ground truth |
| agreement with DA3/MapAnything | vs MapAnything median AbsRel {float(map_da3['absrel_median']):.4f} | vs DA3 median AbsRel {float(map_da3['absrel_median']):.4f}; Spearman {float(map_da3['spearman']):.3f} | only a comparator here; cross-proxy disagreement should not be resolved by declaring MVS ground truth |
| relief consistency | DA3-vs-MVS relative relief error {float(da3_mvs['relative_relief_error']):.3f}; not decisive | MapAnything-vs-MVS relative relief error {float(map_mvs['relative_relief_error']):.3f}; vs DA3 {float(map_da3['relative_relief_error']):.3f} | used only as proxy comparator |
| gradient consistency | DA3-vs-MVS high-gradient cosine median {float(da3_mvs['gradient_primary_cosine_highgrad_median']):.3f}; all-valid gradient is sensitivity only | MapAnything-vs-MVS high-gradient cosine median {float(map_mvs['gradient_primary_cosine_highgrad_median']):.3f}; vs DA3 {float(map_da3['gradient_primary_cosine_highgrad_median']):.3f} | no ground-truth status |
| confidence sensitivity | see `tri_proxy_confidence_sensitivity.csv` | see `tri_proxy_confidence_sensitivity.csv` | no confidence channel |
| computational cost | existing DA3 target-conditioned outputs reused; no new GPU in this comparison | paired rerun used about 7.7 GiB and <3 min per run for 0049 | MVS/Mesh is slower and historically less stable to produce |
| current manuscript suitability | feasibility / rebuttal reserve only unless multi-scene validation is approved | internal / rebuttal reserve only | current validation-gated diagnostic remains unchanged |
| future proxy-regularization suitability | stronger candidate if multi-scene consistency holds | secondary independent neural proxy / cross-check candidate | conventional proxy baseline and diagnostic comparator |
"""
    (out / "tri_proxy_decision_table.md").write_text(decision_md, encoding="utf-8")

    policy_md = f"""# Tri-proxy comparison protocol and limitations

## Inputs

- DA3/MVS NPZ: `{args.da3_mvs_npz}`
- MapAnything target NPZ: `{args.mapanything_target_npz}`
- Robust scale CSV: `{args.robust_scale_csv}`

## Raster and mask policy

- MVS and DA3 arrays are already in the target reference raster: `{target_shape}`.
- MapAnything `depth_z`, confidence, and mask are resized from `{tuple(ma['depth_z'].shape)}` to `{target_shape}`.
- Depth/confidence resize: masked normalized convolution, `resize(value * mask) / resize(mask)`.
- MapAnything resized valid pixels require resized mask weight >= 0.5.
- Metrics are reported for each pair's native common mask and for a triple-common mask shared by DA3, source-scaled MapAnything, and MVS.

## Depth and scale convention

- MapAnything raw `depth_z` is arbitrary-scale for this protocol.
- Primary MapAnything-to-MVS/DA3 metrics use the source-side view-balanced scale from the frozen K24 source anchors.
- The source scale is not fitted on target sparse, MVS, DA3, or target residuals.
- Scale-aligned AbsRel is reported as sensitivity only.

## Negative controls

- Spatial shuffle: candidate depths are permuted within the triple-common mask.
- Mirror control: candidate depth map and candidate mask are horizontally flipped together, then the common mask is recomputed.
- No next-view control is reported because this package is restricted to target 0049 and existing outputs.

## Manuscript policy

This comparison does not modify the current manuscript, Fig.3, Table III, benchmark metrics, UMGS checkpoints, or Gaussian support.
"""
    (out / "TRI_PROXY_COMPARISON_PROTOCOL.md").write_text(policy_md, encoding="utf-8")

    summary_md = f"""# Tri-proxy comparison summary

Status: `cpu_only_unified_proxy_comparison_completed`

## Primary triple-common evidence

- DA3 vs MVS median/P90 AbsRel: {float(da3_mvs['absrel_median']):.4f} / {float(da3_mvs['absrel_p90']):.4f}; Spearman {float(da3_mvs['spearman']):.3f}; high-gradient cosine median {float(da3_mvs['gradient_primary_cosine_highgrad_median']):.3f}.
- Source-scaled MapAnything vs MVS median/P90 AbsRel: {float(map_mvs['absrel_median']):.4f} / {float(map_mvs['absrel_p90']):.4f}; Spearman {float(map_mvs['spearman']):.3f}; high-gradient cosine median {float(map_mvs['gradient_primary_cosine_highgrad_median']):.3f}.
- Source-scaled MapAnything vs DA3 median/P90 AbsRel: {float(map_da3['absrel_median']):.4f} / {float(map_da3['absrel_p90']):.4f}; Spearman {float(map_da3['spearman']):.3f}; high-gradient cosine median {float(map_da3['gradient_primary_cosine_highgrad_median']):.3f}.

## Interpretation

DA3 is much closer to the existing MVS proxy in absolute scaled depth on this Papaya 0049 comparison, but shape evidence is mixed: DA3--MVS rank/gradient agreement is weak and AbsRel is known to be fragile on this narrow-depth target. MapAnything has good internal multi-view consistency and stable source-side scale estimation, and it agrees more with DA3 than with MVS in rank/gradient structure, but this unified comparison still does not support replacing MVS/Mesh with MapAnything as a primary geometry diagnostic.

Current recommendation: keep the manuscript geometry diagnostic unchanged. Treat MapAnything as internal / rebuttal reserve. If the geometry-proxy track continues, prioritize a carefully frozen multi-scene DA3-vs-MVS/MapAnything validation protocol rather than selecting a new proxy from Papaya 0049 alone.
"""
    (out / "TRI_PROXY_COMPARISON_SUMMARY.md").write_text(summary_md, encoding="utf-8")

    input_manifest = {
        "schema": "tri_proxy_comparison_manifest_v1",
        "command_line": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "da3_mvs_npz": str(args.da3_mvs_npz),
            "da3_mvs_npz_sha256": sha256_file(args.da3_mvs_npz),
            "mapanything_target_npz": str(args.mapanything_target_npz),
            "mapanything_target_npz_sha256": sha256_file(args.mapanything_target_npz),
            "robust_scale_csv": str(args.robust_scale_csv),
            "robust_scale_csv_sha256": sha256_file(args.robust_scale_csv),
            "mapanything_manifest": str(args.mapanything_manifest),
            "mapanything_manifest_sha256": sha256_file(args.mapanything_manifest),
            "source_only_manifest": str(args.source_only_manifest),
            "source_only_manifest_sha256": sha256_file(args.source_only_manifest),
            "target_source_consistency_csv": str(args.target_source_consistency_csv),
            "target_source_consistency_csv_sha256": sha256_file(args.target_source_consistency_csv),
        },
        "source_scale": {
            "estimator": "C_two_level_view_balanced_median",
            "derived_original_colmap_mvs_frame_scale": map_source_scale,
            "saved_frame_scale": float(vb["scale"]),
            "valid_source_view_count": int(float(vb["valid_source_view_count"])),
        },
        "no_gpu": True,
        "no_inference": True,
        "no_manuscript_change": True,
    }
    (out / "input_hashes_and_environment.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    package_manifest = "# Package manifest\n\n"
    package_files = [
        out / "TRI_PROXY_COMPARISON_SUMMARY.md",
        out / "TRI_PROXY_COMPARISON_PROTOCOL.md",
        out / "tri_proxy_decision_table.md",
        out / "tri_proxy_pairwise_metrics.csv",
        out / "tri_proxy_confidence_sensitivity.csv",
        out / "tri_proxy_negative_controls.csv",
        out / "input_hashes_and_environment.json",
        Path(__file__),
    ]
    for d in [residual_dir, contact_dir]:
        package_files.extend([p for p in d.rglob("*") if p.is_file()])
    package_files = [p for p in package_files if p.exists()]
    for p in package_files:
        package_manifest += f"- `{p.name if p == Path(__file__) else p.relative_to(out).as_posix()}` ({p.stat().st_size} bytes, sha256 `{sha256_file(p)}`)\n"
    (out / "PACKAGE_MANIFEST.md").write_text(package_manifest, encoding="utf-8")
    package_files.insert(1, out / "PACKAGE_MANIFEST.md")

    zip_path = out.parent / args.package_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in package_files:
            arc = "scripts/tri_proxy_comparison.py" if p == Path(__file__) else p.relative_to(out).as_posix()
            z.write(p, arc)
    print(out / "TRI_PROXY_COMPARISON_SUMMARY.md")
    print(zip_path)


if __name__ == "__main__":
    main()
