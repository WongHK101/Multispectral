#!/usr/bin/env python3
"""Corrected OpenMVS--DA3 overlap calibration from existing outputs.

This evaluator is CPU-only and consumes existing OpenMVS triangle renders,
OpenMVS source-support masks, and existing DA3 evaluator arrays. It does not
run OpenMVS, DA3 inference, UMGS rendering/training, or manuscript edits.

The correction separates native descriptive metrics from shared-mask negative
control comparisons. Any true-vs-control statement is computed on exactly the
same pixel set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
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


SHUFFLE_SEED = 230626
METRIC_COMPARE_EPS = 1e-6
MIN_SHARED_PIXELS = 10_000
MIN_SHARED_COVERAGE = 0.05
MIN_TRUE_PIXELS = 10_000
MIN_TRUE_COVERAGE = 0.05
HIGHGRAD_QUANTILE = 75.0
RESIZE_MIN_WEIGHT = 0.5


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def np_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(bool)


def finite_positive(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & (a > 0)


def resize_float(arr: np.ndarray, shape: tuple[int, int], resample: int = Image.Resampling.BILINEAR) -> np.ndarray:
    h, w = shape
    src = np.asarray(arr, dtype=np.float32)
    im = Image.fromarray(src, mode="F")
    return np.asarray(im.resize((w, h), resample=resample), dtype=np.float32)


def resize_masked_float(
    arr: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
    *,
    min_weight: float = RESIZE_MIN_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(arr, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(src)
    mask_f = valid.astype(np.float32)
    num = resize_float(np.where(valid, src, 0.0).astype(np.float32) * mask_f, shape)
    den = resize_float(mask_f, shape)
    out = np.full(shape, np.nan, dtype=np.float32)
    ok = den >= float(min_weight)
    out[ok] = num[ok] / np.maximum(den[ok], 1e-12)
    return out, ok, den


def branch_native(depth: np.ndarray, accepted: np.ndarray, branch: str) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    accepted = np.asarray(accepted, dtype=bool)
    if branch == "true":
        return depth.copy(), accepted.copy()
    if branch == "mirror":
        return np.fliplr(depth), np.fliplr(accepted)
    if branch == "shuffle":
        rng = np.random.default_rng(SHUFFLE_SEED)
        idx = rng.permutation(depth.size)
        return depth.reshape(-1)[idx].reshape(depth.shape), accepted.reshape(-1)[idx].reshape(accepted.shape)
    raise ValueError(f"unknown branch {branch}")


def resize_da3_branch(
    depth_native: np.ndarray,
    accepted_native: np.ndarray,
    target_shape: tuple[int, int],
    branch: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_b, mask_b = branch_native(depth_native, accepted_native, branch)
    return resize_masked_float(depth_b, mask_b & finite_positive(depth_b), target_shape, min_weight=RESIZE_MIN_WEIGHT)


def safe_corr(kind: str, x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 10:
        return None
    try:
        if kind == "pearson":
            val = pearsonr(x, y).statistic
        else:
            val = spearmanr(x, y, nan_policy="omit").correlation
    except Exception:
        return None
    return float(val) if np.isfinite(val) else None


def complete_local_mask(mask: np.ndarray) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if base.shape[0] < 3 or base.shape[1] < 3:
        return np.zeros_like(base, dtype=bool)
    return binary_erosion(base, structure=np.ones((3, 3), dtype=bool), iterations=1, border_value=0)


@dataclass
class GradientDomain:
    high_mask: np.ndarray
    threshold: float | None
    local_valid_count: int
    high_count: int
    erosion_rule: str = "3x3_binary_erosion_one_iteration_plus_complete_finite_gradient_neighborhood"


def reference_high_gradient_domain(reference: np.ndarray, mask: np.ndarray) -> GradientDomain:
    local = complete_local_mask(mask & np.isfinite(reference))
    if int(local.sum()) < 100:
        return GradientDomain(np.zeros_like(mask, dtype=bool), None, int(local.sum()), 0)
    gy, gx = np.gradient(np.where(np.isfinite(reference), reference, np.nan).astype(np.float64))
    finite = local & np.isfinite(gx) & np.isfinite(gy)
    mag = np.sqrt(gx * gx + gy * gy)
    vals = mag[finite]
    if len(vals) < 100:
        return GradientDomain(np.zeros_like(mask, dtype=bool), None, int(len(vals)), 0)
    threshold = float(np.percentile(vals, HIGHGRAD_QUANTILE))
    high = finite & (mag >= threshold)
    return GradientDomain(high, threshold, int(finite.sum()), int(high.sum()))


def metrics_on_mask(
    openmvs: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    *,
    gradient_domain: GradientDomain | None = None,
) -> dict[str, Any]:
    m = mask & finite_positive(openmvs) & finite_positive(candidate)
    pix = int(m.sum())
    coverage = float(pix / m.size) if m.size else 0.0
    if pix == 0:
        return {
            "pixels": 0,
            "coverage": coverage,
            "absrel_median": None,
            "absrel_p90": None,
            "pearson": None,
            "spearman": None,
            "high_gradient_cosine_median": None,
            "high_gradient_threshold": None,
            "high_gradient_pixels": 0,
            "gradient_local_valid_pixels": 0,
            "gradient_erosion_rule": "3x3_binary_erosion_one_iteration_plus_complete_finite_gradient_neighborhood",
        }
    ov = openmvs[m].astype(np.float64)
    cv = candidate[m].astype(np.float64)
    rel = np.abs(cv - ov) / np.maximum(np.abs(ov), 1e-6)
    gd = gradient_domain or reference_high_gradient_domain(openmvs, m)
    cos_val: float | None = None
    if gd.high_count > 0:
        gy_o, gx_o = np.gradient(openmvs.astype(np.float64))
        gy_c, gx_c = np.gradient(candidate.astype(np.float64))
        h = gd.high_mask & np.isfinite(gx_o) & np.isfinite(gy_o) & np.isfinite(gx_c) & np.isfinite(gy_c)
        if int(h.sum()) > 0:
            dot = gx_o[h] * gx_c[h] + gy_o[h] * gy_c[h]
            norm = np.sqrt(gx_o[h] ** 2 + gy_o[h] ** 2) * np.sqrt(gx_c[h] ** 2 + gy_c[h] ** 2)
            cos = dot / np.maximum(norm, 1e-12)
            cos_val = float(np.median(cos)) if len(cos) else None
    return {
        "pixels": pix,
        "coverage": coverage,
        "absrel_median": float(np.median(rel)),
        "absrel_p90": float(np.percentile(rel, 90)),
        "pearson": safe_corr("pearson", ov, cv),
        "spearman": safe_corr("spearman", ov, cv),
        "high_gradient_cosine_median": cos_val,
        "high_gradient_threshold": gd.threshold,
        "high_gradient_pixels": int(gd.high_count),
        "gradient_local_valid_pixels": int(gd.local_valid_count),
        "gradient_erosion_rule": gd.erosion_rule,
    }


def prefix(d: dict[str, Any], p: str) -> dict[str, Any]:
    return {f"{p}{k}": v for k, v in d.items()}


def control_status(true_native: dict[str, Any], true_shared: dict[str, Any], control_shared: dict[str, Any], usable: bool) -> tuple[str, str]:
    if true_native["pixels"] < MIN_TRUE_PIXELS or true_native["coverage"] < MIN_TRUE_COVERAGE:
        return "insufficient_common_support", ""
    if not usable:
        return "weak_or_mixed_proxy_agreement", "negative_control_inconclusive_due_to_shared_support"
    t_spear = true_native.get("spearman")
    t_grad = true_native.get("high_gradient_cosine_median")
    if t_spear is not None and t_spear < 0:
        return "proxy_contradiction", ""
    if t_grad is not None and t_grad < 0:
        return "proxy_contradiction", ""
    checks: dict[str, bool | None] = {}
    if true_shared.get("absrel_median") is not None and control_shared.get("absrel_median") is not None:
        checks["absrel"] = float(true_shared["absrel_median"]) < float(control_shared["absrel_median"]) - METRIC_COMPARE_EPS
    else:
        checks["absrel"] = None
    if true_shared.get("spearman") is not None and control_shared.get("spearman") is not None:
        checks["spearman"] = float(true_shared["spearman"]) > float(control_shared["spearman"]) + METRIC_COMPARE_EPS
    else:
        checks["spearman"] = None
    if true_shared.get("high_gradient_cosine_median") is not None and control_shared.get("high_gradient_cosine_median") is not None:
        checks["high_gradient_cosine"] = (
            float(true_shared["high_gradient_cosine_median"]) > float(control_shared["high_gradient_cosine_median"]) + METRIC_COMPARE_EPS
        )
    else:
        checks["high_gradient_cosine"] = None
    valid_checks = {k: v for k, v in checks.items() if v is not None}
    failures = sum(1 for ok in valid_checks.values() if not ok)
    if len(valid_checks) >= 2 and failures >= 2:
        return "proxy_contradiction", ""
    caution = "" if len(valid_checks) >= 2 else "metric_inconclusive"
    if (
        (t_spear is not None and t_spear >= 0.70)
        and (t_grad is not None and t_grad >= 0.70)
        and checks["absrel"] is True
        and (checks["spearman"] is True or checks["high_gradient_cosine"] is True)
    ):
        return "strong_proxy_agreement", ""
    return "weak_or_mixed_proxy_agreement", caution


def load_da3_npz(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    data = np.load(path)
    if "target_depth" in data.files:
        depth = data["target_depth"].astype(np.float32)
        accepted = np_bool(data["common_mask"]) if "common_mask" in data.files else finite_positive(depth)
        return depth, accepted, "corrected_multiscene_pilot_protocol"
    if "target_conditioned_da3_depth" in data.files:
        depth = data["target_conditioned_da3_depth"].astype(np.float32)
        if "common_mask" in data.files:
            accepted = np_bool(data["common_mask"])
        elif "target_proxy_valid_mask" in data.files:
            accepted = np_bool(data["target_proxy_valid_mask"])
        else:
            accepted = finite_positive(depth)
        return depth, accepted, "legacy_stage2b_protocol_nonpoolable"
    raise ValueError(f"Unsupported DA3 npz keys in {path}: {data.files}")


def exact_one(paths: list[Path], description: str) -> Path | None:
    if len(paths) == 1:
        return paths[0]
    return None


def target_token_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("DJI_") and part.endswith("_D"):
            return part
        if part.startswith("target_"):
            return part.replace("target_", "")
    m = re.search(r"_(\d{4})_D", str(path))
    if m:
        return m.group(1)
    raise ValueError(f"Cannot infer target token from {path}")


def collect_da3_rows(da3_multiscene_root: Path, papaya_stage2b_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase_d = da3_multiscene_root / "phase_d_subset_pilot"
    for scene in ["maize_02_20260526_1658", "road_01_20260602_1648_40m"]:
        for npz in sorted((phase_d / scene / "eval").glob("DJI_*_D/target_conditioned_target_source_arrays.npz")):
            rows.append(
                {
                    "scene": scene,
                    "target_token": npz.parent.name,
                    "da3_npz": str(npz),
                    "protocol_family": "corrected_multiscene_pilot_protocol",
                }
            )
    pap_eval = papaya_stage2b_root / "results" / "target_conditioned_fusion_vs_mvs"
    for npz in sorted(pap_eval.glob("target_*/target_conditioned_da3_proxy_vs_mvs_reference.npz")):
        rows.append(
            {
                "scene": "papaya_01_20251217",
                "target_token": npz.parent.name.replace("target_", ""),
                "da3_npz": str(npz),
                "protocol_family": "legacy_stage2b_protocol_nonpoolable",
            }
        )
    return rows


def match_openmvs_files(campaign_root: Path, support_root: Path, scene: str, token: str) -> tuple[Path | None, Path | None, str]:
    mask_dir = support_root / "per_scene" / scene / "source_support_masks"
    tri_dir = campaign_root / scene / "source_image_only_formal_audit" / "triangle_renders"
    if token.startswith("DJI_"):
        mask_hits = sorted(mask_dir.glob(f"source_support_masks_{token}.npz"))
        tri_hits = sorted(tri_dir.glob(f"triangle_render_{token}.npz"))
    else:
        mask_hits = sorted(mask_dir.glob(f"source_support_masks_*_{token}_D.npz"))
        tri_hits = sorted(tri_dir.glob(f"triangle_render_*_{token}_D.npz"))
    reason = ""
    if len(mask_hits) != 1:
        reason += f"support_mask_match_count={len(mask_hits)};"
    if len(tri_hits) != 1:
        reason += f"triangle_render_match_count={len(tri_hits)};"
    return exact_one(tri_hits, "triangle"), exact_one(mask_hits, "support"), reason


def build_input_manifest(
    output_path: Path,
    *,
    campaign_root: Path,
    support_root: Path,
    da3_multiscene_root: Path,
    papaya_stage2b_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in collect_da3_rows(da3_multiscene_root, papaya_stage2b_root):
        tri, mask, reason = match_openmvs_files(campaign_root, support_root, item["scene"], item["target_token"])
        status = "matched" if tri and mask else "skipped_missing_or_ambiguous_openmvs_files"
        rows.append(
            {
                **item,
                "openmvs_triangle_render_npz": str(tri) if tri else "",
                "openmvs_support_mask_npz": str(mask) if mask else "",
                "manifest_status": status,
                "manifest_reason": reason,
            }
        )
    write_csv(output_path, rows)
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_preview(path: Path, openmvs: np.ndarray, da3: np.ndarray, mask: np.ndarray, title: str) -> None:
    rel = np.full(openmvs.shape, np.nan, dtype=np.float32)
    valid = mask & finite_positive(openmvs) & finite_positive(da3)
    rel[valid] = np.abs(da3[valid] - openmvs[valid]) / np.maximum(np.abs(openmvs[valid]), 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), dpi=140)
    for ax in axes:
        ax.axis("off")
    panels = [
        (openmvs, "OpenMVS z", "viridis"),
        (da3, "DA3 z", "viridis"),
        (rel, "AbsRel", "magma"),
        (mask.astype(float), "common mask", "gray"),
    ]
    for ax, arr, ttl, cmap in zip(axes, [p[0] for p in panels], [p[1] for p in panels], [p[2] for p in panels]):
        vals = arr[np.isfinite(arr)]
        if len(vals):
            lo, hi = np.nanpercentile(vals, [2, 98])
            if abs(hi - lo) < 1e-12:
                hi = lo + 1.0
            im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        else:
            ax.imshow(np.zeros_like(arr), cmap="gray")
        ax.set_title(ttl, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def evaluate_manifest(manifest_rows: list[dict[str, Any]], output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest_rows:
        if row.get("manifest_status") != "matched":
            skipped.append(row)
            continue
        scene = row["scene"]
        target = row["target_token"]
        tri_path = Path(row["openmvs_triangle_render_npz"])
        mask_path = Path(row["openmvs_support_mask_npz"])
        da3_path = Path(row["da3_npz"])
        tri = np.load(tri_path)
        masks = np.load(mask_path)
        openmvs_depth = tri["depth"].astype(np.float32)
        render_valid = np_bool(tri["valid"]) if "valid" in tri.files else finite_positive(openmvs_depth)
        raw_mask = np_bool(masks["raw_mesh_render_mask"]) & render_valid
        core_mask = np_bool(masks["source_supported_core_mask"]) & render_valid
        eroded_core = binary_erosion(core_mask, iterations=2, border_value=0)
        depth_native, accepted_native, protocol_detected = load_da3_npz(da3_path)
        target_shape = openmvs_depth.shape
        true_depth, true_valid, true_weight = resize_da3_branch(depth_native, accepted_native, target_shape, "true")
        branch_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "true": (true_depth, true_valid, true_weight),
            "shuffle": resize_da3_branch(depth_native, accepted_native, target_shape, "shuffle"),
            "mirror": resize_da3_branch(depth_native, accepted_native, target_shape, "mirror"),
        }
        domains = {
            "raw_pairwise_domain": raw_mask,
            "source_supported_core_domain": core_mask,
            "boundary_eroded_source_supported_core_domain": eroded_core,
        }
        for domain_name, openmvs_domain in domains.items():
            true_native_mask = openmvs_domain & true_valid & finite_positive(true_depth) & finite_positive(openmvs_depth)
            true_native_metrics = metrics_on_mask(openmvs_depth, true_depth, true_native_mask)
            for control in ["shuffle", "mirror"]:
                control_depth, control_valid, control_weight = branch_cache[control]
                control_native_mask = openmvs_domain & control_valid & finite_positive(control_depth) & finite_positive(openmvs_depth)
                shared_control_mask = true_native_mask & control_native_mask
                shared_pixels = int(shared_control_mask.sum())
                shared_coverage = float(shared_pixels / shared_control_mask.size) if shared_control_mask.size else 0.0
                usable = shared_pixels >= MIN_SHARED_PIXELS and shared_coverage >= MIN_SHARED_COVERAGE
                shared_gd = reference_high_gradient_domain(openmvs_depth, shared_control_mask)
                control_native_metrics = metrics_on_mask(openmvs_depth, control_depth, control_native_mask)
                true_shared_metrics = metrics_on_mask(openmvs_depth, true_depth, shared_control_mask, gradient_domain=shared_gd)
                control_shared_metrics = metrics_on_mask(openmvs_depth, control_depth, shared_control_mask, gradient_domain=shared_gd)
                status, caution = control_status(true_native_metrics, true_shared_metrics, control_shared_metrics, usable) if control == "shuffle" else ("spatial_sensitivity_only_not_used_for_strong_gate", "")
                row_out = {
                    "scene": scene,
                    "target": target,
                    "protocol_family": row.get("protocol_family") or protocol_detected,
                    "domain": domain_name,
                    "control_type": control,
                    "openmvs_triangle_render_npz": str(tri_path),
                    "openmvs_support_mask_npz": str(mask_path),
                    "da3_npz": str(da3_path),
                    "target_raster_height": int(target_shape[0]),
                    "target_raster_width": int(target_shape[1]),
                    "da3_native_height": int(depth_native.shape[0]),
                    "da3_native_width": int(depth_native.shape[1]),
                    "pixel_center_convention": "PIL_image_resize_pixel_centers; OpenMVS triangle render raster retained as target grid",
                    "interpolation_library": f"Pillow {Image.__version__}",
                    "depth_resize": "masked_normalized_bilinear_resize_depth_times_valid_over_valid_weight",
                    "valid_weight_threshold": RESIZE_MIN_WEIGHT,
                    "true_native_common_pixels": true_native_metrics["pixels"],
                    "true_native_common_coverage": true_native_metrics["coverage"],
                    "control_native_common_pixels": control_native_metrics["pixels"],
                    "control_native_common_coverage": control_native_metrics["coverage"],
                    "shared_control_pixels": shared_pixels,
                    "shared_control_coverage": shared_coverage,
                    "shared_high_gradient_pixels": shared_gd.high_count,
                    "shared_high_gradient_threshold": shared_gd.threshold,
                    "shared_gradient_erosion_rule": shared_gd.erosion_rule,
                    "control_comparison_usable": bool(usable),
                    "control_inconclusive_reason": "" if usable else "insufficient_shared_control_support",
                    "target_status": status,
                    "target_caution_flag": caution,
                    "claim_boundary": "proxy agreement only; not ground truth and not absolute geometry accuracy",
                    **prefix(true_native_metrics, "true_native_"),
                    **prefix(control_native_metrics, "control_native_"),
                    **prefix(true_shared_metrics, "true_on_shared_"),
                    **prefix(control_shared_metrics, "control_on_shared_"),
                }
                rows.append(row_out)
                support_rows.append(
                    {
                        "scene": scene,
                        "target": target,
                        "domain": domain_name,
                        "control_type": control,
                        "true_native_pixels": true_native_metrics["pixels"],
                        "control_native_pixels": control_native_metrics["pixels"],
                        "shared_control_pixels": shared_pixels,
                        "shared_control_coverage": shared_coverage,
                        "control_comparison_usable": bool(usable),
                        "control_inconclusive_reason": "" if usable else "insufficient_shared_control_support",
                    }
                )
        save_preview(
            output_root / "corrected_overlap_previews" / f"{scene}_{target}.png",
            openmvs_depth,
            true_depth,
            core_mask & true_valid,
            f"{scene} {target} source-supported core",
        )
    return rows, support_rows, skipped


def scene_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    primary = [
        r
        for r in rows
        if r["domain"] == "source_supported_core_domain" and r["control_type"] == "shuffle"
    ]
    for (scene, protocol) in sorted({(r["scene"], r["protocol_family"]) for r in primary}):
        sr = [r for r in primary if r["scene"] == scene and r["protocol_family"] == protocol]
        statuses = [r["target_status"] for r in sr]
        usable = [r for r in sr if r["control_comparison_usable"]]
        out.append(
            {
                "scene": scene,
                "protocol_family": protocol,
                "target_count": len(sr),
                "usable_shuffle_control_count": len(usable),
                "strong_proxy_agreement_count": statuses.count("strong_proxy_agreement"),
                "weak_or_mixed_proxy_agreement_count": statuses.count("weak_or_mixed_proxy_agreement"),
                "proxy_contradiction_count": statuses.count("proxy_contradiction"),
                "insufficient_common_support_count": statuses.count("insufficient_common_support"),
                "median_true_native_coverage": float(np.median([float(r["true_native_common_coverage"]) for r in sr])) if sr else 0.0,
                "median_shared_control_coverage": float(np.median([float(r["shared_control_coverage"]) for r in sr])) if sr else 0.0,
                "pooling_policy": "protocol_family_separated_not_pooled",
            }
        )
    return out


def write_resize_semantics(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# OpenMVS--DA3 Overlap Resize Semantics",
                "",
                "- Formal target raster: OpenMVS held-out triangle-render raster.",
                "- OpenMVS triangle depth stays in the native target raster.",
                "- OpenMVS source-supported masks are read in the same raster and are not bilinear-threshold resized.",
                "- DA3 true branch: read native DA3 depth and DA3 accepted mask, then masked-normalized bilinear resize to the OpenMVS raster.",
                "- DA3 mirror branch: horizontally flip native DA3 depth and accepted mask first, then apply the same masked-normalized resize.",
                "- DA3 shuffle branch: apply deterministic native-raster pixel permutation with seed 230626 to depth and accepted mask together, then apply the same masked-normalized resize.",
                "- Masked-normalized resize computes resize(depth * valid) / resize(valid).",
                "- Resized valid pixels require weight >= 0.5.",
                "- NaN and non-positive DA3 depth are invalid before resize.",
                f"- Interpolation library: Pillow {Image.__version__}.",
                "- align_corners: not applicable to Pillow resize; recorded as PIL image resize pixel-center convention.",
                "- Boundary behavior follows Pillow bilinear resize on finite value and weight rasters.",
                "- Native descriptive metrics and shared-mask control metrics are stored in separate fields.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_review_docs(output_root: Path, rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    cand = output_root / "candidate_manuscript_materials_not_integrated"
    cand.mkdir(parents=True, exist_ok=True)
    (cand / "OPENMVS_MANUSCRIPT_CANDIDATE_INTEGRATION_PLAN.md").write_text(
        "# NOT INTEGRATED / PENDING GPT REVIEW\n\n"
        "OpenMVS and DA3 may only be discussed as complementary diagnostic proxies after GPT review. "
        "OpenMVS is a conventional diagnostic proxy under fixed benchmark COLMAP poses. "
        "DA3 is a fixed-pose neural diagnostic proxy. Neither is ground truth or absolute geometry accuracy.\n",
        encoding="utf-8",
    )
    write_csv(
        cand / "OPENMVS_METHOD_EVALUATION_OUTPUT_INVENTORY.csv",
        [
            {"artifact": "OPENMVS_DA3_OVERLAP_CALIBRATION_CORRECTED.csv", "role": "corrected shared-mask proxy agreement metrics", "integrated": "false"},
            {"artifact": "OPENMVS_SUPPORT_MASK_SENSITIVITY_SUMMARY.csv", "role": "support-mask robustness sensitivity", "integrated": "false"},
            {"artifact": "candidate_manuscript_materials_not_integrated/", "role": "candidate text only", "integrated": "false"},
        ],
    )
    (output_root / "CHUNYA_EXTENT_AUDIT_RESOLUTION.md").write_text(
        "# Chunya Extent Audit Resolution\n\n"
        "Status: `extent_audit_resolved_sparse_outlier_inflation`.\n\n"
        "The earlier low raw mesh/sparse bbox ratio for `chunya_01_20260526_1021` is treated as sparse-outlier inflation in the raw sparse extent, not as a downgrade of the OpenMVS engineering qualification. "
        "The scene remains under the same automatic engineering-gate scope. This note does not claim metric ground truth.\n",
        encoding="utf-8",
    )
    write_csv(
        output_root / "HUMAN_VISUAL_ASSESSMENT_SCOPE.csv",
        [
            {
                "human_visual_assessment_scope": "global_only",
                "per_scene_visual_label": "not_available",
                "notes": "User provided a coarse global CloudCompare pass; no per-scene labels are invented.",
            }
        ],
    )
    (output_root / "MANUSCRIPT_UNCHANGED_STATEMENT.md").write_text(
        "# Manuscript Unchanged Statement\n\n"
        "This correction stage did not modify the manuscript, supplement, benchmark metrics, UMGS checkpoints, Gaussian support, OpenMVS reconstruction outputs, or DA3 inference outputs.\n",
        encoding="utf-8",
    )
    (output_root / "EXECUTIVE_DECISION_SUMMARY.md").write_text(
        "# Executive Decision Summary\n\n"
        "This package corrects the OpenMVS--DA3 overlap evaluator by separating native descriptive metrics from shared-mask negative-control comparisons. "
        "True-vs-shuffle and true-vs-mirror statements now use identical pixel sets. Mirror is reported as spatial sensitivity only. "
        "Papaya legacy Stage-2b outputs are marked non-poolable with corrected Maize/Road pilot outputs.\n\n"
        f"- Corrected metric rows: {len(rows)}\n"
        f"- Scene summary rows: {len(scene_rows)}\n"
        f"- Skipped manifest rows: {len(skipped)}\n",
        encoding="utf-8",
    )
    (output_root / "GPT_REVIEW_REQUEST.md").write_text(
        "# GPT Review Request\n\n"
        "Please review this corrected OpenMVS--DA3 overlap package. The key correction is that native descriptive metrics are separated from shared-mask true-vs-control comparisons, and all negative-control separation/taxonomy uses shared-mask metrics only.\n\n"
        "No OpenMVS reconstruction, DA3 inference, UMGS rendering/training, benchmark metric, checkpoint, support, manuscript, or supplement file was changed.\n",
        encoding="utf-8",
    )
    manifest = [{"path": str(p.relative_to(output_root)), "bytes": p.stat().st_size} for p in sorted(output_root.rglob("*")) if p.is_file()]
    write_csv(output_root / "PACKAGE_MANIFEST.csv", manifest)
    (output_root / "PACKAGE_MANIFEST.md").write_text(
        "# Package Manifest\n\n" + "\n".join(f"- `{m['path']}` ({m['bytes']} bytes)" for m in manifest[:700]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output_root / "OPENMVS_DA3_OVERLAP_CORRECTED_PROTOCOL.json",
        {
            "schema": "openmvs_da3_overlap_corrected_protocol_v1",
            "shuffle_seed": SHUFFLE_SEED,
            "min_shared_pixels": MIN_SHARED_PIXELS,
            "min_shared_coverage": MIN_SHARED_COVERAGE,
            "resize_min_weight": RESIZE_MIN_WEIGHT,
            "highgrad_quantile": HIGHGRAD_QUANTILE,
            "claim_boundary": "proxy agreement only; not ground truth and not absolute geometry accuracy",
            "taxonomy": [
                "insufficient_common_support",
                "strong_proxy_agreement",
                "proxy_contradiction",
                "weak_or_mixed_proxy_agreement",
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--support-root", required=True, type=Path)
    parser.add_argument("--da3-multiscene-root", required=True, type=Path)
    parser.add_argument("--papaya-stage2b-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.input_manifest or (args.output_root / "OPENMVS_DA3_OVERLAP_INPUT_MANIFEST.csv")
    if args.input_manifest and args.input_manifest.exists():
        manifest = load_manifest(args.input_manifest)
    else:
        manifest = build_input_manifest(
            manifest_path,
            campaign_root=args.campaign_root,
            support_root=args.support_root,
            da3_multiscene_root=args.da3_multiscene_root,
            papaya_stage2b_root=args.papaya_stage2b_root,
        )
    write_resize_semantics(args.output_root / "OPENMVS_DA3_OVERLAP_RESIZE_SEMANTICS.md")
    rows, support_rows, skipped = evaluate_manifest(manifest, args.output_root)
    scene_rows = scene_summary(rows)
    write_csv(args.output_root / "OPENMVS_DA3_OVERLAP_CALIBRATION_CORRECTED.csv", rows)
    write_csv(args.output_root / "OPENMVS_DA3_OVERLAP_SCENE_SUMMARY_CORRECTED.csv", scene_rows)
    write_csv(args.output_root / "OPENMVS_DA3_OVERLAP_CONTROL_SUPPORT_AUDIT.csv", support_rows)
    write_csv(args.output_root / "OPENMVS_DA3_OVERLAP_SKIPPED_CORRECTED.csv", skipped)
    build_review_docs(args.output_root, rows, scene_rows, skipped)


if __name__ == "__main__":
    main()
