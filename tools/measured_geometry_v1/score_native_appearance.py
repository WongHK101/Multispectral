"""Common-mask appearance scoring; historical gt_nonzero scores remain separate."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from .common_geometry import unique_rows
from .contracts import read_json, safe_member, sha256, verify_sha
from .score_native_gcp import write_csv, write_json


CHANNELS = ("D", "MS_G", "MS_R", "MS_RE", "MS_NIR")
SOURCE_HASHES = {
    "metrics.py": "a21c020f39909db6874e1136b3f90865433552d98f458d6d0dec89fe80eb192a",
    "evaluate_spectral_indices.py": "627f499983521083e909036fb3370e067977fa85e807a081d791f767ea2f2dae",
    "utils/loss_utils.py": "f27abff3f009edc62b3f336dd36bcae3512fbeee845b07131461e22ea0e4db49",
    "utils/image_utils.py": "92b3b8f43353c09a7afadd348f448d87ba8b3465c5cff8a48fb53c28eabaa3ee",
    "utils/validity_mask_utils.py": "1da61d727c9895956f33dbdb1bfeb97408038c4510909ea0efb5b5834f4a64e4",
    "lpipsPyTorch/__init__.py": "35806c438336e43c00bc7e5edc353c3de03926fc7b58cbb7e40a61678fcef4b5",
    "lpipsPyTorch/modules/lpips.py": "61272ed285c812cb420a3255448b52fd1fdcb14c3bd2c04c16e3fc4b297651d2",
    "lpipsPyTorch/modules/networks.py": "92a21e6eaedd0211f2a535f07855447ff4436cdeff66fecedcc943f827d1be98",
    "lpipsPyTorch/modules/utils.py": "21ea6e3481568028792e94ffcf17196b6be400b36aea09cbd94d1682f2796813",
}
WEIGHTS = {
    "vgg16-397923af.pth": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
    "vgg.pth": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
}


def rgb_shape(array):
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB or scalar image")
    return array


def quantize_gt(array):
    if array.dtype == np.uint8:
        result = array.copy()
    elif array.dtype == np.uint16:
        result = np.clip((array.astype(np.float32) / np.float32(65535)) * np.float32(255), 0, 255).astype(np.uint8)
    else:
        raise ValueError("Expected native uint8 RGB or uint16 scalar GT")
    return rgb_shape(result)


def quantize_prediction(array):
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Expected finite native float32 prediction")
    return rgb_shape(np.clip(array * np.float32(255), 0, 255).astype(np.uint8))


def spectral_metrics(pred4, gt4, mask, eps=1e-6):
    if pred4.shape != gt4.shape or pred4.shape[-1] != 4 or mask.shape != pred4.shape[:2] or not mask.any():
        raise ValueError("Invalid common spectral domain")
    diff = pred4 - gt4
    dot = np.sum(pred4 * gt4, axis=-1)
    norms = np.linalg.norm(pred4, axis=-1) * np.linalg.norm(gt4, axis=-1)
    angle = np.degrees(np.arccos(np.clip(dot / (norms + eps), -1, 1)))
    return {"rmse_4band": float(np.sqrt(np.mean(diff[mask]**2))), "sam_deg": float(np.mean(angle[mask]))}


def module_from(root, filename, module_name):
    if module_name in sys.modules:
        raise ValueError("Ambiguous metric module already imported")
    spec = importlib.util.spec_from_file_location(module_name, root / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(root.resolve() == args.output.resolve() or root.resolve() in args.output.resolve().parents
           for root in (args.export_root, args.data_root, args.runtime_root)):
        raise ValueError("Cannot write metrics into inputs")
    started = time.monotonic()
    verify_sha(args.export_root / "export_manifest.json", args.export_manifest_sha256)
    manifest = read_json(args.export_root / "export_manifest.json")
    verify_sha(args.data_root / "manifest.json", manifest["adapter_manifest_sha256"])
    adapter = read_json(args.data_root / "manifest.json")
    groups = unique_rows([g for g in adapter["groups"] if g["split"] == "eval"], "image_name")
    rows = {(r["image_name"], r["channel"]): r for r in manifest["appearance_views"]}
    if len(rows) != len(manifest["appearance_views"]) or set(rows) != {(n, c) for n in groups for c in CHANNELS}:
        raise ValueError("Incomplete/duplicate heldout appearance population")
    if len(groups) != adapter["counts"]["eval_image_groups"] or manifest["scene"] != adapter["scene_id"]:
        raise ValueError("Heldout scene/count mismatch")
    for name, expected in SOURCE_HASHES.items():
        verify_sha(args.runtime_root / name, expected)
    for name, expected in WEIGHTS.items():
        verify_sha(args.torch_hub / "checkpoints" / name, expected)
    sys.path.insert(0, str(args.runtime_root))
    import torch
    import torchvision
    import lpipsPyTorch
    from utils.loss_utils import ssim
    torch.set_num_threads(2)
    torch.hub.set_dir(str(args.torch_hub))
    metric = module_from(args.runtime_root, "metrics.py", "umgs_frozen_masked_metrics")
    indices = module_from(args.runtime_root, "evaluate_spectral_indices.py", "umgs_frozen_indices")
    # This same immutable VGG/linear network has no stochastic layers; cache only construction.
    criterion = lpipsPyTorch.LPIPS(net_type="vgg", version="0.1").eval().cpu()
    image_rows, product_rows = [], []
    args.output.mkdir(parents=True, exist_ok=False)
    for name, group in sorted(groups.items()):
        mask_record = group["output_masks"]["common"]
        mask_path = safe_member(args.data_root, mask_record["relative_path"])
        verify_sha(mask_path, mask_record["sha256"])
        with Image.open(mask_path) as im:
            mask_array = np.asarray(im)
        if mask_array.dtype != np.uint8 or mask_array.ndim != 2 or not set(np.unique(mask_array)) <= {0, 255}:
            raise ValueError("Expected exact binary common mask")
        mask = mask_array > 0
        if not mask.any():
            raise ValueError("Empty common mask: no fallback to all pixels")
        p_bands, g_bands = {}, {}
        for channel in CHANNELS:
            row = rows[name, channel]
            truth = group["output_images"][channel]
            if row["common_mask"] != mask_record or row["gt_native_file"] != truth["relative_path"] or row["gt_sha256"] != truth["sha256"]:
                raise ValueError("Prediction/GT/common mask binding mismatch")
            p_path = safe_member(args.export_root, row["prediction"])
            g_path = safe_member(args.data_root, truth["relative_path"])
            verify_sha(p_path, row["sha256"])
            verify_sha(g_path, row["gt_sha256"])
            prediction = np.load(p_path, allow_pickle=False)
            if list(prediction.shape) != row["shape"] or str(prediction.dtype) != row["dtype"]:
                raise ValueError("Prediction shape/dtype identity mismatch")
            with Image.open(g_path) as im:
                gt = np.asarray(im)
            if (channel == "D" and (gt.dtype != np.uint8 or gt.ndim != 3 or gt.shape[2] != 3)) or (channel != "D" and (gt.dtype != np.uint16 or gt.ndim != 2)):
                raise ValueError("Wrong native GT channel/type")
            p8, g8 = quantize_prediction(prediction), quantize_gt(gt)
            if p8.shape != g8.shape or p8.shape[:2] != mask.shape:
                raise ValueError("No resizing is allowed at scoring")
            p = p8.astype(np.float32)/np.float32(255)
            g = g8.astype(np.float32)/np.float32(255)
            pt = torch.from_numpy(p.transpose(2, 0, 1).copy())[None]
            gt_t = torch.from_numpy(g.transpose(2, 0, 1).copy())[None]
            mt = torch.from_numpy(mask.astype(np.float32))[None, None]
            with torch.no_grad():
                psnr = metric.masked_psnr(pt, gt_t, mt)
                ssim_v = float(ssim(pt*mt, gt_t*mt))
                lpips_v = float(criterion(pt*mt, gt_t*mt))
            if not np.isfinite([psnr, ssim_v, lpips_v]).all():
                raise ValueError("Nonfinite appearance metric")
            image_rows.append({"image_name": name, "channel": channel, "PSNR": psnr, "SSIM": ssim_v,
                "LPIPS": lpips_v, "valid_pixels": int(mask.sum()), "total_pixels": mask.size,
                "coverage": float(mask.mean()), "prediction_sha256": row["sha256"], "gt_sha256": row["gt_sha256"],
                "mask_sha256": mask_record["sha256"]})
            if channel != "D":
                p_bands[channel[3:]], g_bands[channel[3:]] = p[..., 0], g[..., 0]
        ordered = ["G", "R", "RE", "NIR"]
        product = {"image_name": name, "coverage": float(mask.mean()),
            **spectral_metrics(np.stack([p_bands[b] for b in ordered], -1), np.stack([g_bands[b] for b in ordered], -1), mask)}
        for index in ("NDVI", "GNDVI", "NDRE"):
            pi, gi = indices._index_formula(index, p_bands, 1e-6), indices._index_formula(index, g_bands, 1e-6)
            product[index+"_RMSE"] = float(np.sqrt(np.mean((pi[mask]-gi[mask])**2)))
            product[index+"_MAE"] = float(np.mean(np.abs(pi[mask]-gi[mask])))
            with torch.no_grad():
                product[index+"_SSIM"] = indices._masked_ssim(pi, gi, mask, torch.device("cpu"))
        product_rows.append(product)
        print(json.dumps({"image_name": name, "completed_views": len(product_rows), "total_views": len(groups)}), flush=True)
    band_summary = {c: {k: float(torch.tensor([r[k] for r in image_rows if r["channel"] == c], dtype=torch.float32).mean())
                        for k in ("PSNR", "SSIM", "LPIPS")} for c in CHANNELS}
    product_summary = {k: float(np.mean([r[k] for r in product_rows], dtype=np.float64))
                       for k in product_rows[0] if k != "image_name"}
    write_csv(args.output / "per_image.csv", image_rows)
    write_csv(args.output / "spectral_products.csv", product_rows)
    result = {"schema": "umgs_common_mask_appearance_result_v1", "scene": manifest["scene"], "method": manifest["method"],
        "status": "COMPLETE_PENDING_STAGE_REVIEW", "whole_matrix_row_complete": False,
        "heldout_capture_count": len(groups), "band_frame_count": len(image_rows), "bands": band_summary, "products": product_summary,
        "mask": "fixed_adapter_common_mask_no_gt_nonzero_or_prediction_filter", "psnr_denominator": "valid_pixels_times_channels",
        "ssim_lpips_mask": "zero_outside_mask_then_original_full_frame_function",
        "quantization": "native_float32_times_255_clip_astype_uint8_truncation; uint16_GT_divide_float32_65535_first",
        "lpips_inputs": "RGB_0_to_1_no_extra_minus1_transform; scalar_triplicated",
        "lpips_network": "VGG_v0.1_IMAGENET1K_V1_cached_eval_instance",
        "spectral_epsilon": 1e-6, "aggregation": "per_image_then_equal_scene_mean; legacy_float32_band_mean_float64_product_mean",
        "legacy_gt_nonzero_scores_used": False, "source_hashes": SOURCE_HASHES, "weight_hashes": WEIGHTS,
        "weight_identity_evidence": "verified_current_cache_not_claimed_as_historical_run_hash_audit",
        "export_manifest_sha256": args.export_manifest_sha256, "adapter_manifest_sha256": manifest["adapter_manifest_sha256"],
        "software": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
        "gpu_used": False, "script_sha256": sha256(__file__), "wall_seconds": time.monotonic()-started}
    write_json(args.output / "summary.json", result)
    write_json(args.output / "output_hashes.json", [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
               for p in sorted(args.output.iterdir()) if p.is_file()])
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export-root", "data-root", "runtime-root", "torch-hub", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--export-manifest-sha256", required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
