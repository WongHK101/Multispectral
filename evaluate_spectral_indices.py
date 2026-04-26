import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from utils.validity_mask_utils import load_validity_mask_or_ones


INDEX_REQUIRED_BANDS = {
    "NDVI": ("R", "NIR"),
    "GNDVI": ("G", "NIR"),
    "NDRE": ("RE", "NIR"),
}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate pred-index vs GT-index fidelity for multispectral band renders.")
    ap.add_argument("--g_model_dir", required=True)
    ap.add_argument("--r_model_dir", required=True)
    ap.add_argument("--re_model_dir", required=True)
    ap.add_argument("--nir_model_dir", required=True)
    ap.add_argument("--iteration", type=int, default=60000)
    ap.add_argument("--indices", default="NDVI,GNDVI,NDRE")
    ap.add_argument("--out_json", required=True)
    ap.add_argument(
        "--mask_mode",
        default="gt_nonzero_intersection",
        choices=["gt_nonzero_intersection", "validity_intersection"],
    )
    ap.add_argument("--mask_threshold", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    return ap


def _parse_indices(text: str) -> List[str]:
    out = [x.strip().upper() for x in str(text).replace(",", " ").split() if x.strip()]
    for idx in out:
        if idx not in INDEX_REQUIRED_BANDS:
            raise ValueError(f"Unsupported index {idx!r}; supported={sorted(INDEX_REQUIRED_BANDS)}")
    return out


def _load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _scalar_and_tie_error(path: Path) -> Tuple[np.ndarray, float, float]:
    rgb = _load_rgb01(path)
    c0 = rgb[..., 0]
    err = np.max(np.abs(rgb - c0[..., None]), axis=-1)
    return c0, float(np.mean(err)), float(np.max(err))


def _method_dir(model_dir: Path, iteration: int) -> Path:
    root = model_dir / "test" / f"ours_{int(iteration)}"
    if not root.exists():
        raise FileNotFoundError(f"Missing rendered test directory: {root}")
    return root


def _list_pngs(path: Path) -> List[str]:
    return sorted([f for f in os.listdir(path) if f.lower().endswith(".png")])


def _load_camera_names(model_dir: Path, n_views: int) -> List[str]:
    pj = model_dir / "cameras.json"
    if not pj.exists():
        return ["" for _ in range(n_views)]
    data = json.loads(pj.read_text(encoding="utf-8"))
    if len(data) < n_views:
        raise RuntimeError(f"cameras.json has {len(data)} entries but render has {n_views}: {pj}")
    return [str(d.get("img_name", "")) for d in data[:n_views]]


def _parse_source_path(cfg_path: Path) -> Optional[Path]:
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"source_path\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return None
    return Path(m.group(1))


def _load_validity_mask(model_dir: Path, image_name: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    mask, _, _ = load_validity_mask_or_ones(model_dir / "cfg_args", image_name, shape_hw)
    return mask


def _index_formula(name: str, bands: Mapping[str, np.ndarray], eps: float) -> np.ndarray:
    name = name.upper()
    if name == "NDVI":
        return (bands["NIR"] - bands["R"]) / (bands["NIR"] + bands["R"] + eps)
    if name == "GNDVI":
        return (bands["NIR"] - bands["G"]) / (bands["NIR"] + bands["G"] + eps)
    if name == "NDRE":
        return (bands["NIR"] - bands["RE"]) / (bands["NIR"] + bands["RE"] + eps)
    raise ValueError(f"Unsupported index: {name}")


def _masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, data_range: float = 2.0) -> float:
    if mask.sum() <= 0:
        return float("nan")
    mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(float(data_range)) - 10.0 * math.log10(mse))


def _masked_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, device) -> float:
    import torch
    from utils.loss_utils import ssim

    if mask.sum() <= 0:
        return float("nan")
    pred01 = np.clip((pred + 1.0) * 0.5, 0.0, 1.0)
    gt01 = np.clip((gt + 1.0) * 0.5, 0.0, 1.0)
    m = mask.astype(np.float32)
    pred_t = torch.from_numpy(pred01[None, None, ...].astype(np.float32)).to(device)
    gt_t = torch.from_numpy(gt01[None, None, ...].astype(np.float32)).to(device)
    mask_t = torch.from_numpy(m[None, None, ...]).to(device)
    value = ssim(pred_t * mask_t, gt_t * mask_t)
    return float(value.detach().cpu().item())


def _mean_median(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
    }


def evaluate(args) -> Tuple[Dict[str, object], Dict[str, object]]:
    import torch

    model_dirs = {
        "G": Path(args.g_model_dir).resolve(),
        "R": Path(args.r_model_dir).resolve(),
        "RE": Path(args.re_model_dir).resolve(),
        "NIR": Path(args.nir_model_dir).resolve(),
    }
    indices = _parse_indices(args.indices)
    device = torch.device(args.device)

    artifacts = {}
    ref_files = None
    for band, model_dir in model_dirs.items():
        root = _method_dir(model_dir, int(args.iteration))
        renders_dir = root / "renders"
        gt_dir = root / "gt"
        files = _list_pngs(renders_dir)
        gt_files = _list_pngs(gt_dir)
        if files != gt_files:
            raise RuntimeError(f"Render/GT file mismatch for {band}: {renders_dir} vs {gt_dir}")
        if ref_files is None:
            ref_files = files
        elif files != ref_files:
            raise RuntimeError(f"Rendered file list mismatch for band {band}")
        artifacts[band] = {
            "model_dir": model_dir,
            "renders_dir": renders_dir,
            "gt_dir": gt_dir,
            "image_names": _load_camera_names(model_dir, len(files)),
        }

    summary = {
        "protocol": "pred_index_vs_gt_index",
        "iteration": int(args.iteration),
        "indices": indices,
        "mask_mode": str(args.mask_mode),
        "mask_threshold": float(args.mask_threshold),
        "eps": float(args.eps),
        "model_dirs": {band: str(path) for band, path in model_dirs.items()},
        "view_count": len(ref_files or []),
        "metrics_by_index": {},
        "coverage": {},
        "channel_tie_error": {
            "pred_mean": {},
            "pred_max": {},
            "gt_mean": {},
            "gt_max": {},
        },
    }
    per_view = {
        "protocol": "pred_index_vs_gt_index",
        "iteration": int(args.iteration),
        "indices": {},
    }

    pred_tie_mean = {band: [] for band in model_dirs}
    pred_tie_max = {band: [] for band in model_dirs}
    gt_tie_mean = {band: [] for band in model_dirs}
    gt_tie_max = {band: [] for band in model_dirs}

    for idx_name in indices:
        req_bands = INDEX_REQUIRED_BANDS[idx_name]
        mae_values: List[float] = []
        rmse_values: List[float] = []
        psnr_values: List[float] = []
        ssim_values: List[float] = []
        coverage_values: List[float] = []
        per_view["indices"][idx_name] = {}

        for fname_i, fname in enumerate(ref_files or []):
            pred_bands = {}
            gt_bands = {}
            masks = []
            for band, art in artifacts.items():
                pred, pmean, pmax = _scalar_and_tie_error(art["renders_dir"] / fname)
                gt, gmean, gmax = _scalar_and_tie_error(art["gt_dir"] / fname)
                pred_bands[band] = pred
                gt_bands[band] = gt
                pred_tie_mean[band].append(pmean)
                pred_tie_max[band].append(pmax)
                gt_tie_mean[band].append(gmean)
                gt_tie_max[band].append(gmax)

            for band in req_bands:
                if args.mask_mode == "validity_intersection":
                    img_name = artifacts[band]["image_names"][fname_i]
                    masks.append(_load_validity_mask(artifacts[band]["model_dir"], img_name, gt_bands[band].shape))
                else:
                    masks.append(gt_bands[band] > float(args.mask_threshold))
            mask = masks[0]
            for m in masks[1:]:
                mask = np.logical_and(mask, m)

            pred_idx = _index_formula(idx_name, pred_bands, float(args.eps))
            gt_idx = _index_formula(idx_name, gt_bands, float(args.eps))
            coverage = float(mask.mean())
            coverage_values.append(coverage)

            if mask.sum() <= 0:
                mae = rmse = psnr_v = ssim_v = float("nan")
            else:
                diff = pred_idx - gt_idx
                mae = float(np.mean(np.abs(diff[mask])))
                rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
                psnr_v = _masked_psnr(pred_idx, gt_idx, mask, data_range=2.0)
                ssim_v = _masked_ssim(pred_idx, gt_idx, mask, device=device)
            mae_values.append(mae)
            rmse_values.append(rmse)
            psnr_values.append(psnr_v)
            ssim_values.append(ssim_v)
            per_view["indices"][idx_name][fname] = {
                "MAE": mae,
                "RMSE": rmse,
                "PSNR": psnr_v,
                "SSIM": ssim_v,
                "COVERAGE": coverage,
            }

        summary["metrics_by_index"][idx_name] = {
            "MAE": _mean_median(mae_values),
            "RMSE": _mean_median(rmse_values),
            "PSNR": _mean_median(psnr_values),
            "SSIM": _mean_median(ssim_values),
        }
        summary["coverage"][idx_name] = _mean_median(coverage_values)

    for band in model_dirs:
        summary["channel_tie_error"]["pred_mean"][band] = _mean_median(pred_tie_mean[band])
        summary["channel_tie_error"]["pred_max"][band] = _mean_median(pred_tie_max[band])
        summary["channel_tie_error"]["gt_mean"][band] = _mean_median(gt_tie_mean[band])
        summary["channel_tie_error"]["gt_max"][band] = _mean_median(gt_tie_max[band])

    return summary, per_view


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary, per_view = evaluate(args)
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_view_path = out_json.with_name(out_json.stem + "_per_view.json")
    per_view_path.write_text(json.dumps(per_view, indent=2), encoding="utf-8")
    print(f"[E4b] Index summary: {out_json}")
    print(f"[E4b] Index per-view: {per_view_path}")


if __name__ == "__main__":
    main()
