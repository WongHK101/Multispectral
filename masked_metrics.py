import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _find_latest_iteration_dir(model_path: Path, split: str) -> Tuple[int, Path]:
    split_dir = model_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    candidates: List[Tuple[int, Path]] = []
    for item in split_dir.iterdir():
        if not item.is_dir():
            continue
        m = re.match(r"ours_(\d+)$", item.name)
        if m:
            candidates.append((int(m.group(1)), item))
    if not candidates:
        raise FileNotFoundError(f"No ours_* directory found under: {split_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def _parse_source_path_from_cfg(cfg_args_path: Path) -> str:
    txt = cfg_args_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"source_path='([^']+)'", txt)
    if m:
        return m.group(1)
    m = re.search(r'source_path="([^"]+)"', txt)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse source_path from {cfg_args_path}")


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    mask_img = Image.open(path).convert("L")
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def _masked_mse(render: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(mask.sum()) * 3.0
    if denom <= eps:
        return float("nan")
    diff2 = (render - gt) ** 2
    return float((diff2 * mask[..., None]).sum() / denom)


def _masked_mae(render: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(mask.sum()) * 3.0
    if denom <= eps:
        return float("nan")
    diff = np.abs(render - gt)
    return float((diff * mask[..., None]).sum() / denom)


def _psnr_from_mse(mse: float, eps: float = 1e-12) -> float:
    if not np.isfinite(mse):
        return float("nan")
    mse = max(float(mse), eps)
    return float(-10.0 * math.log10(mse))


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _resolve_dirs(model_path: Path, split: str, iteration: int) -> Tuple[int, Path, Path]:
    if iteration < 0:
        resolved_iter, ours_dir = _find_latest_iteration_dir(model_path, split)
    else:
        resolved_iter = int(iteration)
        ours_dir = model_path / split / f"ours_{resolved_iter}"
    renders_dir = ours_dir / "renders"
    gt_dir = ours_dir / "gt"
    if not renders_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt under {ours_dir}")
    return resolved_iter, renders_dir, gt_dir


def _resolve_mask_dir(model_path: Path, mask_dir: str) -> Path:
    if mask_dir:
        p = Path(mask_dir)
        if not p.exists():
            raise FileNotFoundError(f"Mask dir not found: {p}")
        return p
    cfg_args = model_path / "cfg_args"
    if not cfg_args.exists():
        raise FileNotFoundError(f"cfg_args not found in {model_path}")
    source_path = _parse_source_path_from_cfg(cfg_args)
    p = Path(source_path) / "validity_masks"
    if not p.exists():
        raise FileNotFoundError(f"Auto mask dir not found: {p}")
    return p


def _load_test_image_names(model_path: Path, n_views: int) -> List[str]:
    cameras_json = model_path / "cameras.json"
    if not cameras_json.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_json}")
    cams = json.loads(cameras_json.read_text(encoding="utf-8"))
    if len(cams) < n_views:
        raise RuntimeError(f"cameras.json entries ({len(cams)}) < rendered views ({n_views})")
    # Scene writes cameras.json as [test cams..., train cams...]. For split=test we take first n_views.
    names = [str(c.get("img_name", "")) for c in cams[:n_views]]
    if any(not n for n in names):
        raise RuntimeError("Found empty img_name in cameras.json for test views")
    return names


def evaluate(model_path: Path, split: str, iteration: int, mask_dir: str) -> Dict:
    if split != "test":
        raise ValueError("masked_metrics.py currently supports split='test' only")

    resolved_iter, renders_dir, gt_dir = _resolve_dirs(model_path, split, iteration)
    resolved_mask_dir = _resolve_mask_dir(model_path, mask_dir)

    render_files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])
    if render_files != gt_files:
        raise RuntimeError("render/gt file lists mismatch")
    if not render_files:
        raise RuntimeError(f"No png files found in {renders_dir}")

    image_names = _load_test_image_names(model_path, len(render_files))

    per_view = {}
    m_mse_vals: List[float] = []
    m_mae_vals: List[float] = []
    m_psnr_vals: List[float] = []
    valid_ratios: List[float] = []

    for idx, fname in enumerate(render_files):
        render = _load_rgb(renders_dir / fname)
        gt = _load_rgb(gt_dir / fname)
        if render.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch: {fname}, render={render.shape}, gt={gt.shape}")

        image_name = image_names[idx]
        mask_path = resolved_mask_dir / f"{Path(image_name).stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_name}: {mask_path}")

        mask = _load_mask(mask_path, target_hw=render.shape[:2])
        m_mse = _masked_mse(render, gt, mask)
        m_mae = _masked_mae(render, gt, mask)
        m_psnr = _psnr_from_mse(m_mse)
        valid_ratio = float(mask.mean())

        m_mse_vals.append(m_mse)
        m_mae_vals.append(m_mae)
        m_psnr_vals.append(m_psnr)
        valid_ratios.append(valid_ratio)

        per_view[fname] = {
            "image_name": image_name,
            "mask_path": str(mask_path),
            "valid_ratio": valid_ratio,
            "masked_mse": m_mse,
            "masked_mae": m_mae,
            "masked_psnr": m_psnr,
        }

    summary = {
        "model_path": str(model_path),
        "split": split,
        "iteration": resolved_iter,
        "mask_dir": str(resolved_mask_dir),
        "num_views": len(render_files),
        "valid_ratio": _mean_std(valid_ratios),
        "masked_mse": _mean_std(m_mse_vals),
        "masked_mae": _mean_std(m_mae_vals),
        "masked_psnr": _mean_std(m_psnr_vals),
    }
    return {"summary": summary, "per_view": per_view}


def main():
    ap = argparse.ArgumentParser("Masked offline metrics for rectified band models")
    ap.add_argument("-m", "--model_path", required=True, type=str)
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--mask_dir", default="", type=str, help="Optional; defaults to <source_path>/validity_masks from cfg_args")
    ap.add_argument("--out_json", default="", type=str)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    result = evaluate(model_path=model_path, split=args.split, iteration=args.iteration, mask_dir=args.mask_dir)

    out_json = Path(args.out_json) if args.out_json else model_path / f"masked_results_{args.split}.json"
    out_per_view = model_path / f"masked_per_view_{args.split}.json"

    out_json.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    out_per_view.write_text(json.dumps(result["per_view"], indent=2), encoding="utf-8")

    s = result["summary"]
    print(f"[MASKED] model={s['model_path']}")
    print(f"[MASKED] split={s['split']} iter={s['iteration']} views={s['num_views']}")
    print(f"[MASKED] valid_ratio(mean±std): {s['valid_ratio']['mean']:.6f} ± {s['valid_ratio']['std']:.6f}")
    print(f"[MASKED] masked_psnr(mean±std): {s['masked_psnr']['mean']:.6f} ± {s['masked_psnr']['std']:.6f}")
    print(f"[MASKED] masked_mae(mean±std):  {s['masked_mae']['mean']:.6f} ± {s['masked_mae']['std']:.6f}")
    print(f"[MASKED] masked_mse(mean±std):  {s['masked_mse']['mean']:.6f} ± {s['masked_mse']['std']:.6f}")
    print(f"[MASKED] summary_json={out_json}")
    print(f"[MASKED] per_view_json={out_per_view}")


if __name__ == "__main__":
    main()

