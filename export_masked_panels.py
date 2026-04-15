import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _find_latest_iteration_dir(model_path: Path, split: str) -> Tuple[int, Path]:
    split_dir = model_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    candidates = []
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


def _to_u8(arr01: np.ndarray) -> np.ndarray:
    return np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


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
    names = [str(c.get("img_name", "")) for c in cams[:n_views]]
    if any(not n for n in names):
        raise RuntimeError("Found empty img_name in cameras.json for test views")
    return names


def _pick_indices(n: int, num: int) -> List[int]:
    if n <= 0:
        return []
    num = max(1, min(num, n))
    if num == 1:
        return [n // 2]
    return sorted(set(int(round(i * (n - 1) / (num - 1))) for i in range(num)))


def _make_panel(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask3 = mask[..., None]
    masked_render = render * mask3
    masked_gt = gt * mask3
    abs_diff = np.abs(masked_render - masked_gt)
    mask_viz = np.repeat(mask3, 3, axis=2)
    top = np.concatenate([render, gt, mask_viz], axis=1)
    bottom = np.concatenate([masked_render, masked_gt, abs_diff], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main():
    ap = argparse.ArgumentParser("Export masked visual panels for rectified band models")
    ap.add_argument("-m", "--model_path", required=True, type=str)
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--mask_dir", default="", type=str)
    ap.add_argument("--num_views", default=8, type=int)
    ap.add_argument("--out_dir", default="", type=str)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    resolved_iter, renders_dir, gt_dir = _resolve_dirs(model_path, args.split, args.iteration)
    resolved_mask_dir = _resolve_mask_dir(model_path, args.mask_dir)

    render_files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])
    if render_files != gt_files:
        raise RuntimeError("render/gt file lists mismatch")
    if not render_files:
        raise RuntimeError(f"No png files found in {renders_dir}")

    image_names = _load_test_image_names(model_path, len(render_files))
    out_dir = Path(args.out_dir) if args.out_dir else model_path / f"masked_panels_{args.split}_ours_{resolved_iter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = _pick_indices(len(render_files), args.num_views)
    manifest = []
    for idx in indices:
        fname = render_files[idx]
        image_name = image_names[idx]
        mask_path = resolved_mask_dir / f"{Path(image_name).stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_name}: {mask_path}")

        render = _load_rgb(renders_dir / fname)
        gt = _load_rgb(gt_dir / fname)
        if render.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch: {fname}, render={render.shape}, gt={gt.shape}")
        mask = _load_mask(mask_path, target_hw=render.shape[:2])
        panel = _make_panel(render, gt, mask)

        out_name = f"{idx:05d}_{Path(image_name).stem}.png"
        out_path = out_dir / out_name
        Image.fromarray(_to_u8(panel)).save(out_path)
        manifest.append(
            {
                "index": idx,
                "render_file": fname,
                "image_name": image_name,
                "mask_path": str(mask_path),
                "panel_path": str(out_path),
                "valid_ratio": float(mask.mean()),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "split": args.split,
                "iteration": resolved_iter,
                "mask_dir": str(resolved_mask_dir),
                "num_views": len(indices),
                "panels": manifest,
                "panel_layout": "2x3: [render|gt|mask ; masked_render|masked_gt|abs_diff]",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PANELS] model={model_path}")
    print(f"[PANELS] split={args.split} iter={resolved_iter} selected_views={len(indices)}")
    print(f"[PANELS] out_dir={out_dir}")
    print(f"[PANELS] manifest={manifest_path}")


if __name__ == "__main__":
    main()

