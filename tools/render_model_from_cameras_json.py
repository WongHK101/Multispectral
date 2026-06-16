import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import torch
import torchvision

from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def _load_cameras(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list):
        raise ValueError(f"Expected a list of camera entries in {path}")
    return cameras


def _parse_selection(selection: str, cameras: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not selection or selection.strip().lower() in {"all", "*"}:
        return cameras

    by_img = {str(c.get("img_name", "")): c for c in cameras}
    by_stem = {Path(str(c.get("img_name", ""))).stem: c for c in cameras}
    selected = []
    for token in [t.strip() for t in selection.split(",") if t.strip()]:
        cam = None
        if token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(cameras):
                raise IndexError(f"Camera index {idx} out of range for {len(cameras)} cameras")
            cam = cameras[idx]
        elif token in by_img:
            cam = by_img[token]
        elif token in by_stem:
            cam = by_stem[token]
        else:
            matches = [c for c in cameras if token in str(c.get("img_name", ""))]
            if len(matches) == 1:
                cam = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Selection token {token!r} matched multiple cameras")
            else:
                raise KeyError(f"Selection token {token!r} did not match any camera")
        selected.append(cam)
    return selected


def _read_sh_degree(model_path: Path, default: int) -> int:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        return default
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    for key in ("sh_degree",):
        marker = f"{key}="
        if marker in text:
            tail = text.split(marker, 1)[1]
            value = tail.split(",", 1)[0].split(")", 1)[0].strip()
            try:
                return int(value)
            except ValueError:
                pass
    return default


def _find_ply(model_path: Path, iteration: int) -> Path:
    point_root = model_path / "point_cloud"
    if iteration < 0:
        candidates = []
        for child in point_root.glob("iteration_*"):
            try:
                candidates.append((int(child.name.split("_")[-1]), child))
            except Exception:
                continue
        if not candidates:
            raise FileNotFoundError(f"No point_cloud/iteration_* found under {model_path}")
        iteration, iter_dir = max(candidates, key=lambda x: x[0])
    else:
        iter_dir = point_root / f"iteration_{iteration}"
    ply = iter_dir / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"PLY not found: {ply}")
    return ply


def _camera_to_minicam(entry: Dict[str, object], max_width: int, scale: float) -> MiniCam:
    width = int(entry["width"])
    height = int(entry["height"])
    fx = float(entry["fx"])
    fy = float(entry["fy"])
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    if max_width and max_width > 0 and width > max_width:
        scale = min(scale, float(max_width) / float(width))
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(entry["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(entry["position"], dtype=np.float64)
    rt = np.linalg.inv(c2w)
    r = rt[:3, :3].T
    t = rt[:3, 3]

    znear, zfar = 0.01, 100.0
    world_view = torch.tensor(getWorld2View2(r, t), dtype=torch.float32).transpose(0, 1).cuda()
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(out_width, out_height, fovy, fovx, znear, zfar, world_view, full_proj)


def render_model_from_cameras(
    model_path: Path,
    cameras_json: Path,
    out_dir: Path,
    selection: str,
    iteration: int,
    max_width: int,
    scale: float,
    sh_degree: int,
    white_background: bool,
) -> None:
    cameras = _load_cameras(cameras_json)
    selected = _parse_selection(selection, cameras)
    if not selected:
        raise ValueError("No cameras selected")

    ply = _find_ply(model_path, iteration)
    resolved_sh = _read_sh_degree(model_path, sh_degree)
    gaussians = GaussianModel(resolved_sh)
    gaussians.load_ply(str(ply))

    pipe = SimpleNamespace(
        debug=False,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        antialiasing=False,
    )
    bg = torch.tensor([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_path": str(model_path),
        "cameras_json": str(cameras_json),
        "ply": str(ply),
        "selected_count": len(selected),
        "max_width": max_width,
        "scale": scale,
        "sh_degree": resolved_sh,
        "white_background": white_background,
        "outputs": [],
    }

    with torch.no_grad():
        for idx, entry in enumerate(selected):
            cam = _camera_to_minicam(entry, max_width=max_width, scale=scale)
            result = render(cam, gaussians, pipe, bg, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            image_name = str(entry.get("img_name", f"camera_{idx:05d}"))
            stem = Path(image_name).stem or f"camera_{idx:05d}"
            out_path = out_dir / f"{idx:03d}_{stem}.png"
            torchvision.utils.save_image(result, str(out_path))
            manifest["outputs"].append({
                "camera_index_in_selection": idx,
                "camera_id": entry.get("id"),
                "img_name": image_name,
                "width": int(cam.image_width),
                "height": int(cam.image_height),
                "path": str(out_path),
            })

    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a trained/product Gaussian model directly from cameras.json without loading source images."
    )
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--cameras_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--selection", default="all", help="Comma-separated camera indices, image names, stems, substrings, or 'all'.")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max_width", type=int, default=1600)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    render_model_from_cameras(
        model_path=args.model_path.resolve(),
        cameras_json=args.cameras_json.resolve(),
        out_dir=args.out_dir.resolve(),
        selection=args.selection,
        iteration=args.iteration,
        max_width=args.max_width,
        scale=float(args.scale),
        sh_degree=int(args.sh_degree),
        white_background=bool(args.white_background),
    )


if __name__ == "__main__":
    main()
