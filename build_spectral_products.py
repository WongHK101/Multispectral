import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

from utils.sh_utils import C0


FALSE_COLOR_PRODUCTS = {
    "false_color_nir_r_g": ("NIR", "R", "G"),
    "false_color_re_nir_r": ("RE", "NIR", "R"),
    "false_color_nir_re_g": ("NIR", "RE", "G"),
}


def _load_vertex(path: Path):
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"No vertex element in: {path}")
    return ply, ply["vertex"].data


def _rewrite_model_path(cfg_text: str, new_model_path: str) -> str:
    new_model_path = str(new_model_path).replace("\\", "/")

    def repl(match):
        quote = match.group(2)
        return f"{match.group(1)}{quote}{new_model_path}{quote}"

    out, count = re.subn(r"(model_path\s*=\s*)(['\"])[^'\"]+(['\"])", repl, cfg_text, count=1)
    if count == 0:
        text = cfg_text.strip()
        if text.startswith("Namespace(") and text.endswith(")"):
            out = text[:-1] + f", model_path='{new_model_path}')"
        else:
            out = cfg_text
    return out


def _resolve_iteration(model_dir: Path, iteration: int) -> int:
    if iteration is not None:
        return int(iteration)
    point_cloud_root = model_dir / "point_cloud"
    candidates = []
    for child in point_cloud_root.glob("iteration_*"):
        try:
            candidates.append(int(child.name.split("_")[-1]))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"No point_cloud/iteration_* found under: {model_dir}")
    return max(candidates)


def _resolve_ply(model_dir: Path, iteration: int) -> Path:
    resolved_iter = _resolve_iteration(model_dir, iteration)
    ply_path = model_dir / "point_cloud" / f"iteration_{resolved_iter}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    return ply_path


def _field_groups(vertex) -> Tuple[List[str], List[str], List[str], List[str]]:
    names = list(vertex.dtype.names or [])
    f_dc = sorted([n for n in names if n.startswith("f_dc_")], key=lambda x: int(x.split("_")[-1]))
    f_rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda x: int(x.split("_")[-1]))
    scale = sorted([n for n in names if n.startswith("scale_")], key=lambda x: int(x.split("_")[-1]))
    rot = sorted([n for n in names if n.startswith("rot_")], key=lambda x: int(x.split("_")[-1]))
    return f_dc, f_rest, scale, rot


def _stack_fields(vertex, names: List[str]) -> np.ndarray:
    if not names:
        return np.zeros((len(vertex), 0), dtype=np.float32)
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def _dc_coeff_to_scalar(dc_coeff: np.ndarray) -> np.ndarray:
    return np.clip(dc_coeff * C0 + 0.5, 0.0, 1.0)


def _rgb_to_dc_coeff(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def _extract_model_payload(model_dir: Path, iteration: int) -> Dict[str, object]:
    ply_path = _resolve_ply(model_dir, iteration)
    ply, vertex = _load_vertex(ply_path)
    f_dc_names, f_rest_names, scale_names, rot_names = _field_groups(vertex)
    if len(f_dc_names) != 3:
        raise ValueError(f"Expected exactly 3 f_dc fields in: {ply_path}")
    if len(f_rest_names) % 3 != 0:
        raise ValueError(f"Expected f_rest_* count divisible by 3 in: {ply_path}")

    dc_channels = _stack_fields(vertex, f_dc_names)
    rest_channels = _stack_fields(vertex, f_rest_names).reshape(len(vertex), 3, -1)
    dc_coeff = dc_channels.mean(axis=1)
    rest_coeff = rest_channels.mean(axis=1)

    return {
        "model_dir": model_dir,
        "iteration": _resolve_iteration(model_dir, iteration),
        "ply": ply,
        "vertex": vertex,
        "ply_path": ply_path,
        "f_dc_names": f_dc_names,
        "f_rest_names": f_rest_names,
        "scale_names": scale_names,
        "rot_names": rot_names,
        "dc_coeff": dc_coeff,
        "dc_value": _dc_coeff_to_scalar(dc_coeff),
        "rest_coeff": rest_coeff,
    }


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _assert_topology_consistency(models: Dict[str, Dict[str, object]], require_opacity_match: bool, tol: float) -> None:
    names = list(models.keys())
    ref = models[names[0]]
    ref_vertex = ref["vertex"]
    xyz_fields = ["x", "y", "z"]
    ref_xyz = _stack_fields(ref_vertex, xyz_fields)
    ref_scale = _stack_fields(ref_vertex, ref["scale_names"])
    ref_rot = _stack_fields(ref_vertex, ref["rot_names"])
    ref_opacity = np.asarray(ref_vertex["opacity"], dtype=np.float32)

    for name in names[1:]:
        cur = models[name]
        cur_vertex = cur["vertex"]
        if len(cur_vertex) != len(ref_vertex):
            raise ValueError(f"Gaussian count mismatch: ref={len(ref_vertex)} vs {name}={len(cur_vertex)}")
        xyz_diff = _max_abs_diff(ref_xyz, _stack_fields(cur_vertex, xyz_fields))
        scale_diff = _max_abs_diff(ref_scale, _stack_fields(cur_vertex, cur["scale_names"]))
        rot_diff = _max_abs_diff(ref_rot, _stack_fields(cur_vertex, cur["rot_names"]))
        if xyz_diff > tol:
            raise ValueError(f"xyz mismatch for {name}: max_abs_diff={xyz_diff}")
        if scale_diff > tol:
            raise ValueError(f"scale mismatch for {name}: max_abs_diff={scale_diff}")
        if rot_diff > tol:
            raise ValueError(f"rotation mismatch for {name}: max_abs_diff={rot_diff}")
        if require_opacity_match:
            opacity_diff = _max_abs_diff(ref_opacity, np.asarray(cur_vertex["opacity"], dtype=np.float32))
            if opacity_diff > tol:
                raise ValueError(f"opacity mismatch for {name}: max_abs_diff={opacity_diff}")


def _make_colormap(values: np.ndarray) -> np.ndarray:
    stops = np.array([
        [0.00, 0.20, 0.16, 0.34],
        [0.35, 0.68, 0.45, 0.16],
        [0.50, 0.95, 0.90, 0.25],
        [0.70, 0.26, 0.63, 0.25],
        [1.00, 0.08, 0.39, 0.14],
    ], dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    rgb = np.empty((values.shape[0], 3), dtype=np.float32)
    for c in range(3):
        rgb[:, c] = np.interp(values, stops[:, 0], stops[:, c + 1])
    return rgb


def _write_product_model(out_dir: Path, out_iter: int, reference_model_dir: Path, reference_vertex,
                         vertex_out, text_mode: bool) -> None:
    point_cloud_dir = out_dir / "point_cloud" / f"iteration_{out_iter}"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex_out, "vertex")], text=text_mode).write(str(point_cloud_dir / "point_cloud.ply"))

    for aux_name in ("cameras.json", "input.ply", "exposure.json"):
        src = reference_model_dir / aux_name
        if src.exists():
            shutil.copy2(src, out_dir / aux_name)

    cfg_path = reference_model_dir / "cfg_args"
    if cfg_path.exists():
        cfg_text = cfg_path.read_text(encoding="utf-8", errors="ignore")
        (out_dir / "cfg_args").write_text(_rewrite_model_path(cfg_text, str(out_dir)), encoding="utf-8")

    summary = {
        "out_iter": out_iter,
        "gaussian_count": int(len(reference_vertex)),
    }
    (out_dir / "product_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_false_color_vertex(reference_payload: Dict[str, object], models: Dict[str, Dict[str, object]], mapping: Tuple[str, str, str]):
    ref_vertex = reference_payload["vertex"]
    out = ref_vertex.copy()
    f_dc_names = reference_payload["f_dc_names"]
    f_rest_names = reference_payload["f_rest_names"]
    sh_coeff_count = len(f_rest_names) // 3

    for channel_idx, band_name in enumerate(mapping):
        band_payload = models[band_name]
        out[f_dc_names[channel_idx]] = np.asarray(band_payload["dc_coeff"], dtype=out.dtype[f_dc_names[channel_idx]])
        start = channel_idx * sh_coeff_count
        end = start + sh_coeff_count
        coeff_block = np.asarray(band_payload["rest_coeff"], dtype=np.float32)
        for coeff_idx, field_name in enumerate(f_rest_names[start:end]):
            out[field_name] = coeff_block[:, coeff_idx].astype(out.dtype[field_name])
    return out


def _build_index_vertex(reference_payload: Dict[str, object], rgb_values: np.ndarray):
    ref_vertex = reference_payload["vertex"]
    out = ref_vertex.copy()
    f_dc_names = reference_payload["f_dc_names"]
    f_rest_names = reference_payload["f_rest_names"]
    dc_coeff = _rgb_to_dc_coeff(rgb_values.astype(np.float32))
    for idx, field_name in enumerate(f_dc_names):
        out[field_name] = dc_coeff[:, idx].astype(out.dtype[field_name])
    for field_name in f_rest_names:
        out[field_name] = np.zeros(len(out), dtype=out.dtype[field_name])
    return out


def build_products(model_dirs: Dict[str, Path], iterations: Dict[str, int], out_root: Path, savi_l: float,
                   eps: float, require_opacity_match: bool, tol: float) -> None:
    payloads = {band: _extract_model_payload(model_dir, iterations.get(band)) for band, model_dir in model_dirs.items()}
    _assert_topology_consistency(payloads, require_opacity_match=require_opacity_match, tol=tol)

    ref_name = "G" if "G" in payloads else next(iter(payloads.keys()))
    ref_payload = payloads[ref_name]
    ref_model_dir = Path(ref_payload["model_dir"])
    out_iter = int(ref_payload["iteration"])

    out_root.mkdir(parents=True, exist_ok=True)
    for product_name, mapping in FALSE_COLOR_PRODUCTS.items():
        vertex_out = _build_false_color_vertex(ref_payload, payloads, mapping)
        _write_product_model(out_root / product_name, out_iter, ref_model_dir, ref_payload["vertex"], vertex_out, getattr(ref_payload["ply"], "text", False))

    nir = payloads["NIR"]["dc_value"]
    red = payloads["R"]["dc_value"]
    green = payloads["G"]["dc_value"]
    red_edge = payloads["RE"]["dc_value"]
    indices = {
        "ndvi": (nir - red) / (nir + red + eps),
        "ndre": (nir - red_edge) / (nir + red_edge + eps),
        "gndvi": (nir - green) / (nir + green + eps),
        "savi": ((nir - red) / (nir + red + savi_l + eps)) * (1.0 + savi_l),
        "osavi": (nir - red) / (nir + red + 0.16 + eps),
    }

    for name, values in indices.items():
        gray01 = np.clip((values + 1.0) * 0.5, 0.0, 1.0)
        gray_rgb = np.repeat(gray01[:, None], 3, axis=1)
        pseudo_rgb = _make_colormap(gray01)
        gray_vertex = _build_index_vertex(ref_payload, gray_rgb)
        pseudo_vertex = _build_index_vertex(ref_payload, pseudo_rgb)
        _write_product_model(out_root / f"{name}_gray", out_iter, ref_model_dir, ref_payload["vertex"], gray_vertex, getattr(ref_payload["ply"], "text", False))
        _write_product_model(out_root / f"{name}_pseudocolor", out_iter, ref_model_dir, ref_payload["vertex"], pseudo_vertex, getattr(ref_payload["ply"], "text", False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SpectralIndexGS false-color and index proxy products from four band models.")
    ap.add_argument("--g_model_dir", required=True)
    ap.add_argument("--r_model_dir", required=True)
    ap.add_argument("--re_model_dir", required=True)
    ap.add_argument("--nir_model_dir", required=True)
    ap.add_argument("--g_iter", type=int, default=None)
    ap.add_argument("--r_iter", type=int, default=None)
    ap.add_argument("--re_iter", type=int, default=None)
    ap.add_argument("--nir_iter", type=int, default=None)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--savi_l", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--require_opacity_match", type=str, default="true")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    require_opacity_match = str(args.require_opacity_match).strip().lower() not in ("0", "false", "no", "off")
    build_products(
        model_dirs={
            "G": Path(args.g_model_dir).resolve(),
            "R": Path(args.r_model_dir).resolve(),
            "RE": Path(args.re_model_dir).resolve(),
            "NIR": Path(args.nir_model_dir).resolve(),
        },
        iterations={
            "G": args.g_iter,
            "R": args.r_iter,
            "RE": args.re_iter,
            "NIR": args.nir_iter,
        },
        out_root=Path(args.out_root).resolve(),
        savi_l=float(args.savi_l),
        eps=float(args.eps),
        require_opacity_match=require_opacity_match,
        tol=float(args.tol),
    )
    print(json.dumps({
        "out_root": str(Path(args.out_root).resolve()),
        "products": list(FALSE_COLOR_PRODUCTS.keys()) + [
            "ndvi_gray", "ndvi_pseudocolor",
            "ndre_gray", "ndre_pseudocolor",
            "gndvi_gray", "gndvi_pseudocolor",
            "savi_gray", "savi_pseudocolor",
            "osavi_gray", "osavi_pseudocolor",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
