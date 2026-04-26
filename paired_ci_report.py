import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from utils.validity_mask_utils import load_validity_mask_image, load_validity_mask_or_ones, resolve_validity_mask_policy


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_mask(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    return load_validity_mask_image(path, target_hw)


def _camera_test_names(model_root: Path, n_test: int) -> List[str]:
    cams = json.loads((model_root / "cameras.json").read_text(encoding="utf-8"))
    return [str(c["img_name"]) for c in cams[:n_test]]


def _masked_mse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(mask.sum()) * 3.0
    if denom <= eps:
        return float("nan")
    return float((((pred - gt) ** 2) * mask[..., None]).sum() / denom)


def _psnr_from_mse(mse: float, eps: float = 1e-12) -> float:
    if not np.isfinite(mse):
        return float("nan")
    return float(-10.0 * math.log10(max(float(mse), eps)))


def _sam_deg(gt4: np.ndarray, pred4: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    if mask.sum() <= 0:
        return float("nan")
    gt_v = gt4[mask]
    pr_v = pred4[mask]
    dot = np.sum(gt_v * pr_v, axis=1)
    ng = np.linalg.norm(gt_v, axis=1)
    npv = np.linalg.norm(pr_v, axis=1)
    cosv = dot / np.clip(ng * npv, eps, None)
    cosv = np.clip(cosv, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosv))
    return float(np.mean(ang))


def _rankdata_ordinal(a: np.ndarray) -> np.ndarray:
    # Fast ordinal ranks; sufficient for dense image vectors here.
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _rankdata_ordinal(x)
    ry = _rankdata_ordinal(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= eps:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _index_formula(name: str, g: np.ndarray, r: np.ndarray, re: np.ndarray, nir: np.ndarray, eps: float, savi_l: float) -> np.ndarray:
    n = name.upper()
    if n == "NDVI":
        return (nir - r) / (nir + r + eps)
    if n == "NDRE":
        return (nir - re) / (nir + re + eps)
    if n == "SAVI":
        return ((nir - r) / (nir + r + savi_l + eps)) * (1.0 + savi_l)
    raise ValueError(f"Unsupported index: {name}")


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 1234) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(vals[idx].mean())
    return {
        "mean": float(vals.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n": int(n),
    }


def _resolve_ours(model_path: Path, split: str, iteration: int) -> Path:
    ours = model_path / split / f"ours_{iteration}"
    if not ours.exists():
        raise FileNotFoundError(f"Missing ours dir: {ours}")
    return ours


def _prepare_band_data(method_root: Path, band: str, iteration: int):
    model = method_root / f"Model_{band}"
    ours = _resolve_ours(model, "test", iteration)
    renders = ours / "renders"
    gts = ours / "gt"
    files = sorted([p.name for p in renders.glob("*.png")])
    names = _camera_test_names(model, len(files))
    source_path = None
    cfg = (model / "cfg_args").read_text(encoding="utf-8", errors="ignore")
    m1 = "source_path='"
    m2 = 'source_path="'
    if m1 in cfg:
        source_path = cfg.split(m1, 1)[1].split("'", 1)[0]
    elif m2 in cfg:
        source_path = cfg.split(m2, 1)[1].split('"', 1)[0]
    if not source_path:
        raise RuntimeError(f"Cannot parse source_path from {model / 'cfg_args'}")
    cfg_path = model / "cfg_args"
    _, mask_dir, use_validity_mask = resolve_validity_mask_policy(cfg_path)
    return {
        "model": model,
        "renders": renders,
        "gts": gts,
        "files": files,
        "names": names,
        "cfg_path": cfg_path,
        "mask_dir": mask_dir,
        "use_validity_mask": use_validity_mask,
    }


def _prepare_proxy_data(method_root: Path, index_name: str, iteration: int):
    model = method_root / "Products" / f"{index_name.lower()}_gray"
    ours = _resolve_ours(model, "test", iteration)
    renders = ours / "renders"
    files = sorted([p.name for p in renders.glob("*.png")])
    return {"model": model, "renders": renders, "files": files}


def evaluate(methods: Dict[str, Path], iteration: int, bands: List[str], indices: List[str], eps: float, savi_l: float) -> Dict:
    out = {
        "meta": {
            "methods": {k: str(v) for k, v in methods.items()},
            "iteration": iteration,
            "bands": bands,
            "indices": indices,
            "mask_policy": "common_mask_all_method_intersection",
        },
        "per_view": {"band_psnr": {}, "spectral_sam": {}, "index_rmse": {}, "index_spearman": {}},
        "paired_ci": {"band_psnr_delta": {}, "spectral_sam_delta": {}, "index_rmse_delta": {}, "index_spearman_delta": {}},
    }

    # Band PSNR per view
    for b in bands:
        prep = {m: _prepare_band_data(root, b, iteration) for m, root in methods.items()}
        n = len(next(iter(prep.values()))["files"])
        for m in methods:
            if len(prep[m]["files"]) != n:
                raise RuntimeError(f"View count mismatch in band {b}, method={m}")
        per_method = {m: [] for m in methods}
        for i in range(n):
            common_mask = None
            cache = {}
            for m in methods:
                f = prep[m]["files"][i]
                nm = prep[m]["names"][i]
                render = _load_rgb(prep[m]["renders"] / f)
                gt = _load_rgb(prep[m]["gts"] / f)
                mask, _, _ = load_validity_mask_or_ones(prep[m]["cfg_path"], nm, render.shape[:2])
                common_mask = mask if common_mask is None else (common_mask & mask)
                cache[m] = (render, gt)
            for m in methods:
                render, gt = cache[m]
                mse = _masked_mse(render, gt, common_mask)
                per_method[m].append(_psnr_from_mse(mse))
        out["per_view"]["band_psnr"][b] = per_method

    # 4-band SAM per view
    n = len(next(iter(out["per_view"]["band_psnr"].values()))[next(iter(methods))])
    sam_per_method = {m: [] for m in methods}
    for i in range(n):
        common_joint = None
        gt4 = {}
        pr4 = {}
        for m, root in methods.items():
            gt_channels = []
            pr_channels = []
            masks = []
            for b in bands:
                pb = _prepare_band_data(root, b, iteration)
                f = pb["files"][i]
                nm = pb["names"][i]
                gt = _load_rgb(pb["gts"] / f)[..., 0]
                pr = _load_rgb(pb["renders"] / f)[..., 0]
                mk, _, _ = load_validity_mask_or_ones(pb["cfg_path"], nm, gt.shape[:2])
                gt_channels.append(gt)
                pr_channels.append(pr)
                masks.append(mk)
            method_joint = masks[0] & masks[1] & masks[2] & masks[3]
            common_joint = method_joint if common_joint is None else (common_joint & method_joint)
            gt4[m] = np.stack(gt_channels, axis=-1)
            pr4[m] = np.stack(pr_channels, axis=-1)
        for m in methods:
            sam_per_method[m].append(_sam_deg(gt4[m], pr4[m], common_joint))
    out["per_view"]["spectral_sam"] = sam_per_method

    # Index consistency per view (render-then-index vs proxy-render)
    for idx_name in indices:
        rmse_pm = {m: [] for m in methods}
        spr_pm = {m: [] for m in methods}
        for i in range(n):
            common_idx = None
            rt_idx = {}
            proxy_idx = {}
            for m, root in methods.items():
                # render-then-index source from 4 bands
                pb = {b: _prepare_band_data(root, b, iteration) for b in bands}
                g = _load_rgb(pb["G"]["renders"] / pb["G"]["files"][i])[..., 0]
                r = _load_rgb(pb["R"]["renders"] / pb["R"]["files"][i])[..., 0]
                re = _load_rgb(pb["RE"]["renders"] / pb["RE"]["files"][i])[..., 0]
                nir = _load_rgb(pb["NIR"]["renders"] / pb["NIR"]["files"][i])[..., 0]
                masks = []
                for b in bands:
                    nm = pb[b]["names"][i]
                    mk, _, _ = load_validity_mask_or_ones(pb[b]["cfg_path"], nm, g.shape[:2])
                    masks.append(mk)
                method_joint = masks[0] & masks[1] & masks[2] & masks[3]

                proxy = _prepare_proxy_data(root, idx_name, iteration)
                pimg = _load_rgb(proxy["renders"] / proxy["files"][i])[..., 0]

                common_idx = method_joint if common_idx is None else (common_idx & method_joint)
                rt_idx[m] = _index_formula(idx_name, g, r, re, nir, eps=eps, savi_l=savi_l)
                proxy_idx[m] = pimg
            for m in methods:
                msk = common_idx
                x = rt_idx[m][msk]
                y = proxy_idx[m][msk]
                rmse = float(np.sqrt(np.mean((x - y) ** 2))) if x.size > 0 else float("nan")
                rho = _spearman(x, y)
                rmse_pm[m].append(rmse)
                spr_pm[m].append(rho)
        out["per_view"]["index_rmse"][idx_name] = rmse_pm
        out["per_view"]["index_spearman"][idx_name] = spr_pm

    # Pairwise deltas + CI
    method_keys = list(methods.keys())
    pairs = []
    if "E1" in method_keys and "E0" in method_keys:
        pairs.append(("E1", "E0"))
    if "E1" in method_keys and "E2" in method_keys:
        pairs.append(("E1", "E2"))
    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            a, b = method_keys[i], method_keys[j]
            if (a, b) not in pairs and (b, a) not in pairs:
                pairs.append((a, b))

    for a, b in pairs:
        key = f"{a}-minus-{b}"
        out["paired_ci"]["band_psnr_delta"][key] = {}
        for band in bands:
            arr = np.array(out["per_view"]["band_psnr"][band][a], dtype=np.float64) - np.array(
                out["per_view"]["band_psnr"][band][b], dtype=np.float64
            )
            out["paired_ci"]["band_psnr_delta"][key][band] = _bootstrap_ci(arr)

        arr_sam = np.array(out["per_view"]["spectral_sam"][a], dtype=np.float64) - np.array(
            out["per_view"]["spectral_sam"][b], dtype=np.float64
        )
        out["paired_ci"]["spectral_sam_delta"][key] = _bootstrap_ci(arr_sam)

        out["paired_ci"]["index_rmse_delta"][key] = {}
        out["paired_ci"]["index_spearman_delta"][key] = {}
        for idx_name in indices:
            arr_rmse = np.array(out["per_view"]["index_rmse"][idx_name][a], dtype=np.float64) - np.array(
                out["per_view"]["index_rmse"][idx_name][b], dtype=np.float64
            )
            arr_sp = np.array(out["per_view"]["index_spearman"][idx_name][a], dtype=np.float64) - np.array(
                out["per_view"]["index_spearman"][idx_name][b], dtype=np.float64
            )
            out["paired_ci"]["index_rmse_delta"][key][idx_name] = _bootstrap_ci(arr_rmse)
            out["paired_ci"]["index_spearman_delta"][key][idx_name] = _bootstrap_ci(arr_sp)

    return out


def main():
    ap = argparse.ArgumentParser("Per-view paired delta + bootstrap CI report")
    ap.add_argument("--method", action="append", required=True, help="NAME=RUN_ROOT (repeatable)")
    ap.add_argument("--iteration", type=int, default=60000)
    ap.add_argument("--bands", type=str, default="G,R,RE,NIR")
    ap.add_argument("--indices", type=str, default="NDVI,NDRE,SAVI")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--savi_l", type=float, default=0.5)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    methods: Dict[str, Path] = {}
    for item in args.method:
        if "=" not in item:
            raise ValueError(f"Invalid --method '{item}', expected NAME=PATH")
        k, v = item.split("=", 1)
        methods[k.strip()] = Path(v.strip())

    bands = [b.strip().upper() for b in args.bands.split(",") if b.strip()]
    indices = [i.strip().upper() for i in args.indices.split(",") if i.strip()]

    out = evaluate(
        methods=methods,
        iteration=args.iteration,
        bands=bands,
        indices=indices,
        eps=args.eps,
        savi_l=args.savi_l,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[PAIRED-CI] saved: {out_path}")


if __name__ == "__main__":
    main()

