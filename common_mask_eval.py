import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from utils.image_utils import psnr as torch_psnr
from utils.loss_utils import ssim as torch_ssim
from lpipsPyTorch.modules.lpips import LPIPS

try:
    from scipy.stats import spearmanr as scipy_spearmanr
except Exception:
    scipy_spearmanr = None


def _parse_method_items(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --method item: {it}; expected NAME=RUN_ROOT")
        name, root = it.split("=", 1)
        name = name.strip()
        root = root.strip()
        if not name or not root:
            raise ValueError(f"Invalid --method item: {it}")
        out[name] = Path(root).resolve()
    if not out:
        raise ValueError("No methods provided")
    return out


def _find_ours_dir(model_path: Path, split: str, iteration: int) -> Tuple[int, Path]:
    split_dir = model_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")
    if iteration > 0:
        ours_dir = split_dir / f"ours_{iteration}"
        if not ours_dir.exists():
            raise FileNotFoundError(f"Missing directory: {ours_dir}")
        return int(iteration), ours_dir
    cands: List[Tuple[int, Path]] = []
    for d in split_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"ours_(\d+)$", d.name)
        if m:
            cands.append((int(m.group(1)), d))
    if not cands:
        raise FileNotFoundError(f"No ours_* dir under: {split_dir}")
    cands.sort(key=lambda x: x[0])
    return cands[-1]


def _parse_source_path_from_cfg(cfg_path: Path) -> Path:
    txt = cfg_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"source_path='([^']+)'", txt)
    if not m:
        m = re.search(r'source_path="([^"]+)"', txt)
    if not m:
        raise ValueError(f"Cannot parse source_path from {cfg_path}")
    return Path(m.group(1)).resolve()


def _load_camera_img_names(model_path: Path, n_views: int) -> List[str]:
    pj = model_path / "cameras.json"
    if not pj.exists():
        raise FileNotFoundError(f"Missing cameras.json: {pj}")
    data = json.loads(pj.read_text(encoding="utf-8"))
    if len(data) < n_views:
        raise RuntimeError(f"cameras.json has {len(data)} entries but render has {n_views}")
    names = [str(d.get("img_name", "")) for d in data[:n_views]]
    if any(not n for n in names):
        raise RuntimeError(f"Invalid img_name in first {n_views} entries of {pj}")
    return names


def _load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(mask_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    m = Image.open(mask_path).convert("L")
    if m.size != (w, h):
        m = m.resize((w, h), resample=Image.Resampling.NEAREST)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.bool_)


def _mean_median(xs: List[float]) -> Dict[str, float]:
    arr = np.array(xs, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std(ddof=0))}


def _masked_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    if mask.sum() <= 0:
        return float("nan")
    diff2 = (a - b) ** 2
    if a.ndim == 3:
        val = diff2[mask].mean()
    else:
        val = diff2[mask].mean()
    return float(np.sqrt(max(float(val), eps)))


def _spearman(x: np.ndarray, y: np.ndarray, max_points: int = 200000) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size != y.size:
        raise ValueError("x and y size mismatch")
    n = x.size
    if n > max_points > 0:
        idx = np.random.default_rng(12345).choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    if scipy_spearmanr is not None:
        val = scipy_spearmanr(x, y).statistic
        return float(val) if np.isfinite(val) else float("nan")
    # Fallback: no tie-correction rank correlation
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def _index_formula(name: str, g: np.ndarray, r: np.ndarray, re: np.ndarray, nir: np.ndarray, eps: float, savi_l: float) -> np.ndarray:
    nm = name.lower()
    if nm == "ndvi":
        v = (nir - r) / (nir + r + eps)
    elif nm == "ndre":
        v = (nir - re) / (nir + re + eps)
    elif nm == "savi":
        v = ((nir - r) / (nir + r + savi_l + eps)) * (1.0 + savi_l)
    else:
        raise ValueError(f"Unsupported index: {name}")
    return np.clip((v + 1.0) * 0.5, 0.0, 1.0)


def _torch_metric_psnr_ssim_lpips(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, lpips_model: LPIPS, device: torch.device) -> Tuple[float, float, float]:
    # pred/gt: [H,W,3], mask: [H,W] bool
    m = mask.astype(np.float32)
    if m.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    pt = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    mt = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    pt = (pt * mt).contiguous()
    gt_t = (gt_t * mt).contiguous()
    with torch.no_grad():
        ps = torch_psnr(pt, gt_t).mean().item()
        ss = torch_ssim(pt, gt_t).item()
        lp = lpips_model(pt, gt_t).mean().item()
    return float(ps), float(ss), float(lp)


def _render_if_missing(model_path: Path, iteration: int) -> None:
    it, ours = _find_ours_dir(model_path, "test", iteration)
    rdir = ours / "renders"
    gdir = ours / "gt"
    if rdir.exists() and gdir.exists():
        return
    cmd = [
        sys.executable,
        "render.py",
        "-m", str(model_path),
        "--iteration", str(it if iteration > 0 else -1),
        "--skip_train",
    ]
    subprocess.run(cmd, check=True)


@dataclass
class BandModelArtifacts:
    model_path: Path
    iteration: int
    renders_dir: Path
    gt_dir: Path
    image_names: List[str]
    mask_dir: Path
    files: List[str]


def _prepare_band_artifacts(run_root: Path, band: str, iteration: int) -> BandModelArtifacts:
    model_path = run_root / f"Model_{band}"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model path: {model_path}")
    it, ours = _find_ours_dir(model_path, "test", iteration)
    renders_dir = ours / "renders"
    gt_dir = ours / "gt"
    if not renders_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt in {ours}; run render.py first")
    files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    if files != sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")]):
        raise RuntimeError(f"render/gt file mismatch in {ours}")
    image_names = _load_camera_img_names(model_path, len(files))
    cfg = model_path / "cfg_args"
    src = _parse_source_path_from_cfg(cfg)
    mask_dir = src / "validity_masks"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing validity_masks: {mask_dir}")
    return BandModelArtifacts(
        model_path=model_path,
        iteration=it,
        renders_dir=renders_dir,
        gt_dir=gt_dir,
        image_names=image_names,
        mask_dir=mask_dir,
        files=files,
    )


def _prepare_index_proxy_artifacts(run_root: Path, index_name: str, iteration: int) -> Tuple[Path, Path, List[str]]:
    model_path = run_root / "Products" / f"{index_name.lower()}_gray"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing product model: {model_path}")
    it, ours = _find_ours_dir(model_path, "test", iteration)
    renders_dir = ours / "renders"
    if not renders_dir.exists():
        raise FileNotFoundError(f"Missing proxy renders: {renders_dir}; run render.py on product model first")
    files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    image_names = _load_camera_img_names(model_path, len(files))
    return renders_dir, model_path, image_names


def evaluate_common_mask(
    methods: Dict[str, Path],
    iteration: int,
    bands: Sequence[str],
    indices: Sequence[str],
    savi_l: float,
    eps: float,
    device: str,
) -> Dict:
    method_names = list(methods.keys())
    band_models: Dict[str, Dict[str, BandModelArtifacts]] = {m: {} for m in method_names}

    for m in method_names:
        for b in bands:
            band_models[m][b] = _prepare_band_artifacts(methods[m], b, iteration)

    # Sanity: all bands/methods should align to same file list length and image_name list
    ref_method = method_names[0]
    ref_band = bands[0]
    ref_files = band_models[ref_method][ref_band].files
    ref_img_names = band_models[ref_method][ref_band].image_names
    for m in method_names:
        for b in bands:
            art = band_models[m][b]
            if art.files != ref_files:
                raise RuntimeError(f"File list mismatch for method={m}, band={b}")
            if art.image_names != ref_img_names:
                raise RuntimeError(f"image_name sequence mismatch for method={m}, band={b}")

    lpips_model = LPIPS(net_type="vgg", version="0.1").to(device).eval()

    results: Dict[str, object] = {
        "meta": {
            "methods": {k: str(v) for k, v in methods.items()},
            "iteration": int(iteration),
            "bands": list(bands),
            "indices": [s.upper() for s in indices],
            "mask_policy": "common_mask_all_method_intersection",
            "device": device,
            "eps": float(eps),
            "savi_l": float(savi_l),
        },
        "coverage": {
            "per_method_band": {},
            "common_band": {},
            "per_method_joint4band": {},
            "common_joint4band": {},
            "per_method_index": {},
            "common_index": {},
        },
        "single_band_metrics": {},
        "spectral_4band_metrics": {},
        "index_consistency": {},
    }

    # Single-band common-mask metrics
    for b in bands:
        results["single_band_metrics"][b] = {}
        common_cov_values: List[float] = []
        per_method_cov: Dict[str, List[float]] = {m: [] for m in method_names}
        per_method_psnr: Dict[str, List[float]] = {m: [] for m in method_names}
        per_method_ssim: Dict[str, List[float]] = {m: [] for m in method_names}
        per_method_lpips: Dict[str, List[float]] = {m: [] for m in method_names}

        for i, fn in enumerate(ref_files):
            masks = {}
            for m in method_names:
                art = band_models[m][b]
                img_name = art.image_names[i]
                mask_path = art.mask_dir / f"{Path(img_name).stem}.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask for method={m}, band={b}, image={img_name}: {mask_path}")
                # target shape from render
                render = _load_rgb01(art.renders_dir / fn)
                mask = _load_mask(mask_path, render.shape[:2])
                masks[m] = mask
                per_method_cov[m].append(float(mask.mean()))

            common_mask = None
            for m in method_names:
                common_mask = masks[m] if common_mask is None else np.logical_and(common_mask, masks[m])
            common_cov_values.append(float(common_mask.mean()))

            for m in method_names:
                art = band_models[m][b]
                pred = _load_rgb01(art.renders_dir / fn)
                gt = _load_rgb01(art.gt_dir / fn)
                if pred.shape != gt.shape:
                    raise RuntimeError(f"Shape mismatch method={m}, band={b}, file={fn}")
                p, s, l = _torch_metric_psnr_ssim_lpips(pred, gt, common_mask, lpips_model, torch.device(device))
                per_method_psnr[m].append(p)
                per_method_ssim[m].append(s)
                per_method_lpips[m].append(l)

        for m in method_names:
            results["single_band_metrics"][b][m] = {
                "psnr": _mean_median(per_method_psnr[m]),
                "ssim": _mean_median(per_method_ssim[m]),
                "lpips": _mean_median(per_method_lpips[m]),
            }
            results["coverage"]["per_method_band"].setdefault(m, {})[b] = _mean_median(per_method_cov[m])
        results["coverage"]["common_band"][b] = _mean_median(common_cov_values)

    # 4-band spectral metrics (common joint mask)
    per_method_sam: Dict[str, List[float]] = {m: [] for m in method_names}
    per_method_rmse: Dict[str, List[float]] = {m: [] for m in method_names}
    common_joint_cov: List[float] = []
    per_method_joint_cov: Dict[str, List[float]] = {m: [] for m in method_names}

    for i, fn in enumerate(ref_files):
        # per-method joint(4band) mask
        method_joint_masks: Dict[str, np.ndarray] = {}
        for m in method_names:
            mj = None
            for b in bands:
                art = band_models[m][b]
                img_name = art.image_names[i]
                mask_path = art.mask_dir / f"{Path(img_name).stem}.png"
                render = _load_rgb01(art.renders_dir / fn)
                mask = _load_mask(mask_path, render.shape[:2])
                mj = mask if mj is None else np.logical_and(mj, mask)
            method_joint_masks[m] = mj
            per_method_joint_cov[m].append(float(mj.mean()))
        cm = None
        for m in method_names:
            cm = method_joint_masks[m] if cm is None else np.logical_and(cm, method_joint_masks[m])
        common_joint_cov.append(float(cm.mean()))

        for m in method_names:
            pred_stack = []
            gt_stack = []
            for b in bands:
                art = band_models[m][b]
                pred = _load_rgb01(art.renders_dir / fn)[..., 0]
                gt = _load_rgb01(art.gt_dir / fn)[..., 0]
                pred_stack.append(pred)
                gt_stack.append(gt)
            pred4 = np.stack(pred_stack, axis=-1)
            gt4 = np.stack(gt_stack, axis=-1)

            if cm.sum() <= 0:
                per_method_rmse[m].append(float("nan"))
                per_method_sam[m].append(float("nan"))
                continue

            diff = pred4 - gt4
            rmse = float(np.sqrt(np.mean((diff[cm]) ** 2)))
            dot = np.sum(pred4 * gt4, axis=-1)
            n1 = np.linalg.norm(pred4, axis=-1)
            n2 = np.linalg.norm(gt4, axis=-1)
            cosv = dot / (n1 * n2 + eps)
            cosv = np.clip(cosv, -1.0, 1.0)
            sam_deg = np.degrees(np.arccos(cosv))
            sam_mean = float(np.mean(sam_deg[cm]))

            per_method_rmse[m].append(rmse)
            per_method_sam[m].append(sam_mean)

    for m in method_names:
        results["spectral_4band_metrics"][m] = {
            "rmse": _mean_median(per_method_rmse[m]),
            "sam_deg": _mean_median(per_method_sam[m]),
        }
        results["coverage"]["per_method_joint4band"][m] = _mean_median(per_method_joint_cov[m])
    results["coverage"]["common_joint4band"] = _mean_median(common_joint_cov)

    # Index consistency: render-then-index vs proxy-index-render (common mask across methods)
    for idx_name in indices:
        idx_key = idx_name.lower()
        if idx_key == "ndvi":
            req_bands = ("R", "NIR")
        elif idx_key == "ndre":
            req_bands = ("RE", "NIR")
        elif idx_key == "savi":
            req_bands = ("R", "NIR")
        else:
            raise ValueError(f"Unsupported index for consistency: {idx_name}")

        proxy_artifacts: Dict[str, Tuple[Path, Path, List[str]]] = {}
        for m in method_names:
            proxy_artifacts[m] = _prepare_index_proxy_artifacts(methods[m], idx_key, iteration)

        # sanity check proxy files and names match reference
        for m in method_names:
            pdir, _, pnames = proxy_artifacts[m]
            pfiles = sorted([f for f in os.listdir(pdir) if f.lower().endswith(".png")])
            if pfiles != ref_files:
                raise RuntimeError(f"Proxy file list mismatch for method={m}, index={idx_key}")
            if pnames != ref_img_names:
                raise RuntimeError(f"Proxy image_name mismatch for method={m}, index={idx_key}")

        per_method_rmse_idx: Dict[str, List[float]] = {m: [] for m in method_names}
        per_method_spr_idx: Dict[str, List[float]] = {m: [] for m in method_names}
        common_cov_idx: List[float] = []
        per_method_cov_idx: Dict[str, List[float]] = {m: [] for m in method_names}

        for i, fn in enumerate(ref_files):
            method_masks: Dict[str, np.ndarray] = {}
            method_rt: Dict[str, np.ndarray] = {}
            method_proxy: Dict[str, np.ndarray] = {}

            for m in method_names:
                # band renders
                g = _load_rgb01(band_models[m]["G"].renders_dir / fn)[..., 0]
                r = _load_rgb01(band_models[m]["R"].renders_dir / fn)[..., 0]
                re = _load_rgb01(band_models[m]["RE"].renders_dir / fn)[..., 0]
                nir = _load_rgb01(band_models[m]["NIR"].renders_dir / fn)[..., 0]
                rt = _index_formula(idx_key, g, r, re, nir, eps=eps, savi_l=savi_l)
                method_rt[m] = rt

                # proxy render
                pdir, _, _ = proxy_artifacts[m]
                proxy = _load_rgb01(pdir / fn)[..., 0]
                method_proxy[m] = proxy

                # mask for required bands (proxy has no explicit mask in current pipeline)
                mm = None
                for b in req_bands:
                    art = band_models[m][b]
                    img_name = art.image_names[i]
                    mask_path = art.mask_dir / f"{Path(img_name).stem}.png"
                    render_shape = _load_rgb01(art.renders_dir / fn).shape[:2]
                    msk = _load_mask(mask_path, render_shape)
                    mm = msk if mm is None else np.logical_and(mm, msk)
                method_masks[m] = mm
                per_method_cov_idx[m].append(float(mm.mean()))

            cm = None
            for m in method_names:
                cm = method_masks[m] if cm is None else np.logical_and(cm, method_masks[m])
            common_cov_idx.append(float(cm.mean()))

            for m in method_names:
                if cm.sum() <= 0:
                    per_method_rmse_idx[m].append(float("nan"))
                    per_method_spr_idx[m].append(float("nan"))
                    continue
                a = method_rt[m][cm]
                b = method_proxy[m][cm]
                rmse = float(np.sqrt(np.mean((a - b) ** 2)))
                spr = _spearman(a.reshape(-1), b.reshape(-1))
                per_method_rmse_idx[m].append(rmse)
                per_method_spr_idx[m].append(spr)

        results["index_consistency"][idx_key.upper()] = {}
        for m in method_names:
            results["index_consistency"][idx_key.upper()][m] = {
                "rmse": _mean_median(per_method_rmse_idx[m]),
                "spearman": _mean_median(per_method_spr_idx[m]),
            }
            results["coverage"]["per_method_index"].setdefault(m, {})[idx_key.upper()] = _mean_median(per_method_cov_idx[m])
        results["coverage"]["common_index"][idx_key.upper()] = _mean_median(common_cov_idx)

    return results


def main():
    ap = argparse.ArgumentParser("Common-mask evaluator for SpectralIndexGS full comparisons")
    ap.add_argument("--method", action="append", required=True, help="NAME=RUN_ROOT (repeatable), e.g., E0=... --method E1=...")
    ap.add_argument("--iteration", type=int, default=60000)
    ap.add_argument("--bands", type=str, default="G,R,RE,NIR")
    ap.add_argument("--indices", type=str, default="NDVI,NDRE,SAVI")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--savi_l", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--auto_render_missing_products", type=str, default="false")
    args = ap.parse_args()

    auto_render = str(args.auto_render_missing_products).strip().lower() in ("1", "true", "yes", "on")
    methods = _parse_method_items(args.method)
    bands = [b.strip().upper() for b in args.bands.split(",") if b.strip()]
    indices = [s.strip().upper() for s in args.indices.split(",") if s.strip()]

    if auto_render:
        # Ensure required product renders are present
        for _, run_root in methods.items():
            for idx in indices:
                product_model = run_root / "Products" / f"{idx.lower()}_gray"
                try:
                    _render_if_missing(product_model, args.iteration)
                except Exception as e:
                    raise RuntimeError(f"Failed to auto-render product {product_model}: {e}")

    out = evaluate_common_mask(
        methods=methods,
        iteration=int(args.iteration),
        bands=bands,
        indices=indices,
        savi_l=float(args.savi_l),
        eps=float(args.eps),
        device=args.device,
    )

    if args.out_json:
        out_json = Path(args.out_json).resolve()
    else:
        if len(methods) >= 2:
            names = "_vs_".join(methods.keys())
        else:
            names = next(iter(methods.keys()))
        out_json = Path.cwd() / f"common_mask_eval_{names}_iter{args.iteration}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[COMMON-MASK] saved: {out_json}")
    print(f"[COMMON-MASK] methods: {list(methods.keys())}")
    print(f"[COMMON-MASK] bands: {bands}")
    print(f"[COMMON-MASK] indices: {indices}")
    # concise summary
    for b in bands:
        print(f"  [BAND {b}] common_cov_mean={out['coverage']['common_band'][b]['mean']:.6f}")
        for m in methods.keys():
            sm = out["single_band_metrics"][b][m]
            print(f"    {m}: PSNR={sm['psnr']['mean']:.4f} SSIM={sm['ssim']['mean']:.4f} LPIPS={sm['lpips']['mean']:.4f}")
    print(f"  [4BAND] common_joint_cov_mean={out['coverage']['common_joint4band']['mean']:.6f}")
    for m in methods.keys():
        sp = out["spectral_4band_metrics"][m]
        print(f"    {m}: SAM(deg)={sp['sam_deg']['mean']:.4f} RMSE={sp['rmse']['mean']:.5f}")


if __name__ == "__main__":
    main()
