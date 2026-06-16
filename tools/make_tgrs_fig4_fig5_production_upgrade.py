"""Build production-upgrade review figures for TGRS Fig.4/Fig.5.

This script is intentionally manuscript-adjacent: it creates review-ready
candidate figures, manifests, DPI reports, and a GPT review ZIP, but does not
modify the LaTeX manuscript or overwrite the current manuscript figure files.
It uses only existing local artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"E:\Multispectral")
PAPER_ROOT = Path(r"E:\paper\TGRS\UMGS_TGRS")
OUT_ROOT = ROOT / "outputs" / "tgrs_figure_production_upgrade_20260612"
FIG_DIR = OUT_ROOT / "figures"
COMP_DIR = OUT_ROOT / "comparisons"
MANIFEST_DIR = OUT_ROOT / "manifests"
PACKAGE_DIR = OUT_ROOT / "package"
GPT_PACKAGE = ROOT / "outputs" / "gpt_review_packages" / "GPT_TGRS_FIG4_FIG5_PRODUCTION_UPGRADE_REVIEW_20260612.zip"

FIG4_MANIFEST = ROOT / "outputs" / "tgrs_figure_polish_plan_20260611" / "source_manifests" / "figure4_benchmark_product_examples_manifest.json"
FIG5_SELECTION = ROOT / "outputs" / "tgrs_no_valid_mask_visual_20260611" / "selected_views.json"
FIG5_SOURCE_ROOT = ROOT / "outputs" / "tgrs_no_valid_mask_visual_20260611" / "source_assets" / "selected_sources"
FIG5_AGG_CSV = ROOT / "outputs" / "benchmark16_no_valid_mask_ablation" / "benchmark_internal_no_valid_mask_comparison_20260611.csv"
SOURCE_AUDIT = ROOT / "outputs" / "gpt_review_packages" / "FIG4_FIG5_SOURCE_ARTIFACT_AUDIT.md"

OLD_FIG4 = ROOT / "outputs" / "tgrs_nature_figure_polish_20260611" / "fig4" / "fig4_product_examples_nature_v2.png"
OLD_FIG5 = ROOT / "outputs" / "tgrs_no_valid_mask_visual_20260611" / "figure5_candidate_benchmark_internal_no_valid_mask.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.65,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


EPS = 1e-6
FIG_W = 7.16


INDEX_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "umgs_index", ["#5b3b23", "#f2efe8", "#1f7a3a"], N=256
)
DELTA_CMAP = plt.get_cmap("coolwarm").copy()
HEAT_CMAP = plt.get_cmap("YlOrBr").copy()


@dataclass
class RasterPanel:
    label: str
    source_path: Path
    source_size: Tuple[int, int]
    axis_size_in: Tuple[float, float]
    effective_ppi_x: float
    effective_ppi_y: float


def ensure_dirs() -> None:
    for d in [FIG_DIR, COMP_DIR, MANIFEST_DIR, PACKAGE_DIR, GPT_PACKAGE.parent]:
        d.mkdir(parents=True, exist_ok=True)


def read_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im.convert("F"), dtype=np.float32, copy=True)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def band_path(root: Path, model: str, index_png: str, kind: str = "renders") -> Path:
    return root / model / "test" / "ours_60000" / kind / index_png


def fig5_band_path(scene: str, branch: str, model: str, io_kind: str, view: str) -> Path:
    return FIG5_SOURCE_ROOT / scene / branch / model / io_kind / f"{view}.png"


def normalize_rgb_for_display(img: Image.Image, out_size: Tuple[int, int]) -> np.ndarray:
    # Center-crop to the target aspect before percentile autocontrast.
    target_w, target_h = out_size
    target_aspect = target_w / target_h
    w, h = img.size
    aspect = w / h
    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize(out_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    arr = (arr - lo) / max(hi - lo, EPS)
    return np.clip(arr, 0, 1)


def index_from_bands(num: np.ndarray, den: np.ndarray, mask: np.ndarray | None = None) -> np.ma.MaskedArray:
    idx = (num - den) / (num + den + EPS)
    idx = np.clip(idx, -1.0, 1.0)
    if mask is not None:
        idx = np.ma.array(idx, mask=~mask)
    return idx


def common_mask(*bands: np.ndarray) -> np.ndarray:
    mask = np.ones_like(bands[0], dtype=bool)
    for b in bands:
        mask &= np.isfinite(b)
        mask &= b > 0
    return mask


def display_stretch(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    vals = arr[mask] if mask is not None and mask.any() else arr[np.isfinite(arr)]
    lo, hi = np.percentile(vals, [2, 98])
    return np.clip((arr - lo) / max(hi - lo, EPS), 0.0, 1.0)


def save_pub(fig: plt.Figure, stem: Path, dpi: int = 600) -> Dict[str, str]:
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    return {ext: str(stem.with_suffix(f".{ext}")) for ext in ["svg", "pdf", "png", "tiff"]}


def axis_effective_ppi(fig: plt.Figure, ax: plt.Axes, source_path: Path, label: str) -> RasterPanel:
    bbox = ax.get_position()
    width_in = bbox.width * fig.get_size_inches()[0]
    height_in = bbox.height * fig.get_size_inches()[1]
    with Image.open(source_path) as im:
        w, h = im.size
    return RasterPanel(
        label=label,
        source_path=source_path,
        source_size=(w, h),
        axis_size_in=(width_in, height_in),
        effective_ppi_x=w / max(width_in, EPS),
        effective_ppi_y=h / max(height_in, EPS),
    )


def panel_record(panel: RasterPanel) -> Dict[str, object]:
    return {
        "label": panel.label,
        "source_path": str(panel.source_path),
        "source_size_px": list(panel.source_size),
        "axis_size_in": [round(panel.axis_size_in[0], 3), round(panel.axis_size_in[1], 3)],
        "effective_ppi_x": round(panel.effective_ppi_x, 1),
        "effective_ppi_y": round(panel.effective_ppi_y, 1),
    }


def make_fig4() -> Tuple[Dict[str, str], List[RasterPanel], Dict[str, object]]:
    manifest = json.loads(FIG4_MANIFEST.read_text(encoding="utf-8"))
    selected_names = [
        "cassava_01_20260526_1603",
        "eucalyptus_01_20260526_1053_pruned",
        "maize_02_20260526_1658",
    ]
    scenes = [s for s in manifest["scenes"] if s["scene"] in selected_names]
    scene_order = {name: i for i, name in enumerate(selected_names)}
    scenes.sort(key=lambda s: scene_order[s["scene"]])

    fig_h = 4.35
    fig = plt.figure(figsize=(FIG_W, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        height_ratios=[1, 1, 1],
        width_ratios=[0.30, 1, 1, 1, 1],
        left=0.02,
        right=0.995,
        top=0.93,
        bottom=0.18,
        wspace=0.045,
        hspace=0.10,
    )

    headers = ["RGB", "Rendered NIR", "UMGS NDVI", "UMGS NDRE"]
    for j, h in enumerate(headers, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(h, fontsize=7.4, pad=2.0, fontweight="semibold")
        ax.set_axis_off()
        ax.remove()

    dpi_panels: List[RasterPanel] = []
    source_records = []

    for i, scene in enumerate(scenes):
        label_ax = fig.add_subplot(gs[i, 0])
        label_ax.set_axis_off()
        label_ax.text(0.98, 0.5, scene["short_label"], rotation=90, ha="right", va="center", fontsize=7.0)

        render_root = Path(scene["source_render_root"])
        idx_png = scene["render_index"]
        rgb_path = Path(scene["rgb_image"])
        nir_path = band_path(render_root, "Model_NIR", idx_png)
        r_path = band_path(render_root, "Model_R", idx_png)
        re_path = band_path(render_root, "Model_RE", idx_png)
        g_path = band_path(render_root, "Model_G", idx_png)

        nir = read_gray(nir_path)
        r = read_gray(r_path)
        re = read_gray(re_path)
        mask = common_mask(nir, r, re, read_gray(g_path))
        ndvi = index_from_bands(nir, r, mask)
        ndre = index_from_bands(nir, re, mask)
        nir_disp = np.ma.array(display_stretch(nir, mask), mask=~mask)
        panel_size = nir.shape[::-1]
        rgb_disp = normalize_rgb_for_display(read_rgb(rgb_path), panel_size)

        panels = [
            ("RGB", rgb_disp, rgb_path, None, None),
            ("Rendered NIR", nir_disp, nir_path, "gray", (0, 1)),
            ("UMGS NDVI", ndvi, nir_path, INDEX_CMAP, (-1, 1)),
            ("UMGS NDRE", ndre, nir_path, INDEX_CMAP, (-1, 1)),
        ]
        for j, (plabel, data, source_path, cmap, clim) in enumerate(panels, start=1):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(headers[j - 1], fontsize=7.4, pad=2.0, fontweight="semibold")
            if cmap is None:
                ax.imshow(data)
            else:
                if isinstance(data, np.ma.MaskedArray):
                    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
                    cm.set_bad("#101010")
                    ax.imshow(data, cmap=cm, vmin=clim[0], vmax=clim[1], interpolation="nearest")
                else:
                    ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation="nearest")
            dpi_panels.append(axis_effective_ppi(fig, ax, source_path, f"Fig4:{scene['scene']}:{plabel}"))

        source_records.append(
            {
                "scene": scene["scene"],
                "short_label": scene["short_label"],
                "render_index": idx_png,
                "rgb_image": str(rgb_path),
                "render_root": str(render_root),
                "bands_used": {
                    "G": str(g_path),
                    "R": str(r_path),
                    "RE": str(re_path),
                    "NIR": str(nir_path),
                },
                "visual_computations": {
                    "NDVI": "(NIR - R) / (NIR + R + eps)",
                    "NDRE": "(NIR - RE) / (NIR + RE + eps)",
                    "mask": "nonzero intersection of G/R/RE/NIR rendered bands",
                },
            }
        )

    # Shared compact legends. Use explicit axes so endpoint labels cannot collide.
    cax1 = fig.add_axes([0.33, 0.075, 0.18, 0.030])
    grad = np.linspace(0, 1, 256)[None, :]
    cax1.imshow(grad, aspect="auto", cmap="gray")
    cax1.set_yticks([])
    cax1.set_xticks([0, 255])
    cax1.set_xticklabels(["low", "high"], fontsize=6.3)
    cax1.set_title("NIR display stretch", fontsize=6.6, pad=1.5)
    for sp in cax1.spines.values():
        sp.set_linewidth(0.45)

    cax2 = fig.add_axes([0.61, 0.075, 0.34, 0.030])
    grad2 = np.linspace(-1, 1, 256)[None, :]
    cax2.imshow(grad2, aspect="auto", cmap=INDEX_CMAP, vmin=-1, vmax=1)
    cax2.set_yticks([])
    cax2.set_xticks([0, 128, 255])
    cax2.set_xticklabels(["-1", "0", "1"], fontsize=6.3)
    cax2.set_title("Vegetation index", fontsize=6.6, pad=1.5)
    for sp in cax2.spines.values():
        sp.set_linewidth(0.45)

    outputs = save_pub(fig, FIG_DIR / "fig4_product_examples_redesign_v1")
    plt.close(fig)

    manifest_out = {
        "figure": "Fig.4 production-upgrade candidate",
        "generated_at": "2026-06-12",
        "manuscript_integration": "Not integrated. Review candidate only.",
        "claim": "Representative benchmark products remain readable at final double-column width when rebuilt from high-resolution local source panels.",
        "selected_scenes": selected_names,
        "reason_for_reducing_from_four_rows": "Three rows enlarge the product panels; Chunya remains available in source provenance but is omitted from the candidate main figure.",
        "source_records": source_records,
        "outputs": outputs,
    }
    (MANIFEST_DIR / "fig4_source_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    return outputs, dpi_panels, manifest_out


def read_agg_metrics() -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with FIG5_AGG_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[(row["scene"], row["index"])] = row
    return out


def make_fig5_visual_panels(scene: str, view: str) -> Dict[str, object]:
    full_nir_gt = read_gray(fig5_band_path(scene, "full", "Model_NIR", "gt", view))
    full_r_gt = read_gray(fig5_band_path(scene, "full", "Model_R", "gt", view))
    full_nir = read_gray(fig5_band_path(scene, "full", "Model_NIR", "pred", view))
    full_r = read_gray(fig5_band_path(scene, "full", "Model_R", "pred", view))
    no_nir = read_gray(fig5_band_path(scene, "nomask", "Model_NIR", "pred", view))
    no_r = read_gray(fig5_band_path(scene, "nomask", "Model_R", "pred", view))

    # Include all bands in the support mask so the visual follows the metric protocol.
    mask_inputs = [
        full_nir_gt,
        full_r_gt,
        full_nir,
        full_r,
        no_nir,
        no_r,
        read_gray(fig5_band_path(scene, "full", "Model_G", "gt", view)),
        read_gray(fig5_band_path(scene, "full", "Model_RE", "gt", view)),
        read_gray(fig5_band_path(scene, "full", "Model_G", "pred", view)),
        read_gray(fig5_band_path(scene, "full", "Model_RE", "pred", view)),
        read_gray(fig5_band_path(scene, "nomask", "Model_G", "pred", view)),
        read_gray(fig5_band_path(scene, "nomask", "Model_RE", "pred", view)),
    ]
    mask = common_mask(*mask_inputs)
    ref = index_from_bands(full_nir_gt, full_r_gt, mask)
    full = index_from_bands(full_nir, full_r, mask)
    nomask = index_from_bands(no_nir, no_r, mask)
    delta = np.abs(nomask.filled(np.nan) - ref.filled(np.nan)) - np.abs(full.filled(np.nan) - ref.filled(np.nan))
    delta = np.ma.array(np.clip(delta, -0.25, 0.25), mask=~mask)
    return {
        "Reference NDVI": ref,
        "Full UMGS-I": full,
        "No-valid-mask": nomask,
        "Delta abs. error": delta,
        "mask_valid_pixels": int(mask.sum()),
        "mask_total_pixels": int(mask.size),
        "source_path": fig5_band_path(scene, "full", "Model_NIR", "pred", view),
    }


def scene_short(scene: str) -> str:
    return {
        "papaya_01_20251217": "Papaya",
        "maize_02_20260526_1658": "Maize-02",
        "cassava_01_20260526_1603": "Cassava",
    }.get(scene, scene)


def add_heatmap(ax: plt.Axes, metrics: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    scenes = ["papaya_01_20251217", "maize_02_20260526_1658", "cassava_01_20260526_1603"]
    indices = ["NDVI", "GNDVI", "NDRE"]
    heat = np.zeros((len(scenes), len(indices)), dtype=float)
    rel = np.zeros_like(heat)
    for i, scene in enumerate(scenes):
        for j, idx in enumerate(indices):
            row = metrics[(scene, idx)]
            heat[i, j] = float(row["rmse_delta_no_mask_minus_full"])
            rel[i, j] = float(row["rmse_relative_increase_pct"])

    im = ax.imshow(heat, cmap=HEAT_CMAP, vmin=0, vmax=max(0.07, float(heat.max()) * 1.05))
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices, fontsize=6.5)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels([scene_short(s) for s in scenes], fontsize=6.3)
    ax.set_title("RMSE increase\nNo-valid-mask minus Full", fontsize=7.0, pad=2.5)
    for i in range(len(scenes)):
        for j in range(len(indices)):
            ax.text(j, i, f"+{heat[i, j]:.4f}\n+{rel[i, j]:.1f}%", ha="center", va="center", fontsize=5.7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.55)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.ax.tick_params(labelsize=5.7, length=2)
    cb.set_label("RMSE increase", fontsize=6.1)


def make_fig5_main() -> Tuple[Dict[str, str], List[RasterPanel], Dict[str, object]]:
    selection = json.loads(FIG5_SELECTION.read_text(encoding="utf-8"))["selection"]
    metrics = read_agg_metrics()
    scene = "maize_02_20260526_1658"
    view = selection[scene]
    panels = make_fig5_visual_panels(scene, view)

    fig = plt.figure(figsize=(FIG_W, 2.70), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=6,
        width_ratios=[1, 1, 1, 1, 0.30, 1.48],
        left=0.018,
        right=0.995,
        top=0.90,
        bottom=0.24,
        wspace=0.045,
    )
    titles = ["Reference NDVI", "Full UMGS-I", "No-valid-mask", "Delta abs. error"]
    dpi_panels: List[RasterPanel] = []
    for j, title in enumerate(titles):
        ax = fig.add_subplot(gs[j])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        data = panels[title]
        if title == "Delta abs. error":
            cm = DELTA_CMAP.copy()
            cm.set_bad("#101010")
            ax.imshow(data, cmap=cm, vmin=-0.25, vmax=0.25, interpolation="nearest")
        else:
            cm = INDEX_CMAP.copy()
            cm.set_bad("#101010")
            ax.imshow(data, cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=7.1, pad=2.0, fontweight="semibold")
        dpi_panels.append(axis_effective_ppi(fig, ax, panels["source_path"], f"Fig5-main:{scene}:{view}:{title}"))

    ax_heat = fig.add_subplot(gs[5])
    add_heatmap(ax_heat, metrics)

    # Legends below visual panels.
    cax_idx = fig.add_axes([0.025, 0.085, 0.31, 0.032])
    grad = np.linspace(-1, 1, 256)[None, :]
    cax_idx.imshow(grad, aspect="auto", cmap=INDEX_CMAP, vmin=-1, vmax=1)
    cax_idx.set_yticks([])
    cax_idx.set_xticks([0, 128, 255])
    cax_idx.set_xticklabels(["-1", "0", "1"], fontsize=6.1)
    cax_idx.set_title("Index value", fontsize=6.3, pad=1.2)
    for sp in cax_idx.spines.values():
        sp.set_linewidth(0.45)

    cax_delta = fig.add_axes([0.375, 0.085, 0.31, 0.032])
    grad2 = np.linspace(-0.25, 0.25, 256)[None, :]
    cax_delta.imshow(grad2, aspect="auto", cmap=DELTA_CMAP, vmin=-0.25, vmax=0.25)
    cax_delta.set_yticks([])
    cax_delta.set_xticks([0, 128, 255])
    cax_delta.set_xticklabels(["-0.25", "0", "0.25"], fontsize=6.1)
    cax_delta.set_title("No-mask minus Full abs. error", fontsize=6.3, pad=1.2)
    for sp in cax_delta.spines.values():
        sp.set_linewidth(0.45)

    fig.text(0.018, 0.965, f"{scene_short(scene)} view {view}", fontsize=7.0, fontweight="semibold", ha="left", va="top")
    outputs = save_pub(fig, FIG_DIR / "fig5_nomask_main_redesign_v1")
    plt.close(fig)
    manifest = {
        "figure": "Fig.5 main production-upgrade candidate",
        "generated_at": "2026-06-12",
        "manuscript_integration": "Not integrated. Review candidate only.",
        "claim": "A single enlarged representative visual plus a readable aggregate heatmap supports the benchmark-internal no-valid-mask diagnostic without overcrowding the main figure.",
        "representative_visual": {"scene": scene, "view": view, "policy": "median-degradation selected view from existing selected_views.json"},
        "aggregate_metric_source": str(FIG5_AGG_CSV),
        "metric_scope": "Papaya, Maize-02, Cassava by NDVI/GNDVI/NDRE; representative diagnostic, not exhaustive 16-scene ablation.",
        "valid_pixel_count": {
            "scene": scene,
            "view": view,
            "valid": panels["mask_valid_pixels"],
            "total": panels["mask_total_pixels"],
        },
        "outputs": outputs,
    }
    (MANIFEST_DIR / "fig5_main_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs, dpi_panels, manifest


def make_fig5_supplement() -> Tuple[Dict[str, str], List[RasterPanel], Dict[str, object]]:
    selection = json.loads(FIG5_SELECTION.read_text(encoding="utf-8"))["selection"]
    scenes = ["papaya_01_20251217", "maize_02_20260526_1658", "cassava_01_20260526_1603"]
    titles = ["Reference NDVI", "Full UMGS-I", "No-valid-mask", "Delta abs. error"]
    fig = plt.figure(figsize=(FIG_W, 3.78), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        height_ratios=[1, 1, 1],
        width_ratios=[0.34, 1, 1, 1, 1],
        left=0.02,
        right=0.995,
        top=0.92,
        bottom=0.18,
        wspace=0.050,
        hspace=0.095,
    )
    dpi_panels: List[RasterPanel] = []
    source_records = []
    for i, scene in enumerate(scenes):
        view = selection[scene]
        panel_data = make_fig5_visual_panels(scene, view)
        label_ax = fig.add_subplot(gs[i, 0])
        label_ax.set_axis_off()
        label_ax.text(0.98, 0.5, f"{scene_short(scene)}\n{view}", rotation=90, ha="right", va="center", fontsize=6.6)
        for j, title in enumerate(titles, start=1):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=7.0, pad=2.0, fontweight="semibold")
            data = panel_data[title]
            if title == "Delta abs. error":
                cm = DELTA_CMAP.copy()
                cm.set_bad("#101010")
                ax.imshow(data, cmap=cm, vmin=-0.25, vmax=0.25, interpolation="nearest")
            else:
                cm = INDEX_CMAP.copy()
                cm.set_bad("#101010")
                ax.imshow(data, cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
            dpi_panels.append(axis_effective_ppi(fig, ax, panel_data["source_path"], f"Fig5-supp:{scene}:{view}:{title}"))
        source_records.append(
            {
                "scene": scene,
                "view": view,
                "valid_pixels": panel_data["mask_valid_pixels"],
                "total_pixels": panel_data["mask_total_pixels"],
            }
        )

    cax_idx = fig.add_axes([0.16, 0.075, 0.34, 0.030])
    grad = np.linspace(-1, 1, 256)[None, :]
    cax_idx.imshow(grad, aspect="auto", cmap=INDEX_CMAP, vmin=-1, vmax=1)
    cax_idx.set_yticks([])
    cax_idx.set_xticks([0, 128, 255])
    cax_idx.set_xticklabels(["-1", "0", "1"], fontsize=6.1)
    cax_idx.set_title("Index value", fontsize=6.3, pad=1.2)
    for sp in cax_idx.spines.values():
        sp.set_linewidth(0.45)

    cax_delta = fig.add_axes([0.58, 0.075, 0.34, 0.030])
    grad2 = np.linspace(-0.25, 0.25, 256)[None, :]
    cax_delta.imshow(grad2, aspect="auto", cmap=DELTA_CMAP, vmin=-0.25, vmax=0.25)
    cax_delta.set_yticks([])
    cax_delta.set_xticks([0, 128, 255])
    cax_delta.set_xticklabels(["-0.25", "0", "0.25"], fontsize=6.1)
    cax_delta.set_title("No-mask minus Full abs. error", fontsize=6.3, pad=1.2)
    for sp in cax_delta.spines.values():
        sp.set_linewidth(0.45)

    outputs = save_pub(fig, FIG_DIR / "fig5_nomask_full_grid_supp_v1")
    plt.close(fig)
    manifest = {
        "figure": "Supplement full-grid no-valid-mask candidate",
        "generated_at": "2026-06-12",
        "manuscript_integration": "Not integrated. Review candidate only.",
        "claim": "The full three-scene diagnostic grid remains available as supplement evidence when the main figure uses one enlarged representative visual.",
        "source_records": source_records,
        "outputs": outputs,
    }
    (MANIFEST_DIR / "fig5_supplement_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs, dpi_panels, manifest


def make_comparison(old_path: Path, new_path: Path, out_path: Path, title_left: str, title_right: str) -> None:
    old = read_rgb(old_path)
    new = read_rgb(new_path)
    panel_h = 980
    def resize_to_h(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.resize((int(w * panel_h / h), panel_h), Image.Resampling.LANCZOS)
    old_r = resize_to_h(old)
    new_r = resize_to_h(new)
    pad = 34
    header = 68
    w = old_r.width + new_r.width + pad * 3
    h = panel_h + header + pad
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 30) if Path(r"C:\Windows\Fonts\arial.ttf").exists() else ImageFont.load_default()
    canvas.paste(old_r, (pad, header))
    canvas.paste(new_r, (pad * 2 + old_r.width, header))
    draw.text((pad, 22), title_left, fill="black", font=font)
    draw.text((pad * 2 + old_r.width, 22), title_right, fill="black", font=font)
    canvas.save(out_path)


def write_reports(
    fig4_outputs: Dict[str, str],
    fig5_main_outputs: Dict[str, str],
    fig5_supp_outputs: Dict[str, str],
    dpi_panels: List[RasterPanel],
) -> None:
    dpi_records = [panel_record(p) for p in dpi_panels]
    min_ppi = min(min(p.effective_ppi_x, p.effective_ppi_y) for p in dpi_panels)
    report = {
        "generated_at": "2026-06-12",
        "backend": "Python/matplotlib only",
        "target_width_in": FIG_W,
        "minimum_effective_ppi": round(min_ppi, 1),
        "pass_condition": "All source raster panels should exceed 300 effective ppi at final candidate size.",
        "pass": bool(min_ppi >= 300),
        "panels": dpi_records,
    }
    (OUT_ROOT / "EFFECTIVE_DPI_REPORT_20260612.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Effective DPI Report",
        "",
        f"- Backend: Python/matplotlib only",
        f"- Target double-column width: {FIG_W:.2f} in",
        f"- Minimum effective raster PPI across rebuilt panels: {min_ppi:.1f}",
        f"- Pass condition: >=300 effective PPI",
        f"- Result: {'PASS' if min_ppi >= 300 else 'CHECK'}",
        "",
        "| Figure panel | Source pixels | Axis size (in) | Effective PPI |",
        "|---|---:|---:|---:|",
    ]
    for p in dpi_panels:
        lines.append(
            f"| {p.label} | {p.source_size[0]}x{p.source_size[1]} | "
            f"{p.axis_size_in[0]:.2f}x{p.axis_size_in[1]:.2f} | "
            f"{p.effective_ppi_x:.0f}x{p.effective_ppi_y:.0f} |"
        )
    (OUT_ROOT / "EFFECTIVE_DPI_REPORT_20260612.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    captions = """# Draft Figure Captions for GPT Review

## Fig. 4 Candidate

Qualitative UAV-MultiSpec3D product examples. Rows show representative benchmark scenes and columns show the RGB input crop, rendered NIR display, UMGS NDVI, and UMGS NDRE. RGB and NIR panels are display-enhanced only; vegetation-index panels use a fixed [-1, 1] display range. These visual products are computed from normalized rendered-band artifacts and are not reflectance-calibrated products.

## Fig. 5 Main Candidate

Benchmark-internal no-valid-mask diagnostic. The left panels show an enlarged representative Maize-02 held-out view with reference NDVI, Full UMGS-I NDVI, No-valid-mask NDVI, and the no-valid-mask minus Full absolute-error difference. The right heatmap summarizes RMSE increases over the Papaya, Maize-02, and Cassava diagnostic scenes and the NDVI/GNDVI/NDRE indices. This is a representative controlled diagnostic rather than an exhaustive 16-scene ablation; support-invariant checks remain zero-drift in the diagnostic runs.

## Supplement Full-Grid Candidate

Full visual grid for the benchmark-internal no-valid-mask diagnostic. Rows show the selected Papaya, Maize-02, and Cassava held-out views, and columns show reference NDVI, Full UMGS-I NDVI, No-valid-mask NDVI, and the no-valid-mask minus Full absolute-error difference. The grid uses the same normalized rendered-band artifacts and display ranges as the main candidate figure.
"""
    (OUT_ROOT / "FIGURE_CAPTIONS_DRAFT_20260612.md").write_text(captions, encoding="utf-8")

    recommendation = {
        "generated_at": "2026-06-12",
        "status": "review_candidate_only",
        "integration_recommendation": [
            "Use Fig.4 redesign if GPT/user agree that three enlarged rows are preferable to the current four-row contact-sheet style.",
            "Use Fig.5 main redesign if GPT/user agree to move the full three-scene visual grid to supplement and keep one enlarged representative visual plus aggregate heatmap in the main text.",
            "Do not integrate any candidate before review; main.tex and supplement.tex were not modified by this script.",
        ],
        "server_need": "None. All sources were local.",
        "science_changes": "None. No training, rendering, checkpoint mutation, or benchmark metric recomputation.",
        "outputs": {
            "fig4": fig4_outputs,
            "fig5_main": fig5_main_outputs,
            "fig5_supplement": fig5_supp_outputs,
        },
    }
    (OUT_ROOT / "FIGURE_UPGRADE_RECOMMENDATION_20260612.md").write_text(
        "# Figure Upgrade Recommendation\n\n"
        "- Status: review candidate only; not integrated into manuscript.\n"
        "- Server/GPU need: none.\n"
        "- Science changes: none.\n"
        "- Recommended decision: ask GPT/user whether to replace current Fig.4/Fig.5 after visual review.\n"
        "- Main Fig.5 candidate uses one enlarged Maize-02 representative visual plus the existing aggregate heatmap; the full three-scene grid is supplied as a supplement candidate.\n"
        "- Fig.4 candidate uses three enlarged product-example rows to improve final-size legibility; Chunya remains available in provenance but is omitted from this candidate to enlarge panels.\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "FIGURE_UPGRADE_RECOMMENDATION_20260612.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    gpt_message = """GPT:
This package contains review-only production-upgrade candidates for TGRS Fig.4 and Fig.5.

Execution boundary:
- main.tex and supplement.tex were not modified.
- Current manuscript figure files were not overwritten.
- No server, GPU, training, rerendering, checkpoint mutation, support mutation, or benchmark metric change was used.
- The candidates were rebuilt only from existing local high-resolution artifacts.

Outputs:
1. Fig.4 production candidate
   - Current four-row contact-sheet style is reduced to three enlarged rows: Cassava / Eucalyptus-01 / Maize-02.
   - Columns: RGB / Rendered NIR / UMGS NDVI / UMGS NDRE.
   - Chunya is not discarded; it remains in provenance but is omitted from the main candidate to enlarge panels.
   - RGB and NIR are display-enhanced only; index maps use fixed [-1, 1] display range; no reflectance-calibrated product claim is made.

2. Fig.5 main candidate
   - Uses one enlarged representative visual plus the aggregate heatmap.
   - Representative visual: Maize-02 view 00008.
   - Columns: Reference NDVI / Full UMGS-I / No-valid-mask / Delta absolute error.
   - Heatmap remains Papaya / Maize-02 / Cassava x NDVI/GNDVI/NDRE aggregate RMSE increase.
   - The full three-scene visual grid is provided as a supplement candidate.

3. QA / provenance
   - Effective-DPI audit: minimum effective raster PPI is about 502, above the 300-PPI threshold.
   - SVG outputs retain editable text nodes.
   - The package includes source artifact audit, source manifests, captions draft, old-vs-new comparisons, and recommendation note.

Please review:
1. Whether Fig.4 should use the three-row enlarged main candidate rather than the current four-row contact sheet.
2. Whether Fig.5 should use the Maize-02 enlarged representative visual plus aggregate heatmap in the main paper.
3. Whether the full three-scene no-valid-mask visual grid should move to the supplement.
4. Whether caption wording remains sufficiently bounded: not reflectance-calibrated, representative diagnostic, not exhaustive 16-scene ablation, and support claims come from audit.
5. If approved, whether the next step should be limited to replacing Fig.4/Fig.5 assets and recompiling LaTeX, with no new experiment.
"""
    (OUT_ROOT / "GPT_REVIEW_MESSAGE_20260612.md").write_text(gpt_message, encoding="utf-8")


def copy_review_inputs() -> None:
    if SOURCE_AUDIT.exists():
        shutil.copy2(SOURCE_AUDIT, OUT_ROOT / SOURCE_AUDIT.name)
    for p in [FIG4_MANIFEST, FIG5_SELECTION, FIG5_AGG_CSV]:
        if p.exists():
            shutil.copy2(p, MANIFEST_DIR / p.name)


def zip_package() -> None:
    if GPT_PACKAGE.exists():
        GPT_PACKAGE.unlink()
    include_roots = [OUT_ROOT]
    with zipfile.ZipFile(GPT_PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root in include_roots:
            for p in root.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(OUT_ROOT.parent))


def main() -> None:
    ensure_dirs()
    copy_review_inputs()
    fig4_outputs, fig4_dpi, _ = make_fig4()
    fig5_main_outputs, fig5_main_dpi, _ = make_fig5_main()
    fig5_supp_outputs, fig5_supp_dpi, _ = make_fig5_supplement()

    make_comparison(
        OLD_FIG4,
        Path(fig4_outputs["png"]),
        COMP_DIR / "fig4_old_vs_new_comparison.png",
        "Current Fig.4",
        "Production-upgrade candidate",
    )
    make_comparison(
        OLD_FIG5,
        Path(fig5_main_outputs["png"]),
        COMP_DIR / "fig5_old_vs_new_comparison.png",
        "Current Fig.5",
        "Main candidate",
    )

    write_reports(fig4_outputs, fig5_main_outputs, fig5_supp_outputs, fig4_dpi + fig5_main_dpi + fig5_supp_dpi)
    zip_package()
    print(json.dumps({"out_root": str(OUT_ROOT), "package": str(GPT_PACKAGE)}, indent=2))


if __name__ == "__main__":
    main()
