# Geometric Repeatability Support Package

This package is a clean extraction of the geometric repeatability / self-consistency evaluation code that was added after the original MM26 code submission.

It is intended for rebuttal-stage transparency: reviewers can inspect the protocol, the evaluator, the split construction logic, and the method wrappers we used for the strict geometry-repeatability experiments.

## What This Package Contains

This package preserves the original repo-relative layout for the geometry module:

- `tools/geometric_repeatability/`
- `utils/read_write_model.py`
- `scene/colmap_loader.py`

This layout is intentional so that the evaluator-side scripts still resolve imports correctly.

## Core Files

The most important files are:

- `tools/geometric_repeatability/PROTOCOL.md`
  Fixed paper-facing protocol definition.
- `tools/geometric_repeatability/evaluator.py`
  Core evaluator. Builds ROI from sparse points and computes bidirectional precision / recall / F-score.
- `tools/geometric_repeatability/build_scene_manifest.py`
  Merges odd/even probe bundles into the evaluator-facing manifest.
- `tools/geometric_repeatability/materialize_colmap_odd_even_split.py`
  Builds deterministic `probe_test`, `train_union`, `train_odd`, and `train_even` lists from a COLMAP scene.
- `tools/geometric_repeatability/materialize_rgbt_split_view_dataset.py`
  Creates folder-based split datasets for methods that expect explicit train/test directories.
- `tools/geometric_repeatability/plot_repeatability_curves.py`
  Plots threshold-vs-repeatability curves from `metrics.csv` outputs.
- `tools/geometric_repeatability/sanity_tests.py`
  Synthetic checks for determinism and metric behavior.

The following files are also included because they were part of the actual strict-protocol execution path, but they are more repo-coupled:

- `tools/geometric_repeatability/export_gaussian_probe_bundle.py`
- `tools/geometric_repeatability/materialize_strict_pose_controlled_dataset.py`
- `tools/geometric_repeatability/run_building_m01_repeatability_strict.ps1`
- `tools/geometric_repeatability/run_ommg_repeatability_strict.ps1`
- `tools/geometric_repeatability/run_tg_variant_repeatability_strict.ps1`
- `tools/geometric_repeatability/run_thermal3d_repeatability_strict.ps1`

## Dependency Tiers

### A. Standalone evaluator tier

These files are runnable from this package alone:

- `tools/geometric_repeatability/evaluator.py`
- `tools/geometric_repeatability/build_scene_manifest.py`
- `tools/geometric_repeatability/materialize_colmap_odd_even_split.py`
- `tools/geometric_repeatability/materialize_rgbt_split_view_dataset.py`
- `tools/geometric_repeatability/plot_repeatability_curves.py`
- `tools/geometric_repeatability/sanity_tests.py`
- `utils/read_write_model.py`
- `scene/colmap_loader.py`

Minimal Python requirements for this tier are listed in `requirements_core.txt`.

### B. Repo-coupled export / orchestration tier

These files reflect the exact project-side export and orchestration logic used in our experiments, but they assume the full project codebase / training environments from the main submission:

- `tools/geometric_repeatability/export_gaussian_probe_bundle.py`
- `tools/geometric_repeatability/materialize_strict_pose_controlled_dataset.py`
- the `run_*.ps1` scripts

In other words:

- the evaluator core is independently inspectable and runnable;
- the training/export wrappers are provided for transparency, but they are not meant to be executed in isolation without the rest of the project repository.

## Recommended Rebuttal Usage

For reviewer-facing support, the most useful inspection path is:

1. Read `tools/geometric_repeatability/PROTOCOL.md`
2. Read `tools/geometric_repeatability/evaluator.py`
3. Read `tools/geometric_repeatability/build_scene_manifest.py`
4. Read `tools/geometric_repeatability/materialize_colmap_odd_even_split.py`
5. Read `tools/geometric_repeatability/sanity_tests.py`

That sequence covers protocol definition, split construction, manifest construction, actual metric computation, and sanity validation.

## Quick Start

Open a PowerShell terminal in the package root.

### 1. Run the evaluator sanity tests

```powershell
python .\tools\geometric_repeatability\sanity_tests.py
```

### 2. Build a shared ROI from a training-side sparse point cloud

```powershell
python .\tools\geometric_repeatability\evaluator.py build-roi `
  --scene_name Building `
  --points_path C:\path\to\points3D.bin `
  --out .\roi.json
```

### 3. Build explicit odd/even/probe split lists from a COLMAP scene

```powershell
python .\tools\geometric_repeatability\materialize_colmap_odd_even_split.py `
  --source_path C:\path\to\scene_root `
  --out_dir .\split_lists `
  --llffhold 8
```

### 4. Merge odd/even exported bundles into one scene manifest

```powershell
python .\tools\geometric_repeatability\build_scene_manifest.py `
  --odd_manifest C:\path\to\odd\split_manifest.json `
  --even_manifest C:\path\to\even\split_manifest.json `
  --roi_path .\roi.json `
  --out .\scene_manifest.json
```

### 5. Evaluate one scene

Recommended default for scenes without trustworthy metric scale:

```powershell
python .\tools\geometric_repeatability\evaluator.py evaluate-scene `
  --manifest .\scene_manifest.json `
  --out_dir .\eval_out
```

This uses the paper-facing default:

- thresholds = `0.5%, 1%, 2%` of the ROI scene diagonal
- voxel size = `0.1%` of the ROI scene diagonal

Optional absolute scene/world-unit override, only when a shared metric scale is trustworthy:

```powershell
python .\tools\geometric_repeatability\evaluator.py evaluate-scene `
  --manifest .\scene_manifest.json `
  --out_dir .\eval_out `
  --threshold_abs_m 0.10,0.25,0.50,1.00,2.00 `
  --voxel_size_m 0.05
```

This writes:

- `metrics.json`
- `metrics.csv`
- `manifest_snapshot.json`
- `roi_snapshot.json`
- `odd_points_after_roi.npz`
- `odd_points_after_voxel.npz`
- `even_points_after_roi.npz`
- `even_points_after_voxel.npz`

### 6. Plot threshold-vs-repeatability curves

```powershell
python .\tools\geometric_repeatability\plot_repeatability_curves.py `
  --input ours=C:\path\to\ours\metrics.csv `
  --input baseline=C:\path\to\baseline\metrics.csv `
  --metric fscore `
  --out_png .\curve.png `
  --out_csv .\curve_table.csv
```

## File Map

See `FILE_MAP.md` for a compact file-by-file explanation.

## Notes for Reviewers

- This package contains code only, not large data artifacts or trained models.
- The original submission did not include this geometry-evaluation module; this package was prepared afterward specifically to expose the exact implementation used in the strict repeatability experiments.
- The evaluator itself is deterministic and uses `NumPy + SciPy cKDTree` for the distance computations.

## License / Attribution Notes

- `utils/read_write_model.py` and `scene/colmap_loader.py` retain their original headers and attribution.
- Please keep those headers if this package is forwarded or mirrored.
