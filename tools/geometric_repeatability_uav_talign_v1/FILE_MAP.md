# File Map

## Runnable Evaluator Core

- `tools/geometric_repeatability/evaluator.py`
  Main geometric repeatability evaluator.
- `tools/geometric_repeatability/build_scene_manifest.py`
  Converts odd/even split bundles into one evaluator manifest.
- `tools/geometric_repeatability/materialize_colmap_odd_even_split.py`
  Deterministic odd/even/probe split generation from COLMAP image names.
- `tools/geometric_repeatability/materialize_rgbt_split_view_dataset.py`
  Creates folder-based split datasets for RGB/T methods.
- `tools/geometric_repeatability/plot_repeatability_curves.py`
  Plots threshold-vs-repeatability curves from `metrics.csv`.
- `tools/geometric_repeatability/sanity_tests.py`
  Synthetic consistency tests for the evaluator.
- `utils/read_write_model.py`
  COLMAP model reader used by the evaluator and strict dataset materializer.
- `scene/colmap_loader.py`
  COLMAP image-name reader used by split construction.

## Protocol / Documentation

- `tools/geometric_repeatability/PROTOCOL.md`
  Paper-facing strict protocol definition.
- `tools/geometric_repeatability/README.md`
  Original in-repo module README.
- `tools/geometric_repeatability/STRICT_PROTOCOL_REFACTOR_PLAN_20260420.md`
  Refactor notes from the strict protocol hardening stage.

## Repo-Coupled Export / Orchestration Snapshots

- `tools/geometric_repeatability/export_gaussian_probe_bundle.py`
  Exports per-probe-view depth and opacity bundles from repo Gaussian models.
- `tools/geometric_repeatability/materialize_strict_pose_controlled_dataset.py`
  Builds the strict train-union-only shared-frame dataset.
- `tools/geometric_repeatability/run_building_m01_repeatability_pilot.ps1`
  Earlier pilot runner.
- `tools/geometric_repeatability/run_building_m01_repeatability_strict.ps1`
  Strict runner for our method.
- `tools/geometric_repeatability/run_ommg_repeatability_strict.ps1`
  Strict runner for OMMG.
- `tools/geometric_repeatability/run_tg_variant_repeatability_strict.ps1`
  Strict runner for MFTG/MSMG.
- `tools/geometric_repeatability/run_thermal3d_repeatability_strict.ps1`
  Strict runner for Thermal3D-GS.
