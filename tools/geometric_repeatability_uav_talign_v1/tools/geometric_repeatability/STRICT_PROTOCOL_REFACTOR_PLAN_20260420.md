# Strict Protocol Refactor Plan (2026-04-20)

This plan upgrades the current Building/M01 repeatability pilot from a smoke-test implementation into a reviewer-safer strict protocol run.

## Goal

Replace the current reuse of full-scene sparse/camera assets with a train-union-only shared frame, while keeping the rest of the repeatability evaluator unchanged.

## Protocol Interpretation

- `shared frame` is reconstructed from `Train-O ∪ Train-E` only
- `probe views` are **not** used to build the shared frame or ROI
- `probe views` are post-registered into the frozen shared frame for evaluation only
- the shared train-union frame is frozen during probe registration
- ROI is built from train-union-only sparse points
- odd/even training and probe rendering then use the resulting strict RGB/T datasets plus the existing explicit split files

## Engineering Changes

### 1. Strict dataset materialization

Add a dedicated materializer:

- file: `D:\dataset\FGS\FGS-0202v1\tools\geometric_repeatability\materialize_strict_pose_controlled_dataset.py`

Responsibilities:

- materialize a `train_union` RGB input tree from the original raw RGB images
- rerun `convert-gtgs.py` on that train-union-only input tree with the same COLMAP/GPS-alignment settings used in the original Building preprocessing
- treat the resulting aligned train-union model as the only valid shared-frame source
- materialize an `all_views` RGB input tree containing `train_union + probe_test`
- add probe-view features to the same COLMAP database using the existing camera id
- post-register probe views into the frozen train-union model with:
  - `COLMAP image_registrator`
  - `Mapper.fix_existing_frames = 1`
  - camera refinement disabled
- undistort RGB into a final strict RGB dataset root
- materialize a thermal alias tree with the same registered image names
- undistort thermal into a final strict thermal dataset root
- write a strict protocol manifest containing:
  - source roots
  - split paths
  - shared-frame rule
  - probe-registration rule
  - strict RGB/T dataset roots
  - ROI-source points path

### 2. ROI-source tightening

Use the train-union aligned sparse points directly as the ROI source.

Implementation detail:

- extend `D:\dataset\FGS\FGS-0202v1\tools\geometric_repeatability\evaluator.py`
- add support for COLMAP `points3D.bin` / `points3D.txt` as `build-roi` inputs

This avoids depending on a later full registered model or any probe-influenced sparse export.

### 3. Protocol wording tightening

Update:

- `D:\dataset\FGS\FGS-0202v1\tools\geometric_repeatability\PROTOCOL.md`

New wording clarifies:

- probe views may be post-registered into the frozen train-union frame
- this registration must not change the shared frame, ROI, or thresholds

### 4. Strict long-run entrypoint

Add a resumable Building/M01 strict runner:

- file: `D:\dataset\FGS\FGS-0202v1\tools\geometric_repeatability\run_building_m01_repeatability_strict.ps1`

Stages:

1. materialize strict RGB/T datasets
2. build ROI from train-union-only sparse points
3. train odd RGB
4. train odd T
5. export odd probe bundle
6. train even RGB
7. train even T
8. export even probe bundle
9. build evaluator scene manifest
10. evaluate P/R/F

Resume logic:

- every expensive stage checks for its expected artifact before rerunning
- `status.json` and transcript are updated per stage

## Expected Outputs

Primary run root:

- `F:\databackup\xr6\output\GeometricRepeatability\Building\M01_Strict_v1`

Strict dataset root:

- `F:\databackup\xr6\output\GeometricRepeatability\Building\M01_Strict_v1\strict_dataset`

Key files:

- strict dataset manifest
- train-union-only ROI source points path
- strict RGB dataset root
- strict thermal dataset root
- odd/even model outputs
- odd/even probe bundles
- evaluator metrics/json/csv

## Why this is stricter than the current pilot

The current pilot is valuable as an evaluator smoke test, but it still depends on an existing sparse/camera tree that appears to contain probe-view influence.

This refactor removes that dependency by:

- rebuilding the shared frame from `train_union` only
- using train-union-only sparse points for ROI
- turning probe poses into a pure post-registration step under a frozen shared frame

## Execution Order

1. implement the strict materializer and ROI-source tightening
2. run local sanity checks / compilation
3. launch the strict Building/M01 run
4. inspect whether the strict dataset materialization succeeds cleanly
5. only after that, consider expanding to PVpanel or more methods
