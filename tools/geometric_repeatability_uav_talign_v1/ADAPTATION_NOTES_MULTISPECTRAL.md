# Adaptation Notes for `E:\Multispectral`

This note records how the imported geometric repeatability evaluator should be used in the current UAV multispectral repo.

## 1. Imported location

The evaluator package extracted from `E:\paper\UAV-TAlign\1.zip` is stored at:

- `E:\Multispectral\tools\geometric_repeatability_uav_talign_v1`

The folder name is intentionally versioned so it does not get confused with any future repo-native geometry evaluator.

## 2. Threshold policy for the current repo

The current multispectral datasets do not provide trustworthy global metric scale from GPS/EXIF. Therefore:

- **Do not use the old absolute-meter threshold examples as the default protocol.**
- Use the evaluator's existing paper-facing default:
  - threshold ratios = `0.5%, 1%, 2%` of scene diagonal
  - voxel size = `0.1%` of scene diagonal

This is already the evaluator's default behavior when `--threshold_abs_m` and `--voxel_size_m` are omitted.

For clarity, the imported evaluator package has been lightly adapted so its outputs now refer to:

- `absolute_scene_unit`
- `scene_world_unit`

instead of hard-coding `meter`.

### Rationale

Without trusted global scale, the safest claim is:

- `pose-controlled geometric repeatability`
- `self-consistency`
- `cross-subset stability`

measured in a scene-normalized distance domain, not absolute physical accuracy.

## 3. Recommended paper-facing wording

Use wording like:

> We evaluate pose-controlled cross-subset geometric repeatability using thresholds normalized by the shared ROI scene diagonal.

Avoid wording like:

> We evaluate absolute geometry accuracy in meters.

unless a future dataset with trusted external metric scale is introduced.

## 4. Whether to compare against `ThermalGaussian` / `Thermal3D-GS`

### Short answer

- **Not for the current multispectral main paper.**
- **Possibly yes for a separate thermal/RGB-T line or for a geometry-focused supplemental study.**

### Why not in the current main paper

The current main paper is centered on:

- UAV multispectral reconstruction (`RGB + G/R/RE/NIR`)
- shared Gaussian support
- index-stable product synthesis

`ThermalGaussian` and `Thermal3D-GS` are thermal/RGB-T methods. They are relevant to the separate UAV thermal line, but they are not clean external baselines for the current multispectral main task.

If they are inserted into the current main table, the comparison target becomes ambiguous:

- multispectral reconstruction vs thermal reconstruction
- shared-support index synthesis vs thermal-only rendering/geometry

That is a protocol mismatch, not a clean baseline.

### When they are still useful

These methods remain useful if the goal is specifically:

- a thermal paper (`RGB-T`, `UAV-FGS`, `UAV-TAlign`)
- or a geometry repeatability appendix where all compared methods are evaluated only on the thermal branch under the same fixed-pose protocol

## 5. Practical recommendation for this repo

If we activate geometry evaluation in this repo, the first pilot should be:

1. run the imported evaluator on the current repo's own methods and ablations;
2. use the normalized scene-diagonal thresholds as the only default;
3. keep thermal-method comparisons out of the multispectral main experiment block unless a separate thermal evaluation section is explicitly created.

That preserves task coherence while still allowing the evaluator to strengthen the geometry story.
