# Depth-Reference Geometry Tools

This folder contains the v1 reference-depth geometry evaluator imported for the
multispectral paper. It is intentionally separate from the older
`tools/geometric_repeatability_uav_talign_v1` odd/even evaluator.

## Paper-Facing Goal

The target experiment is:

1. build a training-side RGB MVS reference mesh/depth;
2. render that reference into held-out probe cameras;
3. export trained Gaussian model depth/opacity in the same cameras;
4. compare `D_model - D_ref` with front-intrusion and agreement metrics.

Use wording such as:

- `reference-depth-based geometric evaluation`
- `held-out geometry consistency against a training-only MVS reference`
- `front-intrusion depth analysis`

Do not describe this as ground-truth or absolute geometry accuracy.

## Scale Policy For This Repo

Meter thresholds are valid only for scenes with verified metric scale. Current
raw multispectral runs are not guaranteed to be metric because the default raw
pipeline does not enable COLMAP `model_aligner`.

Use two explicit modes:

- `metric_verified`: physical thresholds such as `0.10/0.25/0.50/1.00 m`
  are allowed.
- `scene_normalized`: thresholds are derived from a frozen scene diagonal or
  depth scale and must be reported as normalized scene units, not meters.
- `relative_depth`: thresholds are ratios on `(D_model - D_ref) / D_ref`.
  This is the preferred cross-scene paper table mode when metric scale is not
  verified, because it normalizes the tolerance by the probe-view reference
  depth instead of using a fixed scene-unit distance.

## COLMAP / GPU Policy

Use one COLMAP selection policy across this repo:

1. `SIGS_COLMAP_EXECUTABLE`
2. `COLMAP_EXECUTABLE`
3. `~/opt/colmap-cuda/bin/colmap`
4. `~/opt/colmap-cuda-3.7/bin/colmap`
5. `colmap` from `PATH`

`build_depth_reference.py` defaults to this same order. For dense reference
construction, PatchMatch tries GPU first with:

```text
--PatchMatchStereo.gpu_index 0
```

Launchers should scope the intended physical GPU with `CUDA_VISIBLE_DEVICES`.
If the GPU attempt fails or OOMs, the script retries once with
`CUDA_VISIBLE_DEVICES` cleared using the same COLMAP executable. Some COLMAP
builds do not support CPU dense stereo; in that case the retry fails explicitly
and writes a `patch_match_failure_audit.json` instead of silently switching to a
different COLMAP binary.

## Core Files

- `DEPTH_REFERENCE_PROTOCOL.md`: protocol definition from the imported package.
- `build_depth_reference.py`: build the training-side RGB MVS reference.
- `export_gaussian_probe_bundle.py`: export model depth/opacity bundles.
- `evaluate_depth_reference.py`: compute reference-depth metrics.
- `summarize_depth_reference_methods.py`: aggregate multiple method outputs.
- `make_smoke_reference_from_bundle.py`: integration smoke helper only.
- `write_depth_adapter_manifest.py`: writes renderer depth adapter manifests.

## Main Metrics

- `FrontIntrusionRate@delta`: lower is better.
- `FrontIntrusionMagnitude@delta`: lower is better.
- `TooDeepRate@delta`: lower is better.
- `MissingRate`: lower is better.
- `DepthAgreementRate@delta`: higher is better, enabled with
  `--enable_agreement_metrics`.
- `AbsDepthError_Mean`, `AbsDepthError_Median`, `SignedDepthBias_Mean`.

For `evaluate_depth_reference.py --error_mode relative_depth`, the same metric
names are reported, but `delta` is a dimensionless ratio. For example,
`delta=0.05` means a 5% relative depth tolerance.

## Smoke Test Path

For integration smoke only, a Gaussian RGB depth bundle can be converted into a
proxy reference with `make_smoke_reference_from_bundle.py`. This validates
export/evaluate plumbing on real scene cameras and real model outputs, but it is
not a paper-facing reference because the reference comes from a Gaussian model
rather than training-only RGB MVS.

Paper-facing runs must use `build_depth_reference.py`.
