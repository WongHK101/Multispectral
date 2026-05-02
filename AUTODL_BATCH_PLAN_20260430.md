# AutoDL Batch Plan 2026-04-30

This document refines `AUTODL_EXPERIMENT_CHECKLIST.md` into the current
execution order requested on 2026-04-30. It is a planning document, not an
automation script. CODEX should still launch, inspect, sync, and clean each
phase manually.

## Current Rule

Paper-facing values should come from AutoDL only. Do not mix old RTX 4090
results into paper tables unless explicitly labeled as historical/debug.

All output roots must remain outside the dataset root:

- dataset root: `/root/autodl-tmp/datasets/Multispectral`
- run root: `/root/autodl-tmp/runs`

## Revised Batch Order

| Batch | Scope | Main work | Reuse policy | Depth policy | Estimated time | Storage if fully kept | Recommended working peak |
|---|---|---|---|---|---:|---:|---:|
| B1 | E3 raw7 | Raw D/RGB COLMAP, MINIMA/RoMA rectification, RGB anchor, 4 band transfers, render, metrics, index, cost | None for raw scenes; this is the base run | Do not run depth in parallel with B1 | Current run: total about 14-18 h; remaining after raw003 about 6-8 h | 350-500 GB | 250-420 GB |
| B2 | E3 official7 | Official aligned preparation, RGB anchor, 4 band transfers, render, metrics, index, cost | No raw rectification/COLMAP; official aligned inputs only | No depth during B2 unless GPU is idle after B2 ends | 7-14 h | 120-300 GB | 100-220 GB |
| B3 | E3 depth14 | Build/freeze scene-level reference geometry, export E3 model depth bundles, evaluate relative-depth, summarize | Reuse E3 trained models and scene splits | Build one reference per scene; E3 depth is evaluated after all E3 scenes finish | 10-25 h, high variance from reference construction | 70-220 GB | 60-160 GB |
| B4 | E4b-zero14 | Joint multispectral appearance training, per-band export views, render, metrics, index, cost | Reuse E3 prepared/rectified scenes and the E3 RGB checkpoint; do not rerun COLMAP, MINIMA, or RGB anchor | After each E4 scene finishes, immediately export/evaluate E4 depth using the already frozen scene reference from B3; sync metrics+depth+cost back to local | 28-55 h | 280-700 GB | 120-260 GB by scene/batch cleanup |

Later batches remain planned but are not part of the immediate sequence above:

- MMSplat official7 external baseline.
- self_m3m MMS retained-subset comparison.
- representative ablations: from-scratch, geometry-unfrozen, E4b-rgb_tied on
  `raw_self`, `raw002`, `ms_lake`.

## Fairness Rules For E4 Reuse

E4b-zero is allowed and expected to reuse E3 artifacts because its method
definition is "RGB-anchored geometry with joint multispectral appearance".

Allowed reuse:

- E3 prepared scene.
- E3 rectified scene for raw scenes.
- E3 RGB checkpoint / Stage-1 geometry anchor.
- Scene-level depth reference built from training RGB views only.
- Frozen train/test split and camera metadata.

Not allowed:

- Reusing E3 band appearance parameters as E4b-zero band banks.
- Rebuilding or tuning the depth reference after inspecting E4 results.
- Writing E4 outputs into E3 output directories.
- Mixing E3 and E4 cost traces.

Cost reporting should keep two fields for E4:

- `e4_incremental_cost`: E4 joint training + export + render + metrics +
  depth export/eval.
- `e4_end_to_end_cost`: shared E3 front-end/RGB anchor cost + E4 incremental
  cost. This is the fair full-pipeline cost if reviewers ask how long E4 takes
  from raw input.

## Per-Scene Sync Rule

After each scene completes, sync these records back to the local metric folder:

- band/RGB quality metrics
- spectral index metrics
- rectification/final QA when applicable
- cost monitor summary
- Gaussian count and exported storage
- depth-reference summary when available
- run root and command/log paths

Local metric snapshots should stay under:

`E:\Multispectral\experiment_metric_records`

This path is ignored by Git.

## Current B1 Status Snapshot

As of 2026-04-30 around 02:56 +08:00:

- completed: `raw_self`, `raw001`, `raw002`, `raw003`
- running: `raw004`
- pending in B1: `raw005`, `raw006`

The B1 run root is:

`/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw7_gpu_colmap_20260429_180200`

The current local metric record root is:

`E:\Multispectral\experiment_metric_records\autodl_e3_raw7_gpu_colmap_20260429_180200`

## Storage Notes

The AutoDL work disk is 600 GB. At 2026-04-30 02:56 +08:00:

- dataset root: about 41 GB
- current B1 run root after `raw003`: about 225 GB
- disk used: about 252 GB
- disk free: about 349 GB

This is enough to finish B1 if no unexpected large temporary products remain,
but not enough to keep all B1+B2+B3+B4 full artifacts online at once.

Before starting B2, CODEX should summarize B1 and then decide whether to
archive/remove bulky non-essential artifacts while preserving:

- final metrics and summaries
- command logs and cost traces
- final checkpoints/PLYs needed for reproduction or visual inspection
- rectification/depth audit files
- failure logs, if any

No automatic deletion should be performed without an explicit phase boundary
decision.
