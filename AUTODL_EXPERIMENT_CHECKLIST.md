# AutoDL Experiment Checklist

This document defines the post-migration experiment matrix. All paper-facing
numbers should be regenerated on the AutoDL RTX PRO 6000 Blackwell server unless
explicitly marked as historical/debug-only.

## Server Baseline

- Code root: `/root/autodl-tmp/Multispectral`
- Dataset root: `/root/autodl-tmp/datasets/Multispectral`
- Environment: `/root/autodl-tmp/envs/spectralindexgs_bw`
- Python executable: `/root/autodl-tmp/envs/spectralindexgs_bw/bin/python`
- Run root: `/root/autodl-tmp/runs`
- GPU policy: use GPU for all supported CUDA stages, including COLMAP SIFT and
  matching. If a GPU-enabled stage fails for a method-independent engineering
  reason, retry only after logging the error and preserving the failed run root.

## Blackwell Environment Notes

AutoDL is not identical to the previous RTX 4090 shared server environment.
Paper-facing reruns should record these differences explicitly.

Observed AutoDL baseline on 2026-04-29:

| Item | AutoDL RTX PRO 6000 Blackwell |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 97,887 MiB |
| Driver / `nvidia-smi` CUDA | Driver 580.82.09 / CUDA 13.0 |
| Python env | `/root/autodl-tmp/envs/spectralindexgs_bw` |
| Python executable | `/root/autodl-tmp/envs/spectralindexgs_bw/bin/python` |
| Python / PyTorch | Python 3.10.20 / PyTorch 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| CUDA arch | `sm_120` / compute capability 12.0 |
| CUDA extensions | `diff-gaussian-rasterization` and `simple-knn` rebuilt for `TORCH_CUDA_ARCH_LIST=12.0` |
| `nvcc` | `/usr/local/cuda-12.8/bin/nvcc`, not on PATH by default |
| `colmap` | Not on PATH as of the last check; raw E3/E4 needs COLMAP installed or an explicit executable path before launch |

Blackwell-specific compatibility notes:

- Use the explicit Python executable above; the env currently does not expose a
  usable `bin/activate` workflow.
- If CUDA extensions are rebuilt, export/use `TORCH_CUDA_ARCH_LIST=12.0`.
- PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`. This repo uses
  trusted local 3DGS checkpoints with non-tensor metadata, so trusted checkpoint
  loading must explicitly use `weights_only=False` when supported.
- Official aligned E4b inputs use `*_aligned` directories, while raw rectified
  inputs use `*_rectified`; both path conventions must remain supported.
- Raw-scene experiments cannot start until COLMAP is installed/found and GPU
  SIFT/matching has passed a smoke check.

## Scene Set

### Raw UAV Scenes

| ID | Scene |
|---|---|
| raw_self | `self_m3m` |
| raw001 | `20240528/DJI_202405281154_001_reynoldsTR01crossrtk` |
| raw002 | `20240528/DJI_202405281154_002_reynoldsAB02crossrtk` |
| raw003 | `20240528/DJI_202405281220_003_reynoldsAb01crossrtk` |
| raw004 | `20240528/DJI_202405281326_004_jprAb01crossrtk` |
| raw005 | `20240528/DJI_202405281358_005_jprAr01crossrtk` |
| raw006 | `20240528/DJI_202405281358_006_jprAb02crossrtk` |

### Official Aligned Scenes

| ID | Scene |
|---|---|
| ms_bud | `ms-bud-swelling` |
| ms_fruit | `ms-fruit-trees` |
| ms_garden | `ms-garden` |
| ms_golf | `ms-golf` |
| ms_lake | `ms-lake` |
| ms_tree | `ms-single-tree` |
| ms_solar | `ms-solar` |

## Main Training Matrix

| Group | Method | Scenes | Count | Role |
|---|---|---:|---:|---|
| M1 | E3 independent band transfer | raw 7 + official 7 | 14 | Main method line |
| M2 | E4b-zero joint multispectral representation | raw 7 + official 7 | 14 | Co-main candidate line |
| M3 | MMSplat official aligned | official 7 | 7 | External baseline, fair aligned comparison |
| M4 | MMSplat raw-default retained subset | raw_self only | 1 | Raw unaligned applicability/fair retained-subset comparison |
| M5 | E3 retained subset for MMS raw comparison | raw_self only | 1 | Matched retained-subset reference |
| M6 | E4b-zero retained subset for MMS raw comparison | raw_self only | 1 | Matched retained-subset co-main reference |

Notes:

- Raw E3/E4 must run the full raw pipeline: D/RGB COLMAP, multispectral
  rectification, QA, RGB anchor, band stage, products, render, metrics.
- Official E3/E4 use the aligned official preparation path and do not run
  rectification.
- MMSplat official runs are only compared on official aligned scenes in the
  main external table.
- MMSplat raw-default retained subset is not a symmetric raw 7 benchmark. It is
  an applicability/fairness table for raw unaligned self_m3m, using the subset
  MMSplat successfully retains.

## Ablation Matrix

Representative scenes are fixed to:

- `raw_self`
- `raw002`
- `ms_lake`

| Group | Method | Scenes | Count | Role |
|---|---|---:|---:|---|
| A1 | From scratch per band | representative 3 | 3 | RGB anchor necessity |
| A2 | Geometry unfrozen stage-2 | representative 3 | 3 | Geometry-lock necessity |
| A3 | E4b-rgb_tied initialization | representative 3 | 3 | E4b initialization ablation, supplementary/discussion |

Notes:

- Geometry-unfrozen must reuse the E3 Stage-1 RGB checkpoint and prepared scene.
  Only stage-2 geometry/opacity freezing is changed.
- From-scratch should use the same splits, resolution, and stage budget as the
  corresponding E3 band stage when possible.
- E4b-rgb_tied is not a main method unless later evidence overturns the current
  conclusion. It is a controlled ablation for E4b-zero.

## Evaluation Matrix

### Image And Band Fidelity

Run for:

- E3 all 14 scenes
- E4b-zero all 14 scenes
- MMSplat official 7 scenes
- retained-subset self_m3m comparison methods: MMSplat-default, E3-retained,
  E4b-zero-retained
- all representative ablations

Metrics:

- RGB: PSNR, SSIM, LPIPS, coverage
- Bands G/R/RE/NIR: PSNR, SSIM, LPIPS, coverage
- Use `gt_nonzero` masking for native metrics.
- Use common-mask paired comparison for E3/E4/MMS official aligned comparisons.

### Spectral Index Fidelity

Run for:

- E3 all 14 scenes
- E4b-zero all 14 scenes
- MMSplat official 7 scenes, if exported bands are available in the same
  protocol
- representative ablations

Primary indices:

- NDVI
- GNDVI
- NDRE

Primary metrics:

- RMSE
- SSIM

Supplementary metrics:

- MAE
- PSNR
- per-view statistics

### Depth-Reference Geometry

Primary protocol:

- `relative_depth = (D_model - D_ref) / D_ref`

Primary table metrics:

- AbsRelMean
- Agree@1%
- Agree@5%
- Agree@10%

Supplementary curve thresholds:

- 0.5%
- 1%
- 2.5%
- 5%
- 10%
- 20%

Run target:

| Method | Raw 7 | Official 7 | Notes |
|---|---:|---:|---|
| E3 | yes | yes | Required |
| E4b-zero | yes | yes | Required |
| MMSplat | no | yes, if depth adapter closes | Do not block main table if unsupported |
| From scratch | representative 3 | representative subset only | Ablation table |
| Geometry unfrozen | representative 3 | representative subset only | Ablation table |
| E4b-rgb_tied | representative 3 | representative subset only | Supplementary |

Official aligned depth-reference must be labeled as sparse/point-splat
reference, not dense metric ground truth, unless a denser reference is later
constructed.

### Cost Metrics

Use the same external non-invasive monitor for every method.

Primary cost fields:

- Wall-clock time
- Peak GPU memory
- Mean GPU memory
- Peak GPU utilization
- Output storage size

Recommended scope:

| Method | Scenes | Count |
|---|---:|---:|
| E3 | all 14 | 14 |
| E4b-zero | all 14 | 14 |
| MMSplat | official 7 | 7 |

If time becomes constrained, the fallback cost table uses the representative
three scenes plus MMSplat `ms_lake`, but the preferred AutoDL protocol is the
full scope above.

## Required Output Blocks

For every main run, preserve:

- command line
- git commit
- environment path
- dataset source path
- prepared/rectified path
- model output path
- render output path
- metric output path
- index output path
- depth output path, if applicable
- cost monitor output path
- failure log, if a stage fails

Do not overwrite old run roots. Use timestamped run directories under
`/root/autodl-tmp/runs`.

## Storage Budget

These estimates are intentionally conservative. They should be treated as
planning bounds, not exact quotas. Paper-facing metrics should be preserved even
when bulky intermediate renders or temporary COLMAP products are later archived
or removed.

### Fixed Baseline

| Item | Expected Size | Notes |
|---|---:|---|
| Raw datasets | 40-50 GB | Current transferred dataset is about 41 GB. |
| Code, environment, CUDA extensions, caches | 20-40 GB | Includes the Python env, compiled extensions, LPIPS cache, and small smoke runs. |
| Reserved free space | 60-100 GB | Keep this free for temporary tensors, COLMAP database expansion, exports, and failed-run logs. |

Practical usable experiment space on a 550 GB work disk is therefore roughly
`350-430 GB`, depending on cache growth.

### Per-Scene Run Estimates

| Run Type | Typical Per Scene | Large-Scene Bound | Main Drivers |
|---|---:|---:|---|
| Raw E3 full | 25-55 GB | 70 GB | COLMAP/prepared assets, RGB anchor, four exported band models, renders. |
| Official E3 full | 15-45 GB | 60 GB | Prepared aligned images, RGB anchor, four band models, renders. |
| Raw E4b-zero full | 20-50 GB | 75 GB | Unified checkpoint, exported band views, renders. Keeping both unified and all exported PLYs is expensive. |
| Official E4b-zero full | 15-45 GB | 65 GB | Same as above but no raw rectification/COLMAP front-end. |
| MMSplat official | 8-30 GB | 45 GB | Method output, adapted renders, compare artifacts. |
| MMS/E3/E4 retained self compare | 20-70 GB total | 90 GB | One retained-subset scene, three methods. |
| One representative ablation scene | 15-50 GB | 70 GB | From-scratch and geometry-unfrozen can duplicate full Gaussian models. |
| Depth-reference evaluation per scene/method | 1-8 GB | 15 GB | Depth bundles, masks, summaries; dense reference variants would be much larger. |
| Cost logs | <1 GB | <1 GB | Text/CSV/JSON monitor traces. |

### Phase-Level Estimates

| Phase | Scope | If Fully Kept Online | Recommended Working Peak |
|---|---|---:|---:|
| P0 baseline + smoke | dataset, env, code, smoke | 60-100 GB | 60-100 GB |
| P1 E3 all scenes | 14 E3 full runs | 300-700 GB | 120-220 GB by running/cleaning in batches |
| P2 E4b-zero all scenes | 14 E4b-zero full runs | 250-650 GB | 120-240 GB by archiving exports/renders after metrics |
| P3 MMS official | 7 official MMS runs | 60-210 GB | 60-120 GB |
| P4 retained self compare | MMS/E3/E4 retained | 20-90 GB | 20-90 GB |
| P5 representative ablations | 3 methods x 3 scenes | 150-450 GB | 100-180 GB by scene batches |
| P6 depth-reference | E3/E4 all scenes + ablations + optional MMS | 50-200 GB | 50-120 GB if summaries are kept and bulky depth dumps archived |
| P7 aggregation/tables | summaries, CSV/JSON/plots | 1-10 GB | 1-10 GB |

### Manual Storage Policy For 550 GB

- Do not attempt to keep every full run root from every phase online at once.
- Storage cleanup is a manual decision made between phases. Do not let a batch
  runner automatically delete artifacts.
- Keep these artifacts on AutoDL until the phase has been checked and backed up:
  - command and environment logs
  - final `results.json`, `per_view.json`, index summaries, depth summaries,
    cost summaries
  - final model checkpoint or PLY needed for reproduction/visual inspection
  - failure logs and protocol/audit JSON
- After a phase is manually summarized and backed up, the following bulky
  derived artifacts may be manually archived or removed:
  1. repeated render PNGs that are already represented by metrics
  2. temporary COLMAP databases and dense workspace products not used by later
     stages
  3. duplicate E4b exported per-band PLYs if the authoritative unified
     checkpoint and metrics are preserved
  4. intermediate checkpoints when a final checkpoint is already preserved
- If all final models, all renders, and all depth bundles must remain online
  simultaneously, request at least 1 TB. A comfortable no-cleanup setup is
  closer to 1.5-2 TB.

## Execution Order

1. Create a batch manifest with all scene/method/run-root mappings.
2. Run one official aligned E3/E4/MMS mini-batch to verify path conventions on
   AutoDL.
3. Run the raw scenes with GPU COLMAP enabled for E3 and E4.
4. Run official aligned E3/E4/MMS.
5. Run retained-subset self_m3m MMS/E3/E4 comparison.
6. Run representative ablations.
7. Run depth-reference evaluation.
8. Run cost aggregation and quality/index/depth summary aggregation.

Each phase should be manually launched, inspected, summarized, and cleaned by
CODEX when instructed by the user. Do not rely on app automation or unattended
cleanup logic for these experiment batches.

## Explicit Non-Goals For This Batch

- Do not run E5 unless it is separately implemented, smoke-tested, and approved.
- Do not promote E4b-rgb_tied to a main method without new evidence.
- Do not mix old 4090 paper-facing values with new AutoDL paper-facing values.
- Do not write derived files into original dataset directories.
