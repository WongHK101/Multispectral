# UMGS measured-geometry CPU adaptation

This directory prepares the Road/3K + 5K measured-geometry revision. It does
not train, render, score real depth, fit Sim(3), access a server, or authorize a
GPU run. Existing UMGS proxy outputs and the GS-GCP release remain unchanged.

## Implemented boundaries

- `preflight.py` verifies an externally pinned GS-GCP source snapshot and
  release root, then invokes the unchanged v1.3 release interface. It preserves
  every canonical row, distinguishes formal rows, recomputes raw projections,
  checks camera/pose record hashes, and checks actual local raw JPEG hashes,
  RGB dimensions and EXIF metadata. The camera inputs are authenticated embedded
  release records, **not** live renderer camera evidence. Full RGB matrix decode
  and the renderer-camera binding are explicitly pending.
- `camera_bridge.py` implements explicit projection/normalization algebra.
  Graphdeco `((ndc+1)*size-1)/2` and mesh-proxy `(ndc+1)*size/2` are distinct
  conventions, not string aliases. Their half-pixel relation is tested; existing
  proxy artifacts are not resampled or declared incorrect. Pose identity must
  be independently established before using intrinsic ray mapping.
- Model normalization is inverted from recorded scale/rotation/translation,
  never fitted to checkpoints or LiDAR. Returning raw moments to source units
  requires A, M1, M2 and H. Missing H cannot be reconstructed from M1.
- `contracts.py` inspects authenticated NPZ/NPY headers only. A schema/dtype/
  shape pass is not a packet/ref numeric pass or formal qualification. Legacy
  six-field UMGS v1 packets cannot be silently promoted to metric packet v2.
- `method_recipe.py` distinguishes resolved joint-gradient, non-RGB isolated,
  and neural-color settings without importing Nerfstudio. This is a mechanism
  check, not certification that a recipe reproduces a paper. RGB gradients and
  densification can still change support in an isolation configuration.
- `training_recipes.py` constructs fixed JO, SIG-mechanism and neural-color
  argument vectors without executing them. The reviewed neural decoder uses
  8 features and a raw 3D direction: 11->32->32->7 (1,671 parameters). The
  SIG comparison changes the non-RGB direct photometric structure gradients;
  MS densification, scale regularization and optimizer state remain active.
  Resolved upstream CLI settings must match before execution. These recipes
  are disclosed shared-input comparisons, not recovered author-exact commands.
- `native_moments.py` wires fresh same-call rasterizer outputs into the
  unchanged, externally authenticated packet-v2 reference. Graphdeco needs
  its live raw inverse-depth accumulator H as well as the six-plane output;
  gsplat uses a zero-background feature render [1,z,z*z,1/z]. Neither path
  manufactures missing H from an archived packet. Array tests are synthetic,
  not CUDA or real-packet qualification.
- Camera corner-origin coordinates and integer array indices are converted
  explicitly by `camera_to_array_intrinsics`. A half-pixel offset is applied
  only to the identified target sampling convention, never silently to the
  frozen raw annotation decimal strings. A source-specific sidecar must bind
  the actual historical image/calibration chain.
- `campaign.py` prepares the approved six-scene ledger: ten new pipelines,
  eighteen paired UMGS/JO/neural-MS rows, two core-scene SIG rows, and two
  RGB-anchor rows. It validates hash-bound result receipts without computing
  metrics. Its start/closeout functions return decisions only, never launch
  training or execute power commands. See [CAMPAIGN.md](CAMPAIGN.md).
- The CLI hides CUDA and blocks GPU imports, network connections and external
  child execution. It has **no** `--execute` or automatic resume option. Explicit
  user notification and fresh resource/protocol checks are required later.

The local profile is deliberately not committed: it contains machine-specific
paths, reference-manifest SHA, externally pinned release-root SHA and raw roots.
The reference bundle is built outside the source repository by the reference
project's own builder; reference source is not copied into the training repo.
`common_full_sfm`, split, sampling, aggregation and geographic transforms are
not redefined here. A future adapter must bind its own complete inputs before
calling the reference evaluator.

## Local CPU commands

Run from the Multispectral repository with Python and NumPy/Pillow available:

`requirements-cpu.txt` records the separate Python 3.12 CPU environment used
for checks and optional frozen-reference synthetic smoke. It may be installed
directly on the execution server in a new isolated environment with no inherited
training packages. Do not install it into an existing training environment.
It contains no torch/CUDA dependencies. Download/build receipts and wheel
hashes belong in machine-local evidence, not in this repository.

```text
python -B -m unittest tools.measured_geometry_v1.test_cpu_adapters tools.measured_geometry_v1.test_campaign tools.measured_geometry_v1.test_training_recipes tools.measured_geometry_v1.test_native_moments -v
python -B -m tools.measured_geometry_v1.preflight --profile LOCAL_PROFILE.json --output NEW_REPORT.json
python -B -m tools.measured_geometry_v1.campaign prepare --campaign_id umgs_tgrs_example --output_dir NEW_PREPARATION_DIRECTORY
```

The preflight refuses an existing output or an output inside a release/raw/
reference root. A successful return means `PASS_CPU_INPUT_ADAPTATION_ONLY`;
the report always says `formal_ready=false` and records pending GPU evidence.

Profile fields:

```json
{
  "reference_root": "/path/to/GS-GCP-evaluation-reference",
  "reference_manifest_sha256": "<64 lowercase hex characters>",
  "release_root": "/path/to/gcp_manual_annotations_v1_3_0",
  "release_root_record_sha256": "<64 lowercase hex characters>",
  "release_payload_root_sha256": "<64 lowercase hex characters>",
  "raw_roots": {
    "gcp_3000_20260602": "/path/to/3k/raw",
    "gcp_5000_20260602": "/path/to/5k/raw"
  }
}
```

This is a profile shape example, not a runnable frozen configuration. No private
credentials, datasets, checkpoints, downloaded methods or outputs belong here.

## Remaining qualification

1. Explicit user GPU availability/authorization; do not interrupt another job.
2. Actual old checkpoint and source/SfM/split/resolution identity, not just paths
   in a historical manifest. Old UMGS/JO outputs from the five self-collected
   scenes are reuse candidates after audit; neural-color MS-Splatting has a new
   pipeline for all six scenes. SIG remains a two-core-scene mechanism check.
   Withdrawn external scenes are excluded.
3. Actual CLI-resolution and environment checks for the reviewed SIG/neural-MS
   recipe, together with the pinned source/license identity.
4. Renderer matrix/ray/native pixel parity and metric packet v2 accumulators,
   including numeric packet/ref consistency on real exports.
5. GCP/LiDAR protocol binding and independent metric recomputation. No new
   scientific gate is inferred from a CPU-only synthetic test.

Installing dependencies does not qualify a renderer, its kernels, a method's
paper recipe or the metric-depth adapter. Those gates remain explicit. The
GS-GCP quarter-resolution experiment contract is a separate project and is not
implicitly applied to these multispectral pipelines.
