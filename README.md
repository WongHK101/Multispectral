# UMGS: Geometry-Consistent UAV Multispectral Gaussian Reconstruction

This repository contains the author-controlled implementation of UMGS. UMGS
uses an RGB reconstruction as a stable Gaussian support, conditions raw
cross-sensor observations into the RGB image domain, propagates observation
validity, and learns one appearance-only Gaussian branch for each of
`G / R / RE / NIR`. The shared Gaussian identity and frozen spatial support
allow the branches to be combined into 3D NDVI, GNDVI, and NDRE products.

The repository intentionally excludes raw imagery, trained checkpoints,
experiment outputs, manuscript sources, and downloaded third-party projects.
`AuthorKit27/`, datasets, models, and `_local_workbench/` are ignored. See
`THIRD_PARTY_NOTICES.md` for upstream source and license information.

## 1. Protocols: reported paper mode versus prospective strict mode

The code supports two RGB SfM modes. They answer different questions and must
not be interchanged when reproducing a reported table.

### Reported paper mode

The UMGS results reported in the paper use a registered common RGB SfM model:
all RGB views are reconstructed first, then the frozen registered-camera split
is installed for held-out rendering and evaluation. For a raw rebuild, select:

```text
--raw_sfm_protocol all_images
```

When the released frozen SfM and split assets are available, use them directly
instead of rerunning COLMAP. This is the most exact reproduction route because
it binds the same camera identities and sparse model used by the reported run.

### Prospective strict mode

The current CLI default is a stricter prospective protocol:

```text
--raw_sfm_protocol train_only_register_test
```

It maps training images only and then localizes held-out images without
post-registration bundle adjustment or point-cloud growth. This mode is useful
for new studies but does not retroactively replace the reported paper protocol.

## 2. Environment

The final campaign used Ubuntu 22.04.5, Python 3.10.20, PyTorch 2.8.0+cu128,
CUDA 12.8, COLMAP 3.7, OpenMVS 2.3.0, and an NVIDIA RTX PRO 6000 Blackwell
Server Edition. `environment_umgs_cuda128.yml` is the canonical Python
environment for those runs. The older `environment.yml` is retained only as a
CUDA 11.8 compatibility reference.

System prerequisites:

- an NVIDIA driver and CUDA toolkit with `nvcc`;
- GCC/G++ on Linux or the MSVC x64 build tools on Windows;
- COLMAP 3.7 for raw RGB SfM;
- ExifTool for DJI metadata extraction;
- OpenMVS 2.3.0 only for the optional geometry-proxy evaluation.

Create the environment and build the bundled CUDA extensions:

```bash
conda env create -f environment_umgs_cuda128.yml
conda activate umgs

python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/fused-ssim
```

`--no-build-isolation` is required so each extension builds against the active
PyTorch/CUDA environment. Useful preflight checks are:

```bash
nvcc --version
colmap -h
exiftool -ver
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization, simple_knn, fused_ssim; print('extensions OK')"
```

The bundled LPIPS wrapper downloads public LPIPS weights on first use. Run the
first metric evaluation with network access or pre-populate PyTorch's checkpoint
cache. Its source and license are documented in `lpipsPyTorch/LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## 3. Install the cross-sensor matcher

Raw Mavic 3M conditioning uses MINIMA's RoMa-large backend. MINIMA is not
vendored. The reported campaign used:

```text
repository  https://github.com/LSXI7/MINIMA
revision    796e7721174f9f829b79b3702bf8c2ae9a3d447a
checkpoint  weights/minima_roma.pth
SHA-256     17f3923bd780e8f1450792706e0643f70c8864ad11bf5fd4f2b54714bac23538
```

Follow MINIMA's upstream environment and submodule instructions, download its
released `minima_roma.pth`, verify the hash, and pass both locations explicitly:

```bash
git clone https://github.com/LSXI7/MINIMA.git /path/to/MINIMA
git -C /path/to/MINIMA checkout 796e7721174f9f829b79b3702bf8c2ae9a3d447a
git -C /path/to/MINIMA submodule update --init --recursive
sha256sum /path/to/MINIMA/weights/minima_roma.pth
```

The UMGS options are `--minima_root /path/to/MINIMA` and
`--minima_ckpt /path/to/MINIMA/weights/minima_roma.pth`. No machine-specific
matcher path is embedded in the code.

## 4. Raw DJI Mavic 3 Multispectral input

The raw organizer recursively scans one scene root. A complete capture group
uses the DJI Mavic 3M filename forms below, where the four-digit frame id is
unique within the scene:

```text
<stem>_<frame>_D.JPG
<stem>_<frame>_MS_G.TIF
<stem>_<frame>_MS_R.TIF
<stem>_<frame>_MS_RE.TIF
<stem>_<frame>_MS_NIR.TIF
```

For example:

```text
DJI_20260526160312_0001_D.JPG
DJI_20260526160312_0001_MS_G.TIF
DJI_20260526160312_0001_MS_R.TIF
DJI_20260526160312_0001_MS_RE.TIF
DJI_20260526160312_0001_MS_NIR.TIF
```

Only complete five-camera groups are prepared. The organizer records missing
modalities in `prepared_manifest.json`.

> **Destructive-output warning:** `prepare_m3m_dataset` removes an existing
> `prepared_root` before recreating it. Never set `prepared_root` to the raw
> dataset, repository root, or a directory containing irreplaceable files.
> `raw_root` must be disjoint from every derived root. Separate prepared,
> rectified, and model roots are strongly recommended for provenance.

The default `hardlink` mode requires raw and prepared roots on the same
filesystem. Use `--link_mode copy` across filesystems or when an independent
prepared copy is desired.

## 5. Frozen split format

`--protocol_split` accepts a UTF-8 JSON object with non-overlapping `train` and
`test` lists (`eval` is accepted as an alias for `test`). Entries may be image
name strings or objects containing `image_name`, `rgb_image`, `path`, or
`file_path`:

```json
{
  "schema": "umgs_split_v1",
  "train": [
    "DJI_20260526160312_0001_D.JPG",
    {"image_name": "DJI_20260526160324_0002_D.JPG"}
  ],
  "test": [
    "DJI_20260526160441_0009_D.JPG"
  ]
}
```

The full dataset release will provide immutable scene-specific split files and
their camera/SfM identities. The small review data examples are previews only
and are not sufficient to reproduce the reported experiments.

## 6. Reproduce a reported UMGS run

The command below rebuilds the reported *protocol family* from raw data. Exact
numerical reproduction should instead add `--sparse_source` pointing to the
released frozen `sparse/0` asset for the scene.

```bash
python run_spectralindexgs_pipeline.py \
  --raw_root /data/raw/scene \
  --prepared_root /work/scene/prepared \
  --rectified_root /work/scene/rectified \
  --out_root /work/scene/models \
  --protocol_split /release/splits/scene_split_v1.json \
  --raw_sfm_protocol all_images \
  --minima_root /path/to/MINIMA \
  --minima_ckpt /path/to/MINIMA/weights/minima_roma.pth \
  --rgb_iter 30000 \
  --band_iter 60000 \
  --rgb_res 8 \
  --band_res 8 \
  --input_dynamic_range uint16 \
  --radiometric_mode exposure_normalized \
  --freeze_opacity true \
  --use_validity_mask true
```

To use the stricter prospective SfM mode on a new experiment, change only the
protocol selector after defining the split:

```bash
--raw_sfm_protocol train_only_register_test
```

The stages are:

1. organize raw Mavic 3M capture groups;
2. prepare or install RGB SfM and the frozen split;
3. train the 30K RGB anchor;
4. estimate quality-gated cross-sensor transforms, propagate valid regions,
   and run rectification QA;
5. restore that anchor and train G, R, RE, and NIR appearance branches;
6. build spectral products and optionally render held-out views.

`band_iter=60000` is a final-global-iteration value. Each band restores the
30K RGB checkpoint and performs 30K new appearance-only updates. One requested
band therefore costs 30K RGB updates plus 30K band updates. The complete suite
trains the RGB anchor once and four branches: `30K + 4 x 30K = 150K` optimizer
updates.

Use `--from_step` and `--to_step` to run a contiguous subset. The principal
outputs are:

```text
<prepared_root>/
  RGB/
  G_raw/ R_raw/ RE_raw/ NIR_raw/
  prepared_manifest.json

<rectified_root>/
  rectification_homographies.json
  rectification_qa/
  G_rectified/ R_rectified/ RE_rectified/ NIR_rectified/

<out_root>/
  Model_RGB/
  Model_G/ Model_R/ Model_RE/ Model_NIR/
  Products/
```

Each spectral branch enforces frozen position, scale, rotation, opacity,
Gaussian count, and Gaussian order. Only its scalar appearance carrier is
optimized, and validity masks gate spectral supervision.

## 7. Evaluation

Standard and validity-masked image metrics:

```bash
python metrics.py -m /work/scene/models/Model_RGB
python masked_metrics.py --help
python common_mask_eval.py --help
```

The first `metrics.py` invocation may download LPIPS weights as noted above.

Spectral products:

```bash
python evaluate_spectral_indices.py \
  --g_model_dir /work/scene/models/Model_G \
  --r_model_dir /work/scene/models/Model_R \
  --re_model_dir /work/scene/models/Model_RE \
  --nir_model_dir /work/scene/models/Model_NIR \
  --iteration 60000 \
  --indices NDVI,GNDVI,NDRE \
  --out_json /work/scene/index_metrics.json \
  --mask_mode gt_nonzero_intersection
```

The optional camera-z geometry diagnostic is validation gated. It binds the
checkpoint, split, camera fingerprint, sparse model, rasterizer source,
OpenMVS mesh, and all intermediate files by SHA-256 before computing metrics:

```bash
python tools/depth_reference_geometry_v2/export_umgs_expected_camera_z_packet.py --help
python tools/depth_reference_geometry_v2/render_openmvs_canonical_camera.py --help
python tools/depth_reference_geometry_v2/validate_openmvs_canonical_camera_render.py --help
python tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py --help
```

The final evaluator compares true and deterministic-shuffle branches on the
same valid pixels and the same reference-defined high-gradient domain. Its
OpenMVS camera-z output is a geometric proxy, not surveyed metric ground truth.

## 8. Tests

CPU-side protocol, camera, exporter, metric, and static checks:

```bash
python -m pytest \
  test_prepare_m3m_multispectral.py \
  tools/depth_reference_geometry_v2/test_expected_camera_z_exporter_static.py \
  tools/depth_reference_geometry_v2/test_expected_camera_z_single_packet_validator.py \
  tools/depth_reference_geometry_v2/test_openmvs_campaign_core.py \
  tools/depth_reference_geometry_v2/test_openmvs_canonical_camera_adapter.py \
  tools/depth_reference_geometry_v2/test_proxy_alignment_metrics.py \
  tools/depth_reference_geometry_v2/test_umgs_openmvs_camera_z_proxy_alignment.py -q
python -m py_compile \
  run_spectralindexgs_pipeline.py \
  prepare_m3m_multispectral.py \
  estimate_band_homographies.py \
  build_rectified_band_dataset.py
```

Before a full GPU run, also inspect the command-line contracts:

```bash
python run_spectralindexgs_pipeline.py --help
python train.py --help
python render.py --help
```

## 9. Method boundary

The implementation supports RGB-anchored shared Gaussian support, tied scalar
band carriers, quality-gated cross-sensor supervision, and 3D spectral-index
products. It does not claim native arbitrary-N-channel rasterization, exact
SH-level nonlinear index closure, or calibrated physical spectral radiance.

Additional utilities for previously aligned multispectral datasets and
diagnostic studies remain in the repository, but they are not the reported
UAV-MultiSpec3D benchmark protocol. Use the frozen configuration and result
inventory in the review code package for paper-table traceability.
