# SpectralIndexGS

This repository contains the current SpectralIndexGS codebase for UAV multispectral 3D Gaussian Splatting.

The current mainline supports:

- RGB geometry anchoring with 3D Gaussian Splatting
- band-specific transfer for `G / R / RE / NIR`
- learning-based rectification for raw UAV multispectral captures
- strict train/test protocols for raw scenes
- false-color and spectral-index proxy product export
- masked and common-mask offline evaluation

The repository includes the current multispectral training pipeline, raw-scene rectification workflow, protocol-freezing utilities, product export, and offline evaluation scripts.

## 1. What Is Included

The main entry points are:

- `run_spectralindexgs_pipeline.py`
  - end-to-end raw multispectral pipeline
- `prepare_scene_colmap.py`
  - raw-scene COLMAP preparation with train-only SfM and held-out test localization support
- `prepare_official_ms_scene.py`
  - converts official aligned multispectral scenes into repo-native scene layout
- `train.py`
  - 3DGS training entry point
- `render.py`
  - offline rendering
- `metrics.py`
  - standard 3DGS metrics
- `masked_metrics.py`
  - offline masked evaluation for a single model
- `common_mask_eval.py`
  - common-mask evaluation across multiple compared methods
- `build_spectral_products.py`
  - false-color and spectral-index product construction
- `freeze_protocol_assets.py`
  - protocol split freezing and command-template generation

The repository also contains required runtime modules:

- `arguments/`
- `gaussian_renderer/`
- `scene/`
- `utils/`
- `lpipsPyTorch/`
- `submodules/`

## 2. Current Protocol Summary

The repository currently supports two scene families:

### 2.1 Raw UAV multispectral scenes

These scenes are expected to come from raw UAV captures with:

- an RGB image per capture
- four single-band images: `G`, `R`, `RE`, `NIR`

For raw scenes, the intended protocol is:

- frozen train/test split
- train-only SfM / COLMAP mapping
- held-out test image localization into the frozen train reconstruction
- no post-registration bundle adjustment
- no point-cloud growth driven by test images

### 2.2 Official aligned multispectral scenes

These scenes already provide aligned image-space geometry / transforms and therefore do **not** go through the raw rectification path.

For official aligned scenes, the intended protocol is:

- dataset-provided aligned transforms
- no raw-scene rectification
- no raw-scene SfM protocol repair

## 3. System Requirements

The CUDA extensions in `submodules/` require:

- a system CUDA toolkit with `nvcc`
- a system C/C++ compiler

The Conda environment installs the PyTorch CUDA runtime, but that runtime alone does not replace the system build toolchain required to compile the CUDA extensions.

### 3.1 Windows prerequisites

- NVIDIA GPU and driver
- CUDA toolkit with `nvcc` on `PATH`
- Visual Studio Build Tools with the MSVC x64 C/C++ toolchain
- COLMAP on `PATH` (or provide the path explicitly)
- ExifTool on `PATH` (or provide the path explicitly)

Preflight checks:

```powershell
where.exe nvcc
where.exe cl
where.exe colmap
where.exe exiftool
```

### 3.2 Linux prerequisites

- NVIDIA GPU and driver
- CUDA toolkit with `nvcc` on `PATH`
- GCC/G++ toolchain
- COLMAP on `PATH`
- ExifTool on `PATH`

For raw-scene GPU SIFT on headless Linux, a distro `colmap` package is often not enough, because it may be built without CUDA support. A user-local CUDA-enabled build is the recommended setup for this repository.

Preflight checks:

```bash
which nvcc
which g++
which colmap
which exiftool
```

Recommended user-local CUDA COLMAP install path:

```bash
$HOME/opt/colmap-cuda/bin/colmap
```

Example headless Linux build flow:

```bash
sudo apt-get install -y \
  build-essential cmake ninja-build git pkg-config \
  libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev \
  libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libflann-dev

export PATH=/usr/local/cuda-11.8/bin:$PATH

cmake -S <COLMAP_SRC> -B <COLMAP_BUILD> -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/colmap-cuda-3.7" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib:/usr/local/cuda-11.8/lib64" \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=ON \
  -DOPENGL_ENABLED=ON \
  -DBOOST_STATIC=OFF \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
  -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-11.8/bin/nvcc \
  -DCUDA_NVCC_FLAGS=--std=c++14 \
  -DCUDA_ARCHS=8.9

cmake --build <COLMAP_BUILD> -j8
cmake --install <COLMAP_BUILD>
ln -sfn "$HOME/opt/colmap-cuda-3.7" "$HOME/opt/colmap-cuda"
```

## 4. Environment Setup

### 4.1 Create a Conda environment

Recommended step-by-step setup:

```bash
conda create -n spectralindexgs python=3.10.18 pip=25.2 numpy=1.26.4 -y
conda activate spectralindexgs
conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia -c defaults
python -m pip install -r requirements.txt
```

The repository also ships an `environment.yml` file if you prefer a one-shot Conda flow:

```bash
conda env create -f environment.yml
conda activate spectralindexgs
```

### 4.2 Install CUDA extensions

From the repository root, install the CUDA extensions explicitly:

```bash
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/fused-ssim
```

### 4.3 Optional external rectification dependency

The raw multispectral rectification path uses an external MINIMA checkout at runtime.

This repository does **not** automatically download MINIMA. If you want to run raw-scene rectification, prepare a local MINIMA checkout and pass its path through:

- `--minima_root`

Supported matcher options currently exposed by the bridge:

- `--minima_method roma`
- `--minima_method xoftr`

The default and more stable backend in the current experiments is `roma`.

### 4.4 Sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2, diff_gaussian_rasterization, simple_knn, fused_ssim; print('extensions OK')"
python train.py -h
python render.py -h
python prepare_scene_colmap.py -h
python run_spectralindexgs_pipeline.py -h
python prepare_official_ms_scene.py -h
python build_spectral_products.py -h
python masked_metrics.py -h
python common_mask_eval.py -h
```

### 4.5 External tools used by the raw-scene pipeline

For raw UAV scenes, the end-to-end pipeline expects the following tools to be available:

- `colmap`
- `exiftool`

They can either be visible on `PATH`, or be provided explicitly through:

- `--colmap_executable`
- `--exiftool_executable`

Current COLMAP resolution order in the scripts is:

1. `SIGS_COLMAP_EXECUTABLE`
2. `COLMAP_EXECUTABLE`
3. `~/opt/colmap-cuda/bin/colmap`
4. `colmap`

For `prepare_scene_colmap.py`, the default `--sift_use_gpu auto` / `--sift_matching_use_gpu auto` behavior is:

- try GPU first
- if COLMAP GPU SIFT fails in the current runtime, retry on CPU automatically

GPU SIFT memory pressure can be tuned from the pipeline through:

- `--sift_max_image_size`
- `--sift_max_num_features`
- `--sift_matching_max_num_matches`
- `--sift_num_threads`

The defaults mirror COLMAP defaults: `3200`, `8192`, `32768`, and `-1`. If the CUDA COLMAP build starts but reports SiftGPU memory errors, lower these values for the raw SfM step.

## 5. Data Assumptions

### 5.1 Raw UAV multispectral scenes

Raw scenes are prepared by `prepare_m3m_multispectral.py`, and their RGB geometry is prepared through `prepare_scene_colmap.py` inside the main pipeline.

The expected capture structure is:

- one RGB image per capture, for example `DJI_xxx_D.JPG`
- four single-band images per capture:
  - `MS_G.TIF`
  - `MS_R.TIF`
  - `MS_RE.TIF`
  - `MS_NIR.TIF`

The current code is designed to tolerate slight naming drift between the RGB and multispectral files in DJI-style data. Pairing is done by capture/frame identity rather than by requiring an exact full timestamp-stem match.

### 5.2 Official aligned multispectral scenes

Official aligned scenes are prepared by `prepare_official_ms_scene.py`.

Supported image roots:

- `images`
- `images_2`
- `images_4`

In the current protocol, `images_2` is the typical choice for the official aligned scenes.

## 6. Frozen Split Assets

If you want to reproduce the strict held-out evaluation protocol, first freeze the protocol assets:

```bash
python freeze_protocol_assets.py --out_root <SPLIT_ROOT>
```

This generates:

- `split_v1.json`
- per-scene `train.txt`
- per-scene `test.txt`
- protocol-pack metadata
- official-MS command templates

## 7. Main Raw-Scene Pipeline

The primary entry point for raw multispectral scenes is:

```bash
python run_spectralindexgs_pipeline.py \
  --raw_root <RAW_SCENE_ROOT> \
  --prepared_root <PREPARED_ROOT> \
  --rectified_root <RECTIFIED_ROOT> \
  --out_root <OUT_ROOT> \
  --protocol_split <SPLIT_JSON> \
  --raw_sfm_protocol train_only_register_test \
  --rectification_backend minima \
  --minima_method roma \
  --minima_root <MINIMA_ROOT> \
  --rgb_iter 30000 \
  --band_iter 60000 \
  --rgb_res 8 \
  --band_res 8 \
  --sift_max_image_size 3200 \
  --sift_max_num_features 8192 \
  --sift_matching_max_num_matches 32768 \
  --input_dynamic_range uint16 \
  --radiometric_mode exposure_normalized \
  --auto_render
```

### 7.1 What the pipeline does

For raw UAV scenes, the pipeline performs:

1. raw scene preparation
2. train-only RGB SfM / COLMAP mapping
3. held-out RGB test localization into the frozen train reconstruction
4. RGB geometry training
5. MINIMA-assisted band rectification
6. per-band stage-2 transfer
7. spectral product export
8. optional offline rendering

### 7.2 Pipeline stages

The current pipeline stages are:

1. `01_prepare`
2. `02_train_rgb`
3. `03_build_rectified_bands`
4. `04_train_band_g`
5. `05_train_band_r`
6. `06_train_band_re`
7. `07_train_band_nir`
8. `08_build_products`
9. `09_optional_render`

### 7.3 Important defaults and semantics

- `band_iter` currently follows a **final-iteration** semantics
- for example, `rgb_iter=30000` and `band_iter=60000` means:
  - RGB is trained to `30000`
  - each band stage restores the RGB checkpoint and continues to iteration `60000`
- `rgb_res` and `band_res` are passed to `train.py`
- raw single-band TIFF data commonly uses:
  - `--input_dynamic_range uint16`
  - `--radiometric_mode exposure_normalized`

### 7.4 Typical output structure

```text
<OUT_ROOT>/
  Model_RGB/
  Model_G/
  Model_R/
  Model_RE/
  Model_NIR/
  Products/
    false_color_nir_r_g/
    false_color_re_nir_r/
    false_color_nir_re_g/
    ndvi_gray/
    ndvi_pseudocolor/
    ndre_gray/
    ndre_pseudocolor/
    gndvi_gray/
    gndvi_pseudocolor/
    savi_gray/
    savi_pseudocolor/
    osavi_gray/
    osavi_pseudocolor/
```

## 8. Official Aligned Scene Workflow

Official aligned scenes do not use the raw rectification pipeline.

### 8.1 Prepare the aligned scene

```bash
python prepare_official_ms_scene.py \
  --source_root <OFFICIAL_SCENE_ROOT> \
  --out_root <PREPARED_ROOT> \
  --image_root images_2 \
  --split_json <SPLIT_JSON>
```

If you only want to prepare and inspect a raw scene COLMAP layout outside the full pipeline, you can also run:

```bash
python prepare_scene_colmap.py \
  -s <PREPARED_RGB_ROOT> \
  --colmap_executable colmap \
  --exiftool_executable exiftool \
  --image_list_path <TRAIN_PLUS_TEST_LIST> \
  --mapper_image_list_path <TRAIN_ONLY_LIST> \
  --register_images_after_mapper \
  --registration_required_image_list_path <TEST_ONLY_LIST> \
  --registration_audit_path <AUDIT_JSON> \
  --fail_on_missing_registration \
  --strict_no_point_growth_after_registration \
  --resize
```

### 8.2 Train the RGB model

```bash
python train.py \
  -s <PREPARED_ROOT>/RGB \
  -m <OUT_ROOT>/Model_RGB \
  -r 1 \
  --eval \
  --iterations 30000 \
  --checkpoint_iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 30000 \
  --disable_viewer \
  --modality_kind rgb
```

### 8.3 Train the four band models

Example for `G`:

```bash
python train.py \
  -s <PREPARED_ROOT>/G_aligned \
  -m <OUT_ROOT>/Model_G \
  -r 1 \
  --eval \
  --iterations 60000 \
  --checkpoint_iterations 60000 \
  --save_iterations 60000 \
  --test_iterations 60000 \
  --start_checkpoint <OUT_ROOT>/Model_RGB/chkpnt30000.pth \
  --modality_kind band \
  --target_band G \
  --single_band_mode true \
  --single_band_replicate_to_rgb true \
  --input_dynamic_range uint8 \
  --radiometric_mode raw_dn \
  --stage2_mode band_transfer \
  --reset_appearance_features true \
  --freeze_geometry true \
  --freeze_opacity true \
  --tied_scalar_carrier true \
  --feature_lr 0.001 \
  --lambda_dssim 0 \
  --require_rectified_band_scene false \
  --use_validity_mask false \
  --disable_viewer
```

Repeat analogously for `R`, `RE`, and `NIR`.

### 8.4 Build spectral products

```bash
python build_spectral_products.py \
  --g_model_dir <OUT_ROOT>/Model_G \
  --r_model_dir <OUT_ROOT>/Model_R \
  --re_model_dir <OUT_ROOT>/Model_RE \
  --nir_model_dir <OUT_ROOT>/Model_NIR \
  --g_iter 60000 \
  --r_iter 60000 \
  --re_iter 60000 \
  --nir_iter 60000 \
  --out_root <OUT_ROOT>/Products \
  --require_opacity_match true
```

## 9. Evaluation

### 9.1 Standard metrics

```bash
python metrics.py -m <MODEL_DIR_1> <MODEL_DIR_2> ...
```

### 9.2 Masked offline metrics

```bash
python masked_metrics.py -m <MODEL_DIR> --split test --out_json <OUT_JSON>
```

By default, the script parses `source_path` from `cfg_args` and looks for validity masks under:

```text
<source_path>/validity_masks
```

### 9.3 Common-mask comparison across methods

```bash
python common_mask_eval.py \
  --method E0=<RUN_ROOT_0> \
  --method E1=<RUN_ROOT_1> \
  --method E2=<RUN_ROOT_2> \
  --out_json <OUT_JSON>
```

### 9.4 Paired confidence-interval reports

```bash
python paired_ci_report.py -h
```

### 9.5 Masked panel export

```bash
python export_masked_panels.py -h
```

## 10. Method Boundary

The current implementation is intended to support:

- shared-geometry band carriers
- tied scalar carriers
- per-Gaussian spectral index proxy export
- rectification-assisted raw multispectral supervision

It is **not** intended to claim:

- native N-channel rendering
- exact SH-level nonlinear index closure
- exact physical spectral radiance reconstruction

## 11. Frequently Asked Questions

### Why is `band_iter` set to `60000`?

Because the current code uses a final-iteration semantics for stage-2 training.

If:

- `rgb_iter = 30000`
- `band_iter = 60000`

then the intended behavior is:

- RGB trains to `30000`
- each band model restores the RGB checkpoint and trains until iteration `60000`

### Do raw scenes need `--protocol_split`?

If you want the intended strict held-out protocol, the answer is yes.

The raw-scene main protocol is built around:

- frozen splits
- train-only SfM
- held-out test localization

### Why do official aligned scenes not use rectification?

Because they already belong to a dataset-provided aligned protocol.

The current preparation script additionally normalizes the split by enforcing the RGB/G/R/RE/NIR group-id intersection so that the train/test views are truly matched across all five modalities.

### Does the pipeline modify the raw data root?

It should not.

Prepared data, rectified data, model outputs, audit files, and protocol artifacts are expected to be written to separate directories such as:

- `prepared_root`
- `rectified_root`
- `out_root`
- frozen protocol-pack directories

## 12. Repository Layout

```text
arguments/                     CLI arguments
gaussian_renderer/             renderer
scene/                         scene, dataset, and camera loaders
utils/                         helper utilities, including the MINIMA bridge
submodules/                    CUDA extensions

prepare_m3m_multispectral.py   raw M3M scene preparation
prepare_scene_colmap.py        raw-scene COLMAP preparation
prepare_official_ms_scene.py   official aligned MS preparation
freeze_protocol_assets.py      frozen split / protocol-pack generation
estimate_band_homographies.py  rectification estimation
build_rectified_band_dataset.py
qa_rectification.py
run_spectralindexgs_pipeline.py
build_spectral_products.py
train.py
render.py
metrics.py
masked_metrics.py
common_mask_eval.py
paired_ci_report.py
export_masked_panels.py
```

## 13. Notes

- This README reflects the current multispectral mainline.
- If the README and the code ever diverge, use the current CLI help and code behavior as the authoritative reference:

```bash
python prepare_scene_colmap.py -h
python run_spectralindexgs_pipeline.py -h
python train.py -h
python render.py -h
```
