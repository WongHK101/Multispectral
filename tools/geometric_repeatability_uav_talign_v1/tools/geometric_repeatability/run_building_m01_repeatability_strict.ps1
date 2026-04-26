param(
    [string]$SceneName = "Building",
    [string]$RepoRoot = "D:\dataset\FGS\FGS-0202v1",
    [string]$PythonExe = "D:\anaconda\envs\fgs\python.exe",
    [string]$DataRoot = "F:\databackup\xr6\input\Building",
    [string]$OutRoot = "F:\databackup\xr6\output\GeometricRepeatability\Building\M01_Strict_v1",
    [string]$SplitDir = "D:\dataset\FGS\FGS-0202v1\tools\geometric_repeatability\artifacts\building_protocol_split"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    Write-Host ""
    Write-Host "===== $Name ====="
    & $Body
}

function Assert-Exists {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Write-Status {
    param(
        [Parameter(Mandatory = $true)][string]$Stage
    )
    $status.stage = $Stage
    $status.updated_at = (Get-Date).ToString("s")
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusJson -Encoding UTF8
}

$ProbeList = Join-Path $SplitDir "probe_test.txt"
$OddTrainList = Join-Path $SplitDir "train_odd.txt"
$EvenTrainList = Join-Path $SplitDir "train_even.txt"

$StrictDatasetRoot = Join-Path $OutRoot "strict_dataset"
$ProtocolManifest = Join-Path $StrictDatasetRoot "strict_protocol_manifest.json"

$OddRoot = Join-Path $OutRoot "odd"
$EvenRoot = Join-Path $OutRoot "even"

$OddRgb = Join-Path $OddRoot "Model_RGB"
$OddT = Join-Path $OddRoot "Model_T"
$EvenRgb = Join-Path $EvenRoot "Model_RGB"
$EvenT = Join-Path $EvenRoot "Model_T"

$OddBundle = Join-Path $OutRoot "bundles\\odd"
$EvenBundle = Join-Path $OutRoot "bundles\\even"
$EvalRoot = Join-Path $OutRoot "evaluation"
$SceneManifest = Join-Path $EvalRoot "scene_manifest.json"
$RoiJson = Join-Path $EvalRoot "roi.json"
$TranscriptPath = Join-Path $OutRoot "run_transcript.txt"
$StatusJson = Join-Path $OutRoot "status.json"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

Start-Transcript -Path $TranscriptPath -Append -Force | Out-Null
$status = [ordered]@{
    scene = $SceneName
    method = "M01_OursFull_Default"
    protocol = "strict_pose_controlled_repeatability"
    stage = "starting"
    started_at = (Get-Date).ToString("s")
    out_root = $OutRoot
    strict_dataset_root = $StrictDatasetRoot
    transcript = $TranscriptPath
}
try {
    Assert-Exists -PathValue $RepoRoot -Label "Repo root"
    Assert-Exists -PathValue $PythonExe -Label "Python executable"
    Assert-Exists -PathValue $DataRoot -Label "Building data root"
    Assert-Exists -PathValue $ProbeList -Label "Probe test list"
    Assert-Exists -PathValue $OddTrainList -Label "Odd train list"
    Assert-Exists -PathValue $EvenTrainList -Label "Even train list"

    Write-Status -Stage "starting"

    Invoke-Step -Name "Materialize Strict Dataset" -Body {
        Write-Status -Stage "materialize_strict_dataset"
        if (Test-Path -LiteralPath $ProtocolManifest) {
            Write-Host "[SKIP] Strict protocol manifest already exists: $ProtocolManifest"
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\materialize_strict_pose_controlled_dataset.py") `
            --scene_name $SceneName `
            --source_root $DataRoot `
            --split_dir $SplitDir `
            --out_root $StrictDatasetRoot `
            --colmap_executable colmap `
            --exiftool_executable exiftool `
            --camera SIMPLE_RADIAL `
            --matching spatial `
            --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" `
            --mapper_multiple_models 1 `
            --min_model_size 5 `
            --init_min_num_inliers 50 `
            --abs_pose_min_num_inliers 20 `
            --use_model_aligner `
            --model_aligner_args "--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0"
        if ($LASTEXITCODE -ne 0) { throw "Strict dataset materialization failed" }
    }

    Assert-Exists -PathValue $ProtocolManifest -Label "Strict protocol manifest"
    $protocol = Get-Content -LiteralPath $ProtocolManifest -Raw | ConvertFrom-Json
    $StrictRgbRoot = [string]$protocol.artifacts.strict_rgb_root
    $StrictThermalRoot = [string]$protocol.artifacts.strict_thermal_root
    $RoiSourcePoints = [string]$protocol.artifacts.roi_source_points_path

    Assert-Exists -PathValue $StrictRgbRoot -Label "Strict RGB root"
    Assert-Exists -PathValue $StrictThermalRoot -Label "Strict thermal root"
    Assert-Exists -PathValue $RoiSourcePoints -Label "ROI source points"

    Invoke-Step -Name "Build ROI" -Body {
        Write-Status -Stage "build_roi"
        if (Test-Path -LiteralPath $RoiJson) {
            Write-Host "[SKIP] ROI already exists: $RoiJson"
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\evaluator.py") build-roi `
            --scene_name $SceneName `
            --points_path $RoiSourcePoints `
            --out $RoiJson
        if ($LASTEXITCODE -ne 0) { throw "ROI build failed" }
    }

    Invoke-Step -Name "Train Odd RGB" -Body {
        Write-Status -Stage "train_odd_rgb"
        if (Test-Path -LiteralPath (Join-Path $OddRgb "chkpnt30000.pth")) {
            Write-Host "[SKIP] Odd RGB checkpoint already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "train.py") `
            --source_path $StrictRgbRoot `
            --model_path $OddRgb `
            --images images `
            --resolution 4 `
            --iterations 30000 `
            --checkpoint_iterations 30000 `
            --save_iterations 30000 `
            --data_device cuda `
            --eval `
            --train_list $OddTrainList `
            --test_list $ProbeList `
            --densify_from_iter 1500 `
            --densify_until_iter 10000 `
            --densification_interval 300 `
            --densify_grad_threshold 0.001 `
            --lambda_dssim 0.3 `
            --ss_enable `
            --ss_source colmap_sparse `
            --ss_use_aabb false `
            --ss_aabb_margin 0 `
            --ss_voxel_size 1.5 `
            --ss_nn_dist_thr 3.5 `
            --ss_adaptive_nn `
            --ss_adaptive_alpha 1.2 `
            --ss_adaptive_beta 0.2 `
            --ss_adaptive_max_scale 2.0 `
            --ss_trim_tail_pct 0.0 `
            --ss_drop_small_islands 10 `
            --ss_island_radius 10.0 `
            --ss_prune_after_rgb `
            --disable_viewer `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Odd RGB training failed" }
    }

    Invoke-Step -Name "Train Odd Thermal" -Body {
        Write-Status -Stage "train_odd_thermal"
        if (Test-Path -LiteralPath (Join-Path $OddT "chkpnt60000.pth")) {
            Write-Host "[SKIP] Odd thermal checkpoint already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "train.py") `
            --source_path $StrictThermalRoot `
            --model_path $OddT `
            --images images `
            --start_checkpoint (Join-Path $OddRgb "chkpnt30000.pth") `
            --resolution 4 `
            --iterations 60000 `
            --checkpoint_iterations 60000 `
            --eval `
            --train_list $OddTrainList `
            --test_list $ProbeList `
            --position_lr_init 0 `
            --position_lr_final 0 `
            --scaling_lr 0 `
            --rotation_lr 0 `
            --opacity_lr 0.0002 `
            --feature_lr 0.001 `
            --densify_from_iter 999999 `
            --densify_until_iter 0 `
            --densification_interval 999999 `
            --opacity_reset_interval 999999 `
            --lambda_dssim 0.05 `
            --clamp_scale_max 10.0 `
            --thermal_reset_features `
            --t_struct_grad_w 0.006 `
            --t_struct_grad_norm true `
            --disable_viewer `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Odd thermal training failed" }
    }

    Invoke-Step -Name "Export Odd Bundle" -Body {
        Write-Status -Stage "export_odd_bundle"
        if (Test-Path -LiteralPath (Join-Path $OddBundle "split_manifest.json")) {
            Write-Host "[SKIP] Odd bundle manifest already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\export_gaussian_probe_bundle.py") `
            --model_path $OddT `
            --iteration 60000 `
            --split_label odd `
            --scene_name_override $SceneName `
            --out_dir $OddBundle `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Odd bundle export failed" }
    }

    Invoke-Step -Name "Train Even RGB" -Body {
        Write-Status -Stage "train_even_rgb"
        if (Test-Path -LiteralPath (Join-Path $EvenRgb "chkpnt30000.pth")) {
            Write-Host "[SKIP] Even RGB checkpoint already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "train.py") `
            --source_path $StrictRgbRoot `
            --model_path $EvenRgb `
            --images images `
            --resolution 4 `
            --iterations 30000 `
            --checkpoint_iterations 30000 `
            --save_iterations 30000 `
            --data_device cuda `
            --eval `
            --train_list $EvenTrainList `
            --test_list $ProbeList `
            --densify_from_iter 1500 `
            --densify_until_iter 10000 `
            --densification_interval 300 `
            --densify_grad_threshold 0.001 `
            --lambda_dssim 0.3 `
            --ss_enable `
            --ss_source colmap_sparse `
            --ss_use_aabb false `
            --ss_aabb_margin 0 `
            --ss_voxel_size 1.5 `
            --ss_nn_dist_thr 3.5 `
            --ss_adaptive_nn `
            --ss_adaptive_alpha 1.2 `
            --ss_adaptive_beta 0.2 `
            --ss_adaptive_max_scale 2.0 `
            --ss_trim_tail_pct 0.0 `
            --ss_drop_small_islands 10 `
            --ss_island_radius 10.0 `
            --ss_prune_after_rgb `
            --disable_viewer `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Even RGB training failed" }
    }

    Invoke-Step -Name "Train Even Thermal" -Body {
        Write-Status -Stage "train_even_thermal"
        if (Test-Path -LiteralPath (Join-Path $EvenT "chkpnt60000.pth")) {
            Write-Host "[SKIP] Even thermal checkpoint already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "train.py") `
            --source_path $StrictThermalRoot `
            --model_path $EvenT `
            --images images `
            --start_checkpoint (Join-Path $EvenRgb "chkpnt30000.pth") `
            --resolution 4 `
            --iterations 60000 `
            --checkpoint_iterations 60000 `
            --eval `
            --train_list $EvenTrainList `
            --test_list $ProbeList `
            --position_lr_init 0 `
            --position_lr_final 0 `
            --scaling_lr 0 `
            --rotation_lr 0 `
            --opacity_lr 0.0002 `
            --feature_lr 0.001 `
            --densify_from_iter 999999 `
            --densify_until_iter 0 `
            --densification_interval 999999 `
            --opacity_reset_interval 999999 `
            --lambda_dssim 0.05 `
            --clamp_scale_max 10.0 `
            --thermal_reset_features `
            --t_struct_grad_w 0.006 `
            --t_struct_grad_norm true `
            --disable_viewer `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Even thermal training failed" }
    }

    Invoke-Step -Name "Export Even Bundle" -Body {
        Write-Status -Stage "export_even_bundle"
        if (Test-Path -LiteralPath (Join-Path $EvenBundle "split_manifest.json")) {
            Write-Host "[SKIP] Even bundle manifest already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\export_gaussian_probe_bundle.py") `
            --model_path $EvenT `
            --iteration 60000 `
            --split_label even `
            --scene_name_override $SceneName `
            --out_dir $EvenBundle `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Even bundle export failed" }
    }

    Invoke-Step -Name "Build Scene Manifest" -Body {
        Write-Status -Stage "build_scene_manifest"
        if (Test-Path -LiteralPath $SceneManifest) {
            Write-Host "[SKIP] Scene manifest already exists: $SceneManifest"
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\build_scene_manifest.py") `
            --odd_manifest (Join-Path $OddBundle "split_manifest.json") `
            --even_manifest (Join-Path $EvenBundle "split_manifest.json") `
            --roi_path $RoiJson `
            --out $SceneManifest
        if ($LASTEXITCODE -ne 0) { throw "Scene manifest build failed" }
    }

    Invoke-Step -Name "Evaluate Repeatability" -Body {
        Write-Status -Stage "evaluate_repeatability"
        if ((Test-Path -LiteralPath (Join-Path $EvalRoot "metrics.json")) -and (Test-Path -LiteralPath (Join-Path $EvalRoot "metrics.csv"))) {
            Write-Host "[SKIP] Evaluation outputs already exist."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\evaluator.py") evaluate-scene `
            --manifest $SceneManifest `
            --out_dir $EvalRoot
        if ($LASTEXITCODE -ne 0) { throw "Repeatability evaluation failed" }
    }

    $status.stage = "completed"
    $status.completed_at = (Get-Date).ToString("s")
    $status.protocol_manifest = $ProtocolManifest
    $status.metrics_json = (Join-Path $EvalRoot "metrics.json")
    $status.metrics_csv = (Join-Path $EvalRoot "metrics.csv")
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusJson -Encoding UTF8
}
catch {
    $status.stage = "failed"
    $status.failed_at = (Get-Date).ToString("s")
    $status.error = $_.Exception.Message
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusJson -Encoding UTF8
    throw
}
finally {
    Stop-Transcript | Out-Null
}
