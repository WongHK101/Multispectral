param(
    [string]$SceneName = "Building",
    [string]$MethodName = "ThermalGaussian_MFTG",
    [string]$TrainScriptName = "train_MFTG.py",
    [string]$RepoRoot = "D:\dataset\FGS\FGS-0202v1",
    [string]$PythonExe = "D:\anaconda\envs\fgs\python.exe",
    [string]$VariantPython = "D:\anaconda\envs\TGS-main310\python.exe",
    [string]$VariantRepo = "E:\3DGS\TGS\Thermal-Gaussian-main",
    [string]$StrictRunRoot = "F:\databackup\xr6\output\GeometricRepeatability\Building\M01_Strict_v1",
    [string]$OutRoot = "F:\databackup\xr6\output\GeometricRepeatability\CrossMethod\ThermalGaussian_MFTG\Building\Strict_v1",
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

function Ensure-PointCloudJunction {
    param(
        [Parameter(Mandatory = $true)][string]$ModelRoot
    )
    $targetDir = Join-Path $ModelRoot "point_cloud_thermal"
    $junction = Join-Path $ModelRoot "point_cloud"
    if (Test-Path -LiteralPath $junction) {
        return
    }
    Assert-Exists -PathValue $targetDir -Label "Thermal point cloud dir"
    cmd /c mklink /J "$junction" "$targetDir" | Out-Null
    if (-not (Test-Path -LiteralPath $junction)) {
        try {
            New-Item -ItemType Junction -Path $junction -Target $targetDir -Force | Out-Null
        }
        catch {
        }
    }
    if (-not (Test-Path -LiteralPath $junction)) {
        throw "Failed to create point_cloud junction for $ModelRoot"
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

$ProtocolManifest = Join-Path $StrictRunRoot "strict_dataset\strict_protocol_manifest.json"
$RoiJson = Join-Path $StrictRunRoot "evaluation\roi.json"
$ProbeList = Join-Path $SplitDir "probe_test.txt"
$OddTrainList = Join-Path $SplitDir "train_odd.txt"
$EvenTrainList = Join-Path $SplitDir "train_even.txt"

$OddDataset = Join-Path $OutRoot "split_datasets\odd"
$EvenDataset = Join-Path $OutRoot "split_datasets\even"
$OddModel = Join-Path $OutRoot "odd"
$EvenModel = Join-Path $OutRoot "even"
$OddBundle = Join-Path $OutRoot "bundles\odd"
$EvenBundle = Join-Path $OutRoot "bundles\even"
$EvalRoot = Join-Path $OutRoot "evaluation"
$SceneManifest = Join-Path $EvalRoot "scene_manifest.json"
$TranscriptPath = Join-Path $OutRoot "run_transcript.txt"
$StatusJson = Join-Path $OutRoot "status.json"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

Start-Transcript -Path $TranscriptPath -Append -Force | Out-Null
$status = [ordered]@{
    scene = $SceneName
    method = $MethodName
    protocol = "strict_pose_controlled_repeatability"
    stage = "starting"
    started_at = (Get-Date).ToString("s")
    strict_run_root = $StrictRunRoot
    out_root = $OutRoot
    transcript = $TranscriptPath
}

try {
    Assert-Exists -PathValue $RepoRoot -Label "Repo root"
    Assert-Exists -PathValue $PythonExe -Label "FGS python"
    Assert-Exists -PathValue $VariantPython -Label "Variant python"
    Assert-Exists -PathValue $VariantRepo -Label "Variant repo"
    Assert-Exists -PathValue $ProtocolManifest -Label "Strict protocol manifest"
    Assert-Exists -PathValue $RoiJson -Label "ROI json"
    Assert-Exists -PathValue $ProbeList -Label "Probe list"
    Assert-Exists -PathValue $OddTrainList -Label "Odd train list"
    Assert-Exists -PathValue $EvenTrainList -Label "Even train list"

    $protocol = Get-Content -LiteralPath $ProtocolManifest -Raw | ConvertFrom-Json
    $StrictRgbRoot = [string]$protocol.artifacts.strict_rgb_root
    $StrictThermalRoot = [string]$protocol.artifacts.strict_thermal_root
    $StrictRgbImages = Join-Path $StrictRgbRoot "images"
    $StrictThermalImages = Join-Path $StrictThermalRoot "images"
    $StrictSparseRoot = Join-Path $StrictRgbRoot "sparse"

    Assert-Exists -PathValue $StrictRgbImages -Label "Strict RGB images"
    Assert-Exists -PathValue $StrictThermalImages -Label "Strict thermal images"
    Assert-Exists -PathValue $StrictSparseRoot -Label "Strict sparse root"

    Write-Status -Stage "starting"

    Invoke-Step -Name "Materialize Odd Split Dataset" -Body {
        Write-Status -Stage "materialize_odd_dataset"
        if (Test-Path -LiteralPath (Join-Path $OddDataset "split_dataset_manifest.json")) {
            Write-Host "[SKIP] Odd split dataset manifest already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\materialize_rgbt_split_view_dataset.py") `
            --rgb_src_dir $StrictRgbImages `
            --thermal_src_dir $StrictThermalImages `
            --sparse_src_root $StrictSparseRoot `
            --train_list $OddTrainList `
            --test_list $ProbeList `
            --out_root $OddDataset
        if ($LASTEXITCODE -ne 0) { throw "Odd split dataset materialization failed" }
    }

    Invoke-Step -Name "Materialize Even Split Dataset" -Body {
        Write-Status -Stage "materialize_even_dataset"
        if (Test-Path -LiteralPath (Join-Path $EvenDataset "split_dataset_manifest.json")) {
            Write-Host "[SKIP] Even split dataset manifest already exists."
            return
        }
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\materialize_rgbt_split_view_dataset.py") `
            --rgb_src_dir $StrictRgbImages `
            --thermal_src_dir $StrictThermalImages `
            --sparse_src_root $StrictSparseRoot `
            --train_list $EvenTrainList `
            --test_list $ProbeList `
            --out_root $EvenDataset
        if ($LASTEXITCODE -ne 0) { throw "Even split dataset materialization failed" }
    }

    Invoke-Step -Name "Train Odd Variant" -Body {
        Write-Status -Stage "train_odd"
        if (Test-Path -LiteralPath (Join-Path $OddModel "point_cloud_thermal\iteration_30000\point_cloud.ply")) {
            Write-Host "[SKIP] Odd variant point cloud already exists."
            return
        }
        & $VariantPython (Join-Path $VariantRepo $TrainScriptName) `
            -s $OddDataset `
            -m $OddModel `
            --iterations 30000 `
            --test_iterations 30000 `
            --save_iterations 30000 `
            --checkpoint_iterations 30000 `
            --eval `
            -r 4
        if ($LASTEXITCODE -ne 0) { throw "Odd $MethodName training failed" }
    }

    Invoke-Step -Name "Export Odd Bundle" -Body {
        Write-Status -Stage "export_odd_bundle"
        if (Test-Path -LiteralPath (Join-Path $OddBundle "split_manifest.json")) {
            Write-Host "[SKIP] Odd bundle manifest already exists."
            return
        }
        Ensure-PointCloudJunction -ModelRoot $OddModel
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\export_gaussian_probe_bundle.py") `
            --model_path $OddModel `
            --source_path $StrictThermalRoot `
            --images images `
            --train_list $OddTrainList `
            --test_list $ProbeList `
            --eval `
            --iteration 30000 `
            --split_label odd `
            --scene_name_override $SceneName `
            --out_dir $OddBundle `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Odd bundle export failed" }
    }

    Invoke-Step -Name "Train Even Variant" -Body {
        Write-Status -Stage "train_even"
        if (Test-Path -LiteralPath (Join-Path $EvenModel "point_cloud_thermal\iteration_30000\point_cloud.ply")) {
            Write-Host "[SKIP] Even variant point cloud already exists."
            return
        }
        & $VariantPython (Join-Path $VariantRepo $TrainScriptName) `
            -s $EvenDataset `
            -m $EvenModel `
            --iterations 30000 `
            --test_iterations 30000 `
            --save_iterations 30000 `
            --checkpoint_iterations 30000 `
            --eval `
            -r 4
        if ($LASTEXITCODE -ne 0) { throw "Even $MethodName training failed" }
    }

    Invoke-Step -Name "Export Even Bundle" -Body {
        Write-Status -Stage "export_even_bundle"
        if (Test-Path -LiteralPath (Join-Path $EvenBundle "split_manifest.json")) {
            Write-Host "[SKIP] Even bundle manifest already exists."
            return
        }
        Ensure-PointCloudJunction -ModelRoot $EvenModel
        & $PythonExe (Join-Path $RepoRoot "tools\geometric_repeatability\export_gaussian_probe_bundle.py") `
            --model_path $EvenModel `
            --source_path $StrictThermalRoot `
            --images images `
            --train_list $EvenTrainList `
            --test_list $ProbeList `
            --eval `
            --iteration 30000 `
            --split_label even `
            --scene_name_override $SceneName `
            --out_dir $EvenBundle `
            --quiet
        if ($LASTEXITCODE -ne 0) { throw "Even bundle export failed" }
    }

    Invoke-Step -Name "Build Scene Manifest" -Body {
        Write-Status -Stage "build_scene_manifest"
        if (Test-Path -LiteralPath $SceneManifest) {
            Write-Host "[SKIP] Scene manifest already exists."
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
    $status.roi_path = $RoiJson
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
