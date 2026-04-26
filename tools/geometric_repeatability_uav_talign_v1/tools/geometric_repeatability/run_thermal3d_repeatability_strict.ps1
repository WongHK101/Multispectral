param(
    [string]$SceneName = "Building",
    [string]$RepoRoot = "D:\dataset\FGS\FGS-0202v1",
    [string]$PythonExe = "D:\anaconda\envs\fgs\python.exe",
    [string]$Thermal3DPython = "D:\anaconda\envs\fgs\python.exe",
    [string]$Thermal3DRepo = "E:\3DGS\Thermal3D-GS",
    [string]$StrictRunRoot = "F:\databackup\xr6\output\GeometricRepeatability\Building\M01_Strict_v1",
    [string]$OutRoot = "F:\databackup\xr6\output\GeometricRepeatability\CrossMethod\Thermal3D_GS\Building\Strict_v1",
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

$ProtocolManifest = Join-Path $StrictRunRoot "strict_dataset\strict_protocol_manifest.json"
$RoiJson = Join-Path $StrictRunRoot "evaluation\roi.json"
$ProbeList = Join-Path $SplitDir "probe_test.txt"
$OddTrainList = Join-Path $SplitDir "train_odd.txt"
$EvenTrainList = Join-Path $SplitDir "train_even.txt"

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
    method = "Thermal3D_GS"
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
    Assert-Exists -PathValue $Thermal3DPython -Label "Thermal3D python"
    Assert-Exists -PathValue $Thermal3DRepo -Label "Thermal3D repo"
    Assert-Exists -PathValue $ProtocolManifest -Label "Strict protocol manifest"
    Assert-Exists -PathValue $RoiJson -Label "ROI json"
    Assert-Exists -PathValue $ProbeList -Label "Probe list"
    Assert-Exists -PathValue $OddTrainList -Label "Odd train list"
    Assert-Exists -PathValue $EvenTrainList -Label "Even train list"

    $protocol = Get-Content -LiteralPath $ProtocolManifest -Raw | ConvertFrom-Json
    $StrictThermalRoot = [string]$protocol.artifacts.strict_thermal_root

    Assert-Exists -PathValue $StrictThermalRoot -Label "Strict thermal root"

    Write-Status -Stage "starting"

    Invoke-Step -Name "Train Odd Thermal3D-GS" -Body {
        Write-Status -Stage "train_odd"
        if (Test-Path -LiteralPath (Join-Path $OddModel "point_cloud\iteration_30000\point_cloud.ply")) {
            Write-Host "[SKIP] Odd Thermal3D point cloud already exists."
            return
        }
        & $Thermal3DPython (Join-Path $Thermal3DRepo "train.py") `
            -s $StrictThermalRoot `
            -m $OddModel `
            --iterations 30000 `
            --test_iterations 30000 `
            --save_iterations 30000 `
            --densify_until_iter 4000 `
            --eval `
            -r 4 `
            --load2gpu_on_the_fly `
            --train_list $OddTrainList `
            --test_list $ProbeList
        if ($LASTEXITCODE -ne 0) { throw "Odd Thermal3D training failed" }
    }

    Invoke-Step -Name "Export Odd Bundle" -Body {
        Write-Status -Stage "export_odd_bundle"
        if (Test-Path -LiteralPath (Join-Path $OddBundle "split_manifest.json")) {
            Write-Host "[SKIP] Odd bundle manifest already exists."
            return
        }
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

    Invoke-Step -Name "Train Even Thermal3D-GS" -Body {
        Write-Status -Stage "train_even"
        if (Test-Path -LiteralPath (Join-Path $EvenModel "point_cloud\iteration_30000\point_cloud.ply")) {
            Write-Host "[SKIP] Even Thermal3D point cloud already exists."
            return
        }
        & $Thermal3DPython (Join-Path $Thermal3DRepo "train.py") `
            -s $StrictThermalRoot `
            -m $EvenModel `
            --iterations 30000 `
            --test_iterations 30000 `
            --save_iterations 30000 `
            --densify_until_iter 4000 `
            --eval `
            -r 4 `
            --load2gpu_on_the_fly `
            --train_list $EvenTrainList `
            --test_list $ProbeList
        if ($LASTEXITCODE -ne 0) { throw "Even Thermal3D training failed" }
    }

    Invoke-Step -Name "Export Even Bundle" -Body {
        Write-Status -Stage "export_even_bundle"
        if (Test-Path -LiteralPath (Join-Path $EvenBundle "split_manifest.json")) {
            Write-Host "[SKIP] Even bundle manifest already exists."
            return
        }
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
