<#
.SYNOPSIS
    Local sandbox test for AINM Object Detection submissions.
    Builds a Docker container matching the competition environment and runs your submission.

.DESCRIPTION
    Replicates: Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0,
    onnxruntime-gpu 1.20.0, NVIDIA L4 constraints (300s timeout, 8GB RAM, no network).

.PARAMETER Submission
    Path to submission directory or .zip file. Default: parent directory (where run.py lives).

.PARAMETER Images
    Path to test images directory. Default: data\coco\train\images (first 5 images).

.PARAMETER MaxImages
    Number of images to test with (copies first N from Images dir). Default: 5.

.PARAMETER NoBuild
    Skip Docker image rebuild (use cached image).

.PARAMETER NoGpu
    Run without GPU (CPU-only mode).

.PARAMETER Timeout
    Container timeout in seconds. Default: 300 (matches sandbox).

.EXAMPLE
    .\test_local.ps1
    .\test_local.ps1 -MaxImages 10
    .\test_local.ps1 -Submission ..\submissions\submission.zip -NoGpu
#>

param(
    [string]$Submission = "",
    [string]$Images = "",
    [int]$MaxImages = 5,
    [switch]$NoBuild,
    [switch]$NoGpu,
    [int]$Timeout = 300
)

$ErrorActionPreference = "Stop"
$IMAGE_NAME = "ainm-sandbox"
$CONTAINER_NAME = "ainm-test"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR

# ── Resolve submission directory ────────────────────────────────────
if ($Submission -eq "") {
    $SubmissionDir = $PROJECT_DIR
} elseif ($Submission -match "\.zip$") {
    # Unzip to temp dir
    $TempDir = Join-Path $env:TEMP "ainm-submission-test"
    if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
    Expand-Archive -Path $Submission -DestinationPath $TempDir
    $SubmissionDir = $TempDir
} else {
    $SubmissionDir = $Submission
}

# Verify run.py exists
if (-not (Test-Path (Join-Path $SubmissionDir "run.py"))) {
    Write-Error "run.py not found in $SubmissionDir"
    exit 1
}
Write-Host "[OK] Found run.py in $SubmissionDir" -ForegroundColor Green

# ── Prepare test images ─────────────────────────────────────────────
$TestImagesDir = Join-Path $SCRIPT_DIR "test_images"
if (Test-Path $TestImagesDir) { Remove-Item -Recurse -Force $TestImagesDir }
New-Item -ItemType Directory -Path $TestImagesDir | Out-Null

if ($Images -eq "") {
    $Images = Join-Path $PROJECT_DIR "data\coco\train\images"
}

if (-not (Test-Path $Images)) {
    Write-Error "Images directory not found: $Images"
    exit 1
}

$allImages = Get-ChildItem -Path $Images -Include *.jpg,*.jpeg,*.png -File | Select-Object -First $MaxImages
if ($allImages.Count -eq 0) {
    Write-Error "No images found in $Images"
    exit 1
}

foreach ($img in $allImages) {
    Copy-Item $img.FullName -Destination $TestImagesDir
}
Write-Host "[OK] Copied $($allImages.Count) test images" -ForegroundColor Green

# ── Output directory ────────────────────────────────────────────────
$OutputDir = Join-Path $SCRIPT_DIR "test_output"
if (Test-Path $OutputDir) { Remove-Item -Recurse -Force $OutputDir }
New-Item -ItemType Directory -Path $OutputDir | Out-Null

# ── Build Docker image ──────────────────────────────────────────────
if (-not $NoBuild) {
    Write-Host "`n=== Building Docker image ===" -ForegroundColor Cyan
    docker build -t $IMAGE_NAME -f (Join-Path $SCRIPT_DIR "Dockerfile") $SCRIPT_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
        exit 1
    }
    Write-Host "[OK] Docker image built" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Using cached Docker image" -ForegroundColor Yellow
}

# ── Remove old container if exists ──────────────────────────────────
docker rm -f $CONTAINER_NAME 2>$null | Out-Null

# ── Run container ───────────────────────────────────────────────────
Write-Host "`n=== Running submission in sandbox ===" -ForegroundColor Cyan
Write-Host "  Submission: $SubmissionDir"
Write-Host "  Images:     $($allImages.Count) files"
Write-Host "  GPU:        $(if ($NoGpu) {'OFF (CPU only)'} else {'ON'})"
Write-Host "  Timeout:    ${Timeout}s"
Write-Host "  Memory:     8GB limit"
Write-Host ""

# Build docker run command
$dockerArgs = @(
    "run"
    "--name", $CONTAINER_NAME
    "--rm"
    # Memory limit matching sandbox
    "--memory", "8g"
    "--memory-swap", "8g"
    # No network (matches sandbox)
    "--network", "none"
    # Timeout
    "--stop-timeout", $Timeout.ToString()
    # Mount submission code (read-only)
    "-v", "${SubmissionDir}:/submission:ro"
    # Mount test images (read-only)
    "-v", "${TestImagesDir}:/data/images:ro"
    # Mount output directory
    "-v", "${OutputDir}:/output"
)

# GPU support
if (-not $NoGpu) {
    $dockerArgs += @("--gpus", "all")
}

$dockerArgs += $IMAGE_NAME

$startTime = Get-Date
Write-Host "--- Container stdout/stderr ---" -ForegroundColor DarkGray

# Run with timeout
$job = Start-Job -ScriptBlock {
    param($args)
    & docker @args 2>&1
} -ArgumentList (,$dockerArgs)

$completed = $job | Wait-Job -Timeout $Timeout
$elapsed = ((Get-Date) - $startTime).TotalSeconds

if ($null -eq $completed) {
    Write-Host "`n[TIMEOUT] Container exceeded ${Timeout}s — killing" -ForegroundColor Red
    docker kill $CONTAINER_NAME 2>$null | Out-Null
    $job | Stop-Job
    $output = $job | Receive-Job
    $output | ForEach-Object { Write-Host $_ }
    $exitCode = 124
} else {
    $output = $job | Receive-Job
    $output | ForEach-Object { Write-Host $_ }
    # Get exit code from docker
    $exitCode = $job.ChildJobs[0].JobStateInfo.Reason.ExitCode
    if ($null -eq $exitCode) { $exitCode = 0 }
}
$job | Remove-Job -Force

Write-Host "--- End container output ---" -ForegroundColor DarkGray

# ── Validate output ─────────────────────────────────────────────────
Write-Host "`n=== Validation Results ===" -ForegroundColor Cyan

$predictionsPath = Join-Path $OutputDir "predictions.json"

# Exit code
if ($exitCode -eq 0) {
    Write-Host "[PASS] Exit code: 0" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Exit code: $exitCode" -ForegroundColor Red
}

# Timing
Write-Host "[INFO] Elapsed: $([math]::Round($elapsed, 1))s / ${Timeout}s" -ForegroundColor $(if ($elapsed -lt $Timeout) {'Green'} else {'Red'})

# predictions.json exists
if (-not (Test-Path $predictionsPath)) {
    Write-Host "[FAIL] predictions.json not found in output" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] predictions.json exists" -ForegroundColor Green

# Parse and validate JSON
try {
    $predictions = Get-Content $predictionsPath -Raw | ConvertFrom-Json
} catch {
    Write-Host "[FAIL] predictions.json is not valid JSON" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Valid JSON array with $($predictions.Count) predictions" -ForegroundColor Green

# Validate prediction format
$errors = @()
$imageIds = @{}
$sampleShown = $false

foreach ($pred in $predictions) {
    # Required fields
    if ($null -eq $pred.image_id) { $errors += "Missing image_id"; continue }
    if ($null -eq $pred.category_id) { $errors += "Missing category_id"; continue }
    if ($null -eq $pred.bbox) { $errors += "Missing bbox"; continue }
    if ($null -eq $pred.score) { $errors += "Missing score"; continue }

    # Types
    if ($pred.category_id -lt 0 -or $pred.category_id -gt 355) {
        $errors += "category_id $($pred.category_id) out of range [0, 355]"
    }
    if ($pred.bbox.Count -ne 4) {
        $errors += "bbox must have 4 elements, got $($pred.bbox.Count)"
    }
    if ($pred.score -lt 0 -or $pred.score -gt 1) {
        $errors += "score $($pred.score) out of range [0, 1]"
    }

    $imageIds[$pred.image_id] = $true

    if (-not $sampleShown) {
        Write-Host "[INFO] Sample prediction:" -ForegroundColor Cyan
        Write-Host "       image_id=$($pred.image_id) category_id=$($pred.category_id) bbox=[$($pred.bbox -join ', ')] score=$($pred.score)"
        $sampleShown = $true
    }
}

$uniqueErrors = $errors | Sort-Object -Unique
if ($uniqueErrors.Count -eq 0) {
    Write-Host "[PASS] All predictions have valid format" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Format errors found:" -ForegroundColor Red
    $uniqueErrors | Select-Object -First 10 | ForEach-Object { Write-Host "       $_" -ForegroundColor Red }
}

Write-Host "[INFO] Unique image_ids: $($imageIds.Count) (expected: $($allImages.Count))" -ForegroundColor $(if ($imageIds.Count -eq $allImages.Count) {'Green'} else {'Yellow'})
Write-Host "[INFO] Total predictions: $($predictions.Count)" -ForegroundColor Cyan
Write-Host "[INFO] Avg predictions/image: $([math]::Round($predictions.Count / [math]::Max(1, $imageIds.Count), 1))" -ForegroundColor Cyan

# ── Summary ─────────────────────────────────────────────────────────
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$allPassed = ($exitCode -eq 0) -and (Test-Path $predictionsPath) -and ($uniqueErrors.Count -eq 0) -and ($elapsed -lt $Timeout)
if ($allPassed) {
    Write-Host "ALL CHECKS PASSED — ready to submit!" -ForegroundColor Green
} else {
    Write-Host "SOME CHECKS FAILED — fix issues before submitting" -ForegroundColor Red
}

# Cleanup temp dir if we unzipped
if ($Submission -match "\.zip$" -and (Test-Path $TempDir)) {
    Remove-Item -Recurse -Force $TempDir
}
