param(
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$OutputDir = "runs/local_single",
    [string]$PythonExe = "python",
    [string]$Device = "auto",
    [ValidateSet("auto", "cpu", "cuda")][string]$OcrDevice = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $PythonExe main.py `
    --image $Image `
    --output (Join-Path $OutputDir "result.json") `
    --orchestrator modular `
    --pipeline_config configs/current.json `
    --weights models/component_yolo11n_continue_v2_best.pt `
    --device $Device `
    --ocr_backend hybrid `
    --ocr_device $OcrDevice `
    --save_debug_images
