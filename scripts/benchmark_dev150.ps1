param(
    [Parameter(Mandatory = $true)][string]$DatasetRoot,
    [string]$Name = "local_dev150",
    [string]$PythonExe = "python",
    [string]$Device = "auto",
    [ValidateSet("auto", "cpu", "cuda")][string]$OcrDevice = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
& $PythonExe tools/run_dev_benchmark.py --dataset-root $DatasetRoot --name $Name --device $Device --ocr-device $OcrDevice
