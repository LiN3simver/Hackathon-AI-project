param(
    [switch]$WithModels,
    [switch]$WithOcr,
    [ValidateSet('cpu', 'cu126')]
    [string]$TorchVariant = 'cpu'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'node_runtime.ps1')
$npm = Get-ProjectNpmCommand
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.13 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $python -c 'import sys; assert sys.version_info[:2] == (3, 13), "Python 3.13 is required"'
if ($LASTEXITCODE -ne 0) { throw 'Create .venv with Python 3.13.' }
& $python -m pip install -r apps/api/requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
& $npm ci
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
if ($WithModels) {
    & $python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url "https://download.pytorch.org/whl/$TorchVariant"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    & $python -m pip install -r apps/api/requirements-ml.txt
    if ($LASTEXITCODE -ne 0) { throw 'Optional model dependencies failed.' }
    & $python scripts/download_model_weights.py
    if ($LASTEXITCODE -ne 0) { throw 'Upstream backbone download failed.' }
}
if ($WithOcr) {
    & (Join-Path $PSScriptRoot 'setup_plate_pretrained.ps1')
}
Write-Host 'Source dependencies are ready. Use -WithModels to provision the backend ML runtime.'
Write-Host 'After runtime provisioning, configure .env and run scripts/dev.ps1.'
Write-Host 'Task-specific checkpoints and authorized datasets must be supplied separately.'
