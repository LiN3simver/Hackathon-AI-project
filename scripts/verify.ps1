$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup.ps1 first.' }
. (Join-Path $PSScriptRoot 'node_runtime.ps1')
$npm = Get-ProjectNpmCommand
& $python scripts/check_public_release.py
if ($LASTEXITCODE -ne 0) { throw 'Public source review failed.' }
& $python -m compileall -q apps/api/app scripts
if ($LASTEXITCODE -ne 0) { throw 'Python compilation failed.' }
& $npm run build:web
if ($LASTEXITCODE -ne 0) { throw 'Capture client build failed.' }
& $npm run build:ops
if ($LASTEXITCODE -ne 0) { throw 'Operations client build failed.' }
Write-Host 'Public source checks and both frontend builds passed.'
