$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup.ps1 first.' }
& $python -c 'import torch'
if ($LASTEXITCODE -ne 0) { throw 'Run scripts/setup.ps1 -WithModels to provision the backend ML runtime.' }
. (Join-Path $PSScriptRoot 'node_runtime.ps1')
$npm = Get-ProjectNpmCommand
$escapedNpm = $npm.Replace("'", "''")
$startedServices = @()
try {
    $startedServices += Start-Process -FilePath $python -WindowStyle Hidden -WorkingDirectory $projectRoot -PassThru -ArgumentList @(
        '-m', 'uvicorn', 'app.main:app', '--app-dir', 'apps/api', '--host', '127.0.0.1', '--port', '8000'
    )
    $startedServices += Start-Process -FilePath $python -WindowStyle Hidden -WorkingDirectory $projectRoot -PassThru -ArgumentList @(
        'scripts/run_inspection_worker.py'
    )
    foreach ($frontend in @('web', 'ops')) {
        $startedServices += Start-Process powershell.exe -WindowStyle Hidden -WorkingDirectory $projectRoot -PassThru -ArgumentList @(
            '-NoProfile', '-Command', "& '$escapedNpm' run dev:$frontend"
        )
    }
    Write-Host 'Capture: https://localhost:5173'
    Write-Host 'Operations: http://localhost:5174'
    Write-Host 'API: http://127.0.0.1:8000/docs'
    Write-Host ('Started process IDs: ' + (($startedServices | ForEach-Object { $_.Id }) -join ', '))
    Write-Host 'Press Ctrl+C to stop the local development processes.'
    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($service in $startedServices) {
            if ($service.HasExited) { throw ('A development process exited: ' + $service.Id) }
        }
    }
} finally {
    foreach ($service in $startedServices) {
        if (-not $service.HasExited) {
            # Stop only the process trees created by this invocation.
            & taskkill.exe /PID $service.Id /T /F 2>$null | Out-Null
        }
    }
}
