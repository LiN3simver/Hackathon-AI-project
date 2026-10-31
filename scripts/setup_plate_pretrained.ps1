$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$ocrEnvironment = Join-Path $projectRoot '.venv-rapidocr'
$python = Join-Path $ocrEnvironment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run scripts/setup.ps1 first.' }
    & $projectPython -m venv $ocrEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the OCR environment.' }
}
& $python -m pip install --no-input 'rapidocr==3.9.2' 'onnxruntime==1.26.0'
if ($LASTEXITCODE -ne 0) { throw 'OCR dependency installation failed.' }
& $python -c 'from rapidocr import RapidOCR; RapidOCR(); print("RapidOCR CPU ready")'
if ($LASTEXITCODE -ne 0) { throw 'OCR model initialization failed.' }
