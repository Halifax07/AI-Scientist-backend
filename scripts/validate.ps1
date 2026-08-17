$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }

Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& $pythonExecutable -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExecutable -m ruff check src tests scripts
exit $LASTEXITCODE
