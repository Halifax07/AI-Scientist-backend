$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }

Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
& $pythonExecutable -m fsad_scientist.cli demo-ledger
