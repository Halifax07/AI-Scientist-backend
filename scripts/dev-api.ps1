$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonExecutable = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }

Set-Location -LiteralPath $projectRoot
& $pythonExecutable -m uvicorn fsad_scientist.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload --reload-dir src
