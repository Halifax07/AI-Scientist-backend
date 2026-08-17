param(
    [switch]$WithAgentRuntime,
    [switch]$WithExperimentTools
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot

$uvArguments = @('sync', '--link-mode', 'copy', '--extra', 'dev')
if ($WithAgentRuntime) {
    $uvArguments += @('--extra', 'agent')
}
if ($WithExperimentTools) {
    $uvArguments += @(
        '--extra', 'experiment',
        '--extra', 'vision',
        '--extra', 'detectors'
    )
}

Write-Host "Creating project environment in $projectRoot\.venv"
& uv @uvArguments

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) {
    Copy-Item -LiteralPath (Join-Path $projectRoot '.env.example') -Destination (Join-Path $projectRoot '.env')
    Write-Host 'Created .env from .env.example; add DASHSCOPE_API_KEY before enabling AgentScope.'
}

Write-Host 'Bootstrap complete.'
