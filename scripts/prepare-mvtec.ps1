param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [switch]$AcceptCcByNcSa
)

$ErrorActionPreference = 'Stop'
if (-not $AcceptCcByNcSa) {
    throw 'MVTec AD is CC BY-NC-SA 4.0. Re-run with -AcceptCcByNcSa after reviewing the official license.'
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rawRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'data\raw'))
$archive = (Resolve-Path -LiteralPath $ArchivePath).Path
$target = [IO.Path]::GetFullPath((Join-Path $rawRoot 'mvtec_ad'))
$staging = [IO.Path]::GetFullPath((Join-Path $rawRoot ('.mvtec-extract-' + [guid]::NewGuid().ToString('N'))))

foreach ($path in @($target, $staging)) {
    if (-not $path.StartsWith($rawRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved path escapes data/raw: $path"
    }
}
if (Test-Path -LiteralPath $target) {
    throw "Target already exists; refusing to overwrite: $target"
}
if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
    throw 'The tar executable is required to extract the official archive.'
}

New-Item -ItemType Directory -Path $staging | Out-Null
Write-Host "Extracting $archive"
& tar -xf $archive -C $staging
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE; staging data was preserved at $staging"
}

$candidateRoots = @($staging) + @(
    Get-ChildItem -LiteralPath $staging -Directory -Recurse -Depth 2 | Select-Object -ExpandProperty FullName
)
$datasetRoot = $candidateRoots | Where-Object {
    (Test-Path -LiteralPath (Join-Path $_ 'bottle\train\good')) -and
    (Test-Path -LiteralPath (Join-Path $_ 'zipper\test'))
} | Select-Object -First 1
if (-not $datasetRoot) {
    throw "No MVTec AD category root was found; staging data was preserved at $staging"
}

$resolvedDatasetRoot = [IO.Path]::GetFullPath($datasetRoot)
if (-not $resolvedDatasetRoot.StartsWith($staging, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Discovered dataset root escapes staging: $resolvedDatasetRoot"
}
Move-Item -LiteralPath $resolvedDatasetRoot -Destination $target

if (Test-Path -LiteralPath $staging) {
    $remaining = @(Get-ChildItem -LiteralPath $staging -Force)
    if ($remaining.Count -eq 0) {
        Remove-Item -LiteralPath $staging
    }
}

Set-Location -LiteralPath $projectRoot
& uv run fsad-scientist scan-dataset $target
if ($LASTEXITCODE -ne 0) {
    throw "Dataset extraction succeeded, but the audit failed for $target"
}
Write-Host "MVTec AD prepared at $target"
