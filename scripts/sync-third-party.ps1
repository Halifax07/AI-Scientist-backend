$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$thirdPartyRoot = (Resolve-Path (Join-Path $projectRoot 'third_party')).Path
$manifestPath = Join-Path $thirdPartyRoot 'manifest.lock.json'
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json

foreach ($repository in $manifest.repositories) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $repository.path))
    if (-not $target.StartsWith($thirdPartyRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use a target outside third_party: $target"
    }

    if (-not (Test-Path -LiteralPath $target)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        $previousLfsSkip = $env:GIT_LFS_SKIP_SMUDGE
        $env:GIT_LFS_SKIP_SMUDGE = '1'
        try {
            git clone --quiet --no-checkout --filter=blob:none $repository.url $target
            git -C $target fetch --quiet --depth 1 origin $repository.commit
            if ($repository.PSObject.Properties.Name -contains 'sparse_exclude') {
                git -C $target sparse-checkout init --no-cone
                $sparsePatterns = @('/*')
                foreach ($excludedPath in $repository.sparse_exclude) {
                    $sparsePatterns += "!/$excludedPath/"
                }
                git -C $target sparse-checkout set --no-cone @sparsePatterns
            }
            git -C $target checkout --quiet --detach $repository.commit
        }
        finally {
            $env:GIT_LFS_SKIP_SMUDGE = $previousLfsSkip
        }
        Write-Host "Downloaded $($repository.name) at $($repository.commit)"
        continue
    }

    if (-not (Test-Path -LiteralPath (Join-Path $target '.git'))) {
        throw "Existing target is not a Git repository: $target"
    }
    $actualRemote = git -C $target remote get-url origin
    $actualCommit = git -C $target rev-parse HEAD
    if ($actualRemote -ne $repository.url) {
        throw "Remote mismatch for $($repository.name): $actualRemote"
    }
    if ($actualCommit -ne $repository.commit) {
        throw "Commit mismatch for $($repository.name): expected $($repository.commit), got $actualCommit"
    }
    if ($repository.PSObject.Properties.Name -contains 'sparse_exclude') {
        foreach ($excludedPath in $repository.sparse_exclude) {
            if (Test-Path -LiteralPath (Join-Path $target $excludedPath)) {
                throw "Excluded path is present for $($repository.name): $excludedPath"
            }
        }
    }
    Write-Host "Verified $($repository.name) at $actualCommit"
}
