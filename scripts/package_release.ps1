param([switch]$InstallPyInstaller, [string]$GptRoot)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build_launcher.ps1') -InstallPyInstaller:$InstallPyInstaller -GptRoot $GptRoot
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Copy-Item -LiteralPath (Join-Path $taskRoot 'README.md') -Destination (Join-Path $taskRoot 'dist\README.md') -Force
$taskFiles = @('VoiceExtractor.exe', 'VoiceExtractor_resources.zip')
$taskModel = Join-Path $taskRoot 'model\speaker\wavlm-base-plus-sv\pytorch_model.bin'
$taskManifest = Get-Content -LiteralPath (Join-Path $taskRoot 'launcher\desktop_bootstrap.py') -Raw
$taskMatch = [regex]::Match($taskManifest, 'DirectFile\(\s*"pytorch_model.bin",\s*"[^"]+",\s*"([a-f0-9]{64})"')
if (-not $taskMatch.Success) { throw 'The desktop manifest must pin the trained WavLM checksum.' }
if ((Get-FileHash -LiteralPath $taskModel -Algorithm SHA256).Hash -ne $taskMatch.Groups[1].Value) {
    throw 'Trained WavLM checksum differs from the desktop manifest. Update the release manifest before packaging.'
}
$taskModelName = 'wavlm_reinforced_pytorch_model.bin'
Copy-Item -LiteralPath $taskModel -Destination (Join-Path $taskRoot "dist\$taskModelName") -Force
$taskFiles += $taskModelName
$taskHashes = foreach ($taskFile in $taskFiles) {
    $taskHash = Get-FileHash -LiteralPath (Join-Path $taskRoot "dist\$taskFile") -Algorithm SHA256
    '{0}  {1}' -f $taskHash.Hash.ToLowerInvariant(), $taskFile
}
$taskHashes | Set-Content -LiteralPath (Join-Path $taskRoot 'dist\SHA256SUMS.txt') -Encoding ascii
Write-Host 'Release artifacts and SHA256SUMS.txt are ready in dist.'
