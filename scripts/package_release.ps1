param([switch]$InstallPyInstaller, [string]$GptRoot)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build_launcher.ps1') -InstallPyInstaller:$InstallPyInstaller -GptRoot $GptRoot
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Copy-Item -LiteralPath (Join-Path $taskRoot 'README.md') -Destination (Join-Path $taskRoot 'dist\README.md') -Force
$taskFiles = @('VoiceExtractor.exe', 'VoiceExtractor_resources.zip')
$taskHashes = foreach ($taskFile in $taskFiles) {
    $taskHash = Get-FileHash -LiteralPath (Join-Path $taskRoot "dist\$taskFile") -Algorithm SHA256
    '{0}  {1}' -f $taskHash.Hash.ToLowerInvariant(), $taskFile
}
$taskHashes | Set-Content -LiteralPath (Join-Path $taskRoot 'dist\SHA256SUMS.txt') -Encoding ascii
Write-Host 'Upload VoiceExtractor.exe and VoiceExtractor_resources.zip to the v3.0 GitHub Release.'
