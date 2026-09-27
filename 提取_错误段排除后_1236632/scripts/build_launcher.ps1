param(
    [switch]$InstallPyInstaller,
    [string]$GptRoot
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$defaultGptRoot = Join-Path $root '..\GPT-SoVITS-v2pro-20250604'
if (-not $GptRoot) { $GptRoot = $defaultGptRoot }
$GptRoot = (Resolve-Path $GptRoot).Path
$runtime = Join-Path $GptRoot 'runtime\python.exe'
if (-not (Test-Path $runtime)) { throw "GPT-SoVITS runtime was not found: $runtime" }
if ($InstallPyInstaller) { & $runtime -m pip install pyinstaller }
& $runtime -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller is not installed. Run .\scripts\build_launcher.ps1 -InstallPyInstaller' }
$required = @(
    (Join-Path $GptRoot 'tools\uvr5'),
    (Join-Path $GptRoot 'GPT_SoVITS\eres2net'),
    (Join-Path $GptRoot 'runtime\ffmpeg.exe'),
    (Join-Path $GptRoot 'runtime\ffprobe.exe')
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) { throw "Desktop build dependency was not found: $path" }
}
$previousBuildRoot = $env:VOICE_EXTRACT_BUILD_GPT_ROOT
$env:VOICE_EXTRACT_BUILD_GPT_ROOT = $GptRoot
Push-Location $root
try {
    & $runtime -m PyInstaller --noconfirm --clean (Join-Path $root 'VoiceExtractor.spec')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
} finally {
    Pop-Location
    $env:VOICE_EXTRACT_BUILD_GPT_ROOT = $previousBuildRoot
}
Write-Host ("Built {0}" -f (Join-Path $root 'dist\VoiceExtractor.exe')) -ForegroundColor Green
