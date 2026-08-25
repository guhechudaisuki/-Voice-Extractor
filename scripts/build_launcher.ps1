param([switch]$InstallPyInstaller)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = Join-Path $root '..\GPT-SoVITS-v2pro-20250604\runtime\python.exe'
if (-not (Test-Path $runtime)) { throw "GPT-SoVITS runtime was not found: $runtime" }
if ($InstallPyInstaller) { & $runtime -m pip install pyinstaller }
& $runtime -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller is not installed. Run .\scripts\build_launcher.ps1 -InstallPyInstaller' }
Push-Location $root
try {
    & $runtime -m PyInstaller --noconfirm --clean --onefile --name VoiceExtractor (Join-Path $root 'launcher\voice_extractor_launcher.py')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
} finally {
    Pop-Location
}
Write-Host ("Built {0}" -f (Join-Path $root 'dist\VoiceExtractor.exe')) -ForegroundColor Green
