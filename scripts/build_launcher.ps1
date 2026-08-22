param([switch]$InstallPyInstaller)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = Join-Path $root '..\GPT-SoVITS-v2pro-20250604\runtime\python.exe'
if (-not (Test-Path $runtime)) { throw "找不到 GPT-SoVITS runtime: $runtime" }
if ($InstallPyInstaller) { & $runtime -m pip install pyinstaller }
& $runtime -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) { throw '未安装 PyInstaller。运行 .\scripts\build_launcher.ps1 -InstallPyInstaller' }
& $runtime -m PyInstaller --noconfirm --clean --onefile --name VoiceExtractor (Join-Path $root 'launcher\voice_extractor_launcher.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller 构建失败' }
Write-Host "已生成 dist\VoiceExtractor.exe" -ForegroundColor Green
