# 打包发布：代码 + 模型权重 + vendor → 自包含 ZIP
# 用法: .\scripts\package_release.ps1 [-OutputDir <目录>] [-IncludeGPTRuntime]
# 产出: VoiceExtractor_v3.0_release.zip（约 5.8 GB，不含 GPT-SoVITS 运行时）
#       加 -IncludeGPTRuntime 则约 30 GB（含完整运行时，一般不需要——对方自己装 GPT-SoVITS 即可）
param(
    [string]$OutputDir = "$PSScriptRoot\..\dist",
    [switch]$IncludeGPTRuntime
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot | Split-Path
$Version = "3.0"
$Staging = Join-Path $OutputDir "VoiceExtractor_v${Version}"
$ZipPath = Join-Path $OutputDir "VoiceExtractor_v${Version}_release.zip"

if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item -ItemType Directory -Path $Staging -Force | Out-Null

Write-Host "=== 1/4 复制程序代码（git archive 保证与仓库一致）==="
git archive --format=tar HEAD | tar -x -C $Staging

Write-Host "=== 2/4 复制模型权重（4.8G + 993M）==="
Copy-Item "$Root\model" "$Staging\model" -Recurse -Force
Copy-Item "$Root\models" "$Staging\models" -Recurse -Force

Write-Host "=== 3/4 复制 vendor（818K）==="
Copy-Item "$Root\vendor" "$Staging\vendor" -Recurse -Force

if ($IncludeGPTRuntime) {
    Write-Host "=== 3.5/4 复制 GPT-SoVITS 运行时（24G，慎用）==="
    $GptRoot = $Root | Split-Path | Join-Path -ChildPath "GPT-SoVITS-v2pro-20250604"
    if (Test-Path $GptRoot) {
        Copy-Item $GptRoot "$Staging\GPT-SoVITS-v2pro-20250604" -Recurse -Force
    }
}

Write-Host "=== 4/4 打包 ZIP ==="
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path $Staging -DestinationPath $ZipPath -CompressionLevel Optimal
Remove-Item $Staging -Recurse -Force

$Size = (Get-Item $ZipPath).Length / 1GB
Write-Host "完成: $ZipPath ($([math]::Round($Size, 2)) GB)"
Write-Host ""
Write-Host "=== 使用方法 ==="
Write-Host "1. 解压 ZIP 到任意目录"
Write-Host "2. 安装 GPT-SoVITS-v2pro-20250604 到同级目录（如未包含）"
Write-Host "3. 双击 启动Desktop.bat 或 启动WebUI.bat"
Write-Host ""
Write-Host "模型文件与本地完全一致 → 提取效果 100% 相同"
