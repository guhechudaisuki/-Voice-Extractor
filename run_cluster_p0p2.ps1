# P0–P2 一键运行：单元测试 → 导出嵌入 → 出聚类对照报告
#
# 用法（在 PowerShell 里、项目根目录下）：
#     powershell -ExecutionPolicy Bypass -File .\run_cluster_p0p2.ps1
#
# 三件事都不改任何现有产物：单测只读内存、导出写 work/embeddings/、
# 报告写 docs/。不会重跑 pipeline，不会动 output/。

param([switch]$RefreshEmbeddings)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

$Root   = $PSScriptRoot
$LogDir = Join-Path $Root 'work\cluster_p0p2_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$BatchId   = '20260928_214330_batch_5bd1f3'
$BatchDir  = Join-Path $Root "output\$BatchId"
$EmbedFile = Join-Path $Root 'work\embeddings\5bd1f3.npz'
$ReportFile= Join-Path $Root 'docs\cluster_vs_review_ep3_p2_verified.json'

# ── 1. 找 Python 运行时 ────────────────────────────────────────
function Find-Python {
    $candidates = @()

    # 项目自带的运行时（与启动当前版本.bat 用的是同一套）
    $candidates += Join-Path (Split-Path $Root -Parent) 'GPT-SoVITS-v2pro-20250604\runtime\python.exe'
    $candidates += Join-Path $Root 'runtime\python.exe'

    foreach ($envName in @('VOICE_EXTRACT_PYTHON', 'PYTHONHOME', 'PYTHON_ROOT')) {
        $value = [Environment]::GetEnvironmentVariable($envName)
        if ($value) {
            $candidates += if (Test-Path $value -PathType Container) {
                Join-Path $value 'python.exe'
            } else { $value }
        }
    }

    foreach ($name in @('python.exe', 'python')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $candidates += $cmd.Source }
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate -PathType Leaf)) { return $candidate }
    }
    throw '找不到 python.exe。请设置环境变量 VOICE_EXTRACT_PYTHON 指向它。'
}

$Python = Find-Python
Write-Host "Python: $Python" -ForegroundColor Cyan
Write-Host ""

function Invoke-Step {
    param([string]$Title, [string[]]$Arguments, [string]$LogName)

    $log = Join-Path $LogDir $LogName
    Write-Host "── $Title " -ForegroundColor Yellow -NoNewline
    Write-Host ('─' * [Math]::Max(0, 60 - $Title.Length)) -ForegroundColor DarkGray

    Push-Location $Root
    $previousPreference = $ErrorActionPreference
    try {
        # unittest writes its successful summary to stderr; Windows PowerShell
        # must use the process exit code rather than treating that stream as failure.
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message }
            else { $_ }
        } | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
        Pop-Location
    }

    if ($code -ne 0) {
        Write-Host ""
        Write-Host "这一步失败了（退出码 $code）。日志：$log" -ForegroundColor Red
        Write-Host "后面的步骤依赖它，已停止。" -ForegroundColor Red
        exit $code
    }
    Write-Host ""
}

# ── 2. 单测 ────────────────────────────────────────────────────
Invoke-Step 'P0 单元测试' `
    @('-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test*cluster*.py', '-v') `
    '01_unittest.log'

# ── 3. 导出第三集嵌入 ──────────────────────────────────────────
$excludeDir = Join-Path $Root '京吹原片音频\其他人物'
$excludeArgs = @()
foreach ($file in Get-ChildItem -LiteralPath $excludeDir -Filter *.wav | Sort-Object Name) {
    $excludeArgs += '--exclude-role'
    $excludeArgs += $file.FullName
}

$exportArgs = @(
    '-B', 'evaluation/export_episode_embeddings.py',
    '--batch', $BatchDir,
    '--reference', (Join-Path $Root '京吹原片音频\kumiko'),
    '--output', $EmbedFile
) + $excludeArgs

# Always refresh turn metadata from stage_audit. The content-addressed cache
# reuses vectors only when audio, model and preprocessing fingerprints match.
if ($RefreshEmbeddings) { $exportArgs += '--no-cache' }
Invoke-Step 'P1 校验输入指纹并导出说话轮次嵌入' $exportArgs '02_export.log'

# ── 4. 聚类对照报告 ────────────────────────────────────────────
$reportArgs = @(
    '-B', 'evaluation/cluster_report.py',
    '--embeddings', $EmbedFile,
    '--reference', (Join-Path $Root '京吹原片音频\kumiko'),
    '--no-dominant-check',
    '--review', (Join-Path $Root 'evaluation\episode3_user_review_20260928_5bd1f3.json'),
    '--output', $ReportFile
)
foreach ($file in Get-ChildItem -LiteralPath $excludeDir -Filter *.wav | Sort-Object Name) {
    $reportArgs += '--exclude-role'
    $reportArgs += $file.FullName
}

Invoke-Step 'P2 聚类对照报告' $reportArgs '03_cluster_report.log'

Write-Host "P0-P2 检查执行完成；报告结论决定是否能继续接入。" -ForegroundColor Green
Write-Host "报告: $ReportFile" -ForegroundColor Green
Write-Host "日志: $LogDir" -ForegroundColor Green
Write-Host ""
$report = Get-Content -LiteralPath $ReportFile -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "P2: $($report.review_check.verdict)" -ForegroundColor Cyan
if ($report.review_check.verdict -ne 'review_errors_blocked') {
    Write-Host 'P2 尚未通过：保留诊断结果，暂不进入正式接入。' -ForegroundColor Yellow
    exit 2
}
