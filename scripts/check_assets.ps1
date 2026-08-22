param(
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
)

$ErrorActionPreference = 'Stop'
$toolRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gptRoot = Join-Path $WorkspaceRoot 'GPT-SoVITS-v2pro-20250604'
$python = if ($env:VOICE_EXTRACT_PYTHON) { $env:VOICE_EXTRACT_PYTHON } else { Join-Path $gptRoot 'runtime\python.exe' }
$checks = @(
    @{ Name='Python runtime'; Path=$python },
    @{ Name='FFmpeg'; Path=(Join-Path $gptRoot 'runtime\ffmpeg.exe') },
    @{ Name='UVR5 vocals'; Path=(Join-Path $gptRoot 'tools\uvr5\uvr5_weights\HP2_all_vocals.pth') },
    @{ Name='ERes2NetV2'; Path=(Join-Path $gptRoot 'GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt') },
    @{ Name='Paraformer'; Path=(Join-Path $gptRoot 'tools\asr\models\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\model.pt') },
    @{ Name='FSMN-VAD'; Path=(Join-Path $gptRoot 'tools\asr\models\speech_fsmn_vad_zh-cn-16k-common-pytorch\model.pt') },
    @{ Name='CT-Punc'; Path=(Join-Path $gptRoot 'tools\asr\models\punc_ct-transformer_zh-cn-common-vocab272727-pytorch\model.pt') },
    @{ Name='CAM++'; Path=(Join-Path $toolRoot 'models\campplus_voxceleb\campplus_voxceleb.bin') },
    @{ Name='Overlap ONNX'; Path=(Join-Path $toolRoot 'models\overlap\model.onnx') },
    @{ Name='PANNs'; Path=(Join-Path $toolRoot 'models\panns\Cnn10_mAP=0.380.pth') },
    @{ Name='PANNs labels'; Path=(Join-Path $toolRoot 'models\panns\class_labels_indices.csv') }
)
$missing = @($checks | Where-Object { -not (Test-Path -LiteralPath $_.Path) })
foreach ($item in $checks) {
    if (Test-Path -LiteralPath $item.Path) { Write-Host ("[OK]   {0}: {1}" -f $item.Name,$item.Path) -ForegroundColor Green }
    else { Write-Host ("[MISS] {0}: {1}" -f $item.Name,$item.Path) -ForegroundColor Yellow }
}
if ($missing.Count -gt 0) { Write-Error ('Missing {0} asset(s). Run download_assets.ps1 or follow README.' -f $missing.Count) }
Write-Host 'Asset check passed.' -ForegroundColor Green
