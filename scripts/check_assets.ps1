param(
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')),
    [string]$GptRoot = $env:VOICE_EXTRACT_GPT_ROOT
)

$ErrorActionPreference = 'Stop'
$toolRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $GptRoot) { $GptRoot = Join-Path $WorkspaceRoot 'GPT-SoVITS' }
$modelRoot = Join-Path $toolRoot 'model'
$python = if ($env:VOICE_EXTRACT_PYTHON) { $env:VOICE_EXTRACT_PYTHON } else { Join-Path $gptRoot 'runtime\python.exe' }

function Test-Asset {
    param([hashtable]$Asset)
    if (-not (Test-Path -LiteralPath $Asset.Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Asset.Path).Length -eq 0) { return $false }
    if ($Asset.Hash) {
        return (Get-FileHash -LiteralPath $Asset.Path -Algorithm SHA256).Hash -eq $Asset.Hash
    }
    return $true
}

$checks = @(
    @{ Name='Python runtime'; Path=$python },
    @{ Name='FFmpeg'; Path=(Join-Path $gptRoot 'runtime\ffmpeg.exe') },
    @{ Name='FFprobe'; Path=(Join-Path $gptRoot 'runtime\ffprobe.exe') },
    @{ Name='UVR5 vocals'; Path=(Join-Path $gptRoot 'tools\uvr5\uvr5_weights\HP2_all_vocals.pth') },
    @{ Name='ERes2NetV2'; Path=(Join-Path $modelRoot 'speaker\eres2net\pretrained_eres2netv2w24s4ep4.ckpt') },
    @{ Name='Paraformer'; Path=(Join-Path $modelRoot 'stt\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch\model.pt') },
    @{ Name='FSMN-VAD'; Path=(Join-Path $modelRoot 'stt\speech_fsmn_vad_zh-cn-16k-common-pytorch\model.pt') },
    @{ Name='CT-Punc'; Path=(Join-Path $modelRoot 'stt\punc_ct-transformer_zh-cn-common-vocab272727-pytorch\model.pt') },
    @{ Name='CAM++'; Path=(Join-Path $modelRoot 'speaker\campplus_voxceleb\campplus_voxceleb.bin') },
    @{ Name='WavLM SV'; Path=(Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\pytorch_model.bin'); Hash='0fde81fc23bdf3944f9167156fab1852025159d4414aa98bb22e58071bb39586' },
    @{ Name='Identity classifier'; Path=(Join-Path $toolRoot 'models\anime_t3\classifier_v2.pt'); Hash='a43ed3b3d0285d61b72d52b102ab5b0aae388e5e3e2dc88a26c0f56a108747a4' },
    @{ Name='Anime character ONNX'; Path=(Join-Path $toolRoot 'models\anime-speaker-char\anime_speaker_char_ecapa.onnx') },
    @{ Name='Anime voice actor ONNX'; Path=(Join-Path $toolRoot 'models\anime-speaker-va\anime_speaker_va_ecapa.onnx') },
    @{ Name='WavLM config'; Path=(Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\config.json') },
    @{ Name='WavLM preprocessor'; Path=(Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\preprocessor_config.json') },
    @{
        Name='WeSpeaker ONNX'
        Path=(Join-Path $modelRoot 'speaker\wespeaker-resnet34-lm\onnx\model.onnx')
        Hash='3955447B0499DC9E0A4541A895DF08B03C69098EBA4E56C02B5603E9F7F4FCBB'
    },
    @{ Name='Overlap ONNX'; Path=(Join-Path $toolRoot 'models\overlap\model.onnx') },
    @{ Name='PANNs'; Path=(Join-Path $toolRoot 'models\panns\Cnn10_mAP=0.380.pth') },
    @{ Name='PANNs labels'; Path=(Join-Path $toolRoot 'models\panns\class_labels_indices.csv') }
)
$missing = @($checks | Where-Object { -not (Test-Asset $_) })
foreach ($item in $checks) {
    if (Test-Asset $item) { Write-Host ("[OK]   {0}: {1}" -f $item.Name,$item.Path) -ForegroundColor Green }
    else { Write-Host ("[MISS] {0}: {1}" -f $item.Name,$item.Path) -ForegroundColor Yellow }
}
if ($missing.Count -gt 0) { Write-Error ('Missing {0} asset(s). Run download_assets.ps1 or follow README.' -f $missing.Count) }

$whisperSnapshots = Join-Path $modelRoot 'stt\whisper\hf_cache\models--openai--whisper-large-v3-turbo\snapshots'
$whisperReady = (Test-Path -LiteralPath $whisperSnapshots -PathType Container) -and
    (@(Get-ChildItem -LiteralPath $whisperSnapshots -Directory -ErrorAction SilentlyContinue).Count -gt 0)
if ($whisperReady) {
    Write-Host ("[OK]   Whisper large-v3-turbo: {0}" -f $whisperSnapshots) -ForegroundColor Green
} else {
    Write-Error ("Missing Whisper large-v3-turbo snapshot: {0}" -f $whisperSnapshots)
}
Write-Host 'Asset check passed.' -ForegroundColor Green
