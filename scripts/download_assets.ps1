param(
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')),
    [string]$GptRoot = $env:VOICE_EXTRACT_GPT_ROOT,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$toolRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $GptRoot) { $GptRoot = Join-Path $WorkspaceRoot 'GPT-SoVITS' }
$modelRoot = Join-Path $toolRoot 'model'

function Fetch-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$ExpectedSha256 = '',
        [switch]$PreserveExisting
    )

    $usable = Test-Path -LiteralPath $Destination -PathType Leaf
    if ($usable) {
        $usable = (Get-Item -LiteralPath $Destination).Length -gt 0
    }
    if ($usable -and $ExpectedSha256) {
        $usable = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $ExpectedSha256
        if (-not $usable -and $PreserveExisting) {
            throw "Existing trained model has a different checksum; it was preserved: $Destination"
        }
    }
    if ($usable) {
        Write-Host ('[OK] {0}' -f $Destination) -ForegroundColor Green
        return
    }

    Write-Host ('DOWNLOAD {0} -> {1}' -f $Url, $Destination)
    if ($WhatIf) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $partial = $Destination + '.part'
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    try {
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
        if ((Get-Item -LiteralPath $partial).Length -eq 0) {
            throw "Downloaded an empty file: $Url"
        }
        if ($ExpectedSha256 -and (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $ExpectedSha256) {
            throw "Checksum mismatch: $Destination"
        }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
    } finally {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    }
}

Write-Host 'This script does not install CUDA or PyTorch.' -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $gptRoot)) {
    Write-Warning ('GPT-SoVITS runtime not found: {0}' -f $gptRoot)
}

Fetch-File 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth?download=true' (Join-Path $gptRoot 'tools\uvr5\uvr5_weights\HP2_all_vocals.pth')
Fetch-File 'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt?download=true' (Join-Path $modelRoot 'speaker\eres2net\pretrained_eres2netv2w24s4ep4.ckpt')
Fetch-File 'https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/resolve/master/campplus_cn_common.bin' (Join-Path $modelRoot 'speaker\campplus_voxceleb\campplus_voxceleb.bin')
Fetch-File 'https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/config.json?download=true' (Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\config.json')
Fetch-File 'https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/preprocessor_config.json?download=true' (Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\preprocessor_config.json')
Fetch-File 'https://github.com/guhechudaisuki/-Voice-Extractor/releases/download/v3.0/wavlm_reinforced_pytorch_model.bin' (Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\pytorch_model.bin') 'e906bce2fa42fb497a1d1a9ecf81548adb7e03b12a5644e32d2f42f0d6500fad' -PreserveExisting
Fetch-File 'https://github.com/guhechudaisuki/-Voice-Extractor/raw/main/models/anime_t3/classifier_v2.pt' (Join-Path $toolRoot 'models\anime_t3\classifier_v2.pt') 'a86a47947dddf8f8d638e2764bee79c086b19910d4673b53f371a9b821e4401c' -PreserveExisting
Fetch-File 'https://github.com/guhechudaisuki/-Voice-Extractor/releases/download/model-assets/anime_speaker_char_ecapa.onnx' (Join-Path $toolRoot 'models\anime-speaker-char\anime_speaker_char_ecapa.onnx') 'd4e0ce7d55501fbb1a71d18ae7d3146d6fbd2d5aee8369d3171a82418f44d3a8'
Fetch-File 'https://github.com/guhechudaisuki/-Voice-Extractor/releases/download/model-assets/anime_speaker_va_ecapa.onnx' (Join-Path $toolRoot 'models\anime-speaker-va\anime_speaker_va_ecapa.onnx') '9d2d285e55cb5eb4ba09fe703119cfec4ae715ba3ba4df89a97cfec606e3d986'

$prototypeSource = Join-Path $toolRoot 'models\wavlm-base-plus-sv\auto_negatives.npz'
$prototypeTarget = Join-Path $modelRoot 'speaker\wavlm-base-plus-sv\auto_negatives.npz'
if ((Test-Path -LiteralPath $prototypeSource -PathType Leaf) -and
    -not (Test-Path -LiteralPath $prototypeTarget)) {
    if ($WhatIf) {
        Write-Host ('INSTALL {0} -> {1}' -f $prototypeSource, $prototypeTarget)
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $prototypeTarget) | Out-Null
        Copy-Item -LiteralPath $prototypeSource -Destination $prototypeTarget
    }
}

Fetch-File 'https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx?download=true' (Join-Path $modelRoot 'speaker\wespeaker-resnet34-lm\onnx\model.onnx') '3955447B0499DC9E0A4541A895DF08B03C69098EBA4E56C02B5603E9F7F4FCBB'
Fetch-File 'https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx?download=true' (Join-Path $toolRoot 'models\overlap\model.onnx') '220AD67CA923BEF2FA91F2390C786097BF305BCEB5E261D4AF67B38E938E1079'
Fetch-File 'https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1' (Join-Path $toolRoot 'models\panns\Cnn10_mAP=0.380.pth')
Fetch-File 'https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv' (Join-Path $toolRoot 'models\panns\class_labels_indices.csv')

$whisperCache = Join-Path $modelRoot 'stt\whisper\hf_cache'
$whisperSnapshots = Join-Path $whisperCache 'models--openai--whisper-large-v3-turbo\snapshots'
$whisperReady = (Test-Path -LiteralPath $whisperSnapshots -PathType Container) -and
    (@(Get-ChildItem -LiteralPath $whisperSnapshots -Directory -ErrorAction SilentlyContinue).Count -gt 0)
if ($whisperReady) {
    Write-Host ('[OK] Whisper large-v3-turbo: {0}' -f $whisperSnapshots) -ForegroundColor Green
} else {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if (-not $hf) { $hf = Get-Command huggingface-cli -ErrorAction SilentlyContinue }
    if ($hf) {
        if ($WhatIf) {
            Write-Host ('DOWNLOAD openai/whisper-large-v3-turbo -> {0}' -f $whisperCache)
        } else {
            & $hf.Source download openai/whisper-large-v3-turbo --cache-dir $whisperCache
            if ($LASTEXITCODE -ne 0) { throw 'Whisper download failed.' }
        }
    } else {
        Write-Warning 'hf/huggingface-cli not found; download Whisper manually.'
    }
}

$asrRoot = Join-Path $modelRoot 'stt'
$funAsr = @(
    @('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'),
    @('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', 'speech_fsmn_vad_zh-cn-16k-common-pytorch'),
    @('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
)
$missingFunAsr = @()
foreach ($item in $funAsr) {
    $destination = Join-Path $asrRoot $item[1]
    $modelFile = Join-Path $destination 'model.pt'
    if ((Test-Path -LiteralPath $modelFile -PathType Leaf) -and (Get-Item -LiteralPath $modelFile).Length -gt 0) {
        Write-Host ('[OK] FunASR: {0}' -f $destination) -ForegroundColor Green
    } else {
        $missingFunAsr += ,$item
    }
}

if ($missingFunAsr.Count -gt 0) {
    $modelscope = Get-Command modelscope -ErrorAction SilentlyContinue
    foreach ($item in $missingFunAsr) {
        $destination = Join-Path $asrRoot $item[1]
        if ($WhatIf) {
            Write-Host ('DOWNLOAD {0} -> {1}' -f $item[0], $destination)
        } elseif ($modelscope) {
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            & $modelscope.Source download --model $item[0] --local_dir $destination
            if ($LASTEXITCODE -ne 0) { throw "ModelScope download failed: $($item[0])" }
        } else {
            Write-Warning ('modelscope CLI not found; cannot download {0}.' -f $item[0])
        }
    }
}

if ($WhatIf) {
    Write-Host 'WhatIf complete. No files were downloaded.'
} else {
    & (Join-Path $PSScriptRoot 'check_assets.ps1') -WorkspaceRoot $WorkspaceRoot -GptRoot $GptRoot
}
