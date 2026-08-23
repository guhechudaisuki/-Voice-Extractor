param(
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')),
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$toolRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$gptRoot = Join-Path $WorkspaceRoot 'GPT-SoVITS-v2pro-20250604'

function Fetch-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$ExpectedSha256 = ''
    )

    $usable = Test-Path -LiteralPath $Destination -PathType Leaf
    if ($usable) {
        $usable = (Get-Item -LiteralPath $Destination).Length -gt 0
    }
    if ($usable -and $ExpectedSha256) {
        $usable = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $ExpectedSha256
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
Fetch-File 'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt?download=true' (Join-Path $gptRoot 'GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt')
Fetch-File 'https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/resolve/master/campplus_cn_common.bin' (Join-Path $toolRoot 'models\campplus_voxceleb\campplus_voxceleb.bin')
Fetch-File 'https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/config.json?download=true' (Join-Path $toolRoot 'models\wavlm-base-plus-sv\config.json')
Fetch-File 'https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/preprocessor_config.json?download=true' (Join-Path $toolRoot 'models\wavlm-base-plus-sv\preprocessor_config.json')
Fetch-File 'https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/pytorch_model.bin?download=true' (Join-Path $toolRoot 'models\wavlm-base-plus-sv\pytorch_model.bin')
Fetch-File 'https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/6a61a1833ff2583aabeba044f5c8221f00b67ceb/onnx/model.onnx?download=true' (Join-Path $toolRoot 'models\wespeaker-resnet34-lm\onnx\model.onnx') '3955447B0499DC9E0A4541A895DF08B03C69098EBA4E56C02B5603E9F7F4FCBB'
Fetch-File 'https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx?download=true' (Join-Path $toolRoot 'models\overlap\model.onnx') '220AD67CA923BEF2FA91F2390C786097BF305BCEB5E261D4AF67B38E938E1079'
Fetch-File 'https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1' (Join-Path $toolRoot 'models\panns\Cnn10_mAP=0.380.pth')
Fetch-File 'https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv' (Join-Path $toolRoot 'models\panns\class_labels_indices.csv')

$whisperCache = Join-Path $WorkspaceRoot 'omnvoice\hf_cache'
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

$asrRoot = Join-Path $gptRoot 'tools\asr\models'
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
    & (Join-Path $PSScriptRoot 'check_assets.ps1') -WorkspaceRoot $WorkspaceRoot
}
