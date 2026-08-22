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
        [string]$Destination
    )
    Write-Host ('DOWNLOAD {0} -> {1}' -f $Url, $Destination)
    if ($WhatIf) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        Write-Host ('exists: {0}' -f $Destination)
        return
    }
    Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
}

Write-Host 'This script does not install CUDA or PyTorch.' -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $gptRoot)) {
    Write-Warning ('GPT-SoVITS runtime not found: {0}' -f $gptRoot)
}

Fetch-File 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth?download=true' (Join-Path $gptRoot 'tools\uvr5\uvr5_weights\HP2_all_vocals.pth')
Fetch-File 'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt?download=true' (Join-Path $gptRoot 'GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt')
Fetch-File 'https://zenodo.org/records/3987831/files/Cnn10_mAP%3D0.380.pth?download=1' (Join-Path $toolRoot 'models\panns\Cnn10_mAP=0.380.pth')
Fetch-File 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv' (Join-Path $toolRoot 'models\panns\class_labels_indices.csv')

$hf = Get-Command hf -ErrorAction SilentlyContinue
if (-not $hf) { $hf = Get-Command huggingface-cli -ErrorAction SilentlyContinue }
if ($hf) {
    if ($WhatIf) {
        Write-Host 'hf download openai/whisper-large-v3-turbo --cache-dir WORKSPACE\omnvoice\hf_cache'
    } else {
        & $hf.Source download openai/whisper-large-v3-turbo --cache-dir (Join-Path $WorkspaceRoot 'omnvoice\hf_cache')
    }
} else {
    Write-Warning 'hf/huggingface-cli not found; download Whisper manually.'
}

$modelscope = Get-Command modelscope -ErrorAction SilentlyContinue
if ($modelscope) {
    $asrRoot = Join-Path $gptRoot 'tools\asr\models'
    $funAsr = @(
        @('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'),
        @('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', 'speech_fsmn_vad_zh-cn-16k-common-pytorch'),
        @('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
    )
    foreach ($item in $funAsr) {
        $destination = Join-Path $asrRoot $item[1]
        if ($WhatIf) {
            Write-Host ('modelscope download --model {0} --local_dir {1}' -f $item[0], $destination)
        } elseif (-not (Test-Path -LiteralPath (Join-Path $destination 'model.pt'))) {
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            & $modelscope.Source download --model $item[0] --local_dir $destination
        }
    }
} else {
    Write-Warning 'modelscope CLI not found; download FunASR models manually.'
}

Write-Host 'CAM++, overlap and any missing model files must be supplied separately.' -ForegroundColor Yellow
Write-Host 'After downloading, run: .\scripts\check_assets.ps1'
