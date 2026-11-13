param([string]$MakeNsis, [switch]$StageOnly)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dist = Join-Path $root 'dist'
if (-not $MakeNsis) {
    $command = Get-Command makensis -ErrorAction SilentlyContinue
    if ($command) { $MakeNsis = $command.Source }
}
if (-not $MakeNsis -or -not (Test-Path -LiteralPath $MakeNsis -PathType Leaf)) {
    throw 'Pass -MakeNsis with the path to the Unicode NSIS compiler.'
}
$MakeNsis = (Resolve-Path -LiteralPath $MakeNsis).Path
$checksums = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $dist 'SHA256SUMS.txt') -Encoding utf8) {
    if ($line -match '^([a-f0-9]{64})\s+(.+)$') { $checksums[$Matches[2]] = $Matches[1] }
}
foreach ($name in @('VoiceExtractor.exe', 'VoiceExtractor_resources.zip', 'wavlm_reinforced_pytorch_model.bin')) {
    $path = Join-Path $dist $name
    if (-not $checksums.ContainsKey($name) -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $checksums[$name]) {
        throw "Release checksum mismatch: $name. Run package_release.ps1 first."
    }
}

$stage = Join-Path $root ('build\nsis\' + [guid]::NewGuid().ToString('N'))
$payload = Join-Path $stage 'payload'
$resource = Join-Path $payload 'resource'
$null = New-Item -ItemType Directory -Path $resource -Force
Expand-Archive -LiteralPath (Join-Path $dist 'VoiceExtractor_resources.zip') -DestinationPath $resource
# The ZIP contains a blank self-checksum; use the manifest paired with the EXE.
Copy-Item -LiteralPath (Join-Path $root 'launcher\release_manifest.py') -Destination (Join-Path $resource 'launcher\release_manifest.py') -Force
$programFiles = @(Get-ChildItem -LiteralPath $resource -Recurse -File | ForEach-Object {
    $_.FullName.Substring($resource.Length + 1).Replace('\', '/')
})
@{ sha256=$checksums['VoiceExtractor_resources.zip']; files=$programFiles } |
    ConvertTo-Json -Depth 3 | ForEach-Object { [IO.File]::WriteAllText((Join-Path $resource 'program_manifest.json'), $_, [Text.UTF8Encoding]::new($false)) }
$setup = Join-Path $resource 'setup'
$null = New-Item -ItemType Directory -Path $setup -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'nsis_setup.py') -Destination $setup
$bootstrapArchive = Join-Path $root 'tools\nsis\python-3.9.13-embed-amd64.zip'
if (-not (Test-Path -LiteralPath $bootstrapArchive)) {
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip' -OutFile $bootstrapArchive
}
if ((Get-FileHash -LiteralPath $bootstrapArchive -Algorithm SHA256).Hash -ne '938a1f3b80d580320836260612084d74ce094a261e36f9ff3ac7b9463df5f5e4') {
    throw 'Embedded Python checksum mismatch.'
}
Expand-Archive -LiteralPath $bootstrapArchive -DestinationPath (Join-Path $stage 'bootstrap')
Copy-Item -LiteralPath (Join-Path $dist 'VoiceExtractor.exe') -Destination $payload
foreach ($name in @('README.md', 'LICENSE')) {
    Copy-Item -LiteralPath (Join-Path $root $name) -Destination $payload
}
$wavlm = Join-Path $resource 'model\speaker\wavlm-base-plus-sv'
$null = New-Item -ItemType Directory -Path $wavlm -Force
Copy-Item -LiteralPath (Join-Path $dist 'wavlm_reinforced_pytorch_model.bin') -Destination (Join-Path $wavlm 'pytorch_model.bin')
foreach ($name in @('config.json', 'preprocessor_config.json')) {
    Copy-Item -LiteralPath (Join-Path $root "model\speaker\wavlm-base-plus-sv\$name") -Destination $wavlm
}

function ConvertTo-NsisLiteral([string]$Value) {
    return $Value.Replace('$', '$$').Replace('"', '$\"')
}
$installLines = [System.Collections.Generic.List[string]]::new()
$uninstallLines = [System.Collections.Generic.List[string]]::new()
$directories = [System.Collections.Generic.HashSet[string]]::new()
$files = @(Get-ChildItem -LiteralPath $payload -Recurse -File | Sort-Object FullName)
$lastDirectory = $null
$installedBytes = 0L
foreach ($file in $files) {
    $relative = $file.FullName.Substring($payload.Length + 1)
    $directory = [IO.Path]::GetDirectoryName($relative)
    if ($directory -ne $lastDirectory) {
        $installLines.Add('SetOutPath "$INSTDIR\' + (ConvertTo-NsisLiteral $directory) + '"')
        $lastDirectory = $directory
    }
    $modelFile = $relative.StartsWith('resource\model\') -or $relative.StartsWith('resource\models\')
    if ($modelFile) { $installLines.Add('SetOverwrite off') }
    $storeDirectly = $relative -eq 'VoiceExtractor.exe' -or $file.Extension -in @('.bin', '.onnx')
    if ($storeDirectly) { $installLines.Add('SetCompress off') }
    $installLines.Add('File "' + (ConvertTo-NsisLiteral $file.FullName) + '"')
    if ($storeDirectly) { $installLines.Add('SetCompress auto') }
    if ($modelFile) {
        $installLines.Add('SetOverwrite on')
    } else {
        $uninstallLines.Add('Delete "$INSTDIR\' + (ConvertTo-NsisLiteral $relative) + '"')
    }
    while ($directory) {
        $null = $directories.Add($directory)
        $directory = [IO.Path]::GetDirectoryName($directory)
    }
    $installedBytes += $file.Length
}
foreach ($directory in ($directories | Sort-Object Length -Descending)) {
    $uninstallLines.Add('RMDir "$INSTDIR\' + (ConvertTo-NsisLiteral $directory) + '"')
}
$installLines | Set-Content -LiteralPath (Join-Path $stage 'install-files.nsh') -Encoding utf8
$uninstallLines | Set-Content -LiteralPath (Join-Path $stage 'uninstall-files.nsh') -Encoding utf8
if ($StageOnly) {
    [PSCustomObject]@{Stage=$stage;Payload=$payload;InstalledSizeKB=[math]::Ceiling($installedBytes / 1KB)}
    return
}
$output = Join-Path $dist 'Voice-Extractor1.0-Full-Setup.exe'
& $MakeNsis /V3 /INPUTCHARSET UTF8 "/DPROJECT_ROOT=$root" "/DSTAGE_DIR=$stage" "/DOUTPUT_FILE=$output" "/DINSTALLED_SIZE_KB=$([math]::Ceiling($installedBytes / 1KB))" (Join-Path $PSScriptRoot 'voice_extractor.nsi')
if ($LASTEXITCODE -ne 0) { throw 'NSIS compilation failed.' }
$hash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($output))" | Set-Content -LiteralPath "$output.sha256" -Encoding utf8
Write-Output "Built NSIS installer: $output"
Write-Output "SHA256: $hash"
