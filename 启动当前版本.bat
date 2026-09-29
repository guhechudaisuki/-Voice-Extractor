@echo off
setlocal
title Voice Extractor - Current Repository
cd /d "%~dp0"

for %%I in ("%~dp0.") do set "VOICE_EXTRACT_INSTALL_ROOT=%%~fI"
for %%I in ("%~dp0..\GPT-SoVITS-v2pro-20250604\runtime\python.exe") do set "VOICE_EXTRACT_PYTHON=%%~fI"

if exist "%~dp0dist\VoiceExtractor.exe" (
    start "" "%~dp0dist\VoiceExtractor.exe"
    exit /b 0
)

if exist "%VOICE_EXTRACT_PYTHON%" (
    start "" "%VOICE_EXTRACT_PYTHON%" "%~dp0launcher\voice_extractor_desktop.py"
    exit /b 0
)

echo Current launcher could not find the built application or Python runtime.
echo Build with scripts\build_launcher.ps1 or check the GPT-SoVITS runtime path.
pause
exit /b 1
