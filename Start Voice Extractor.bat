@echo off
setlocal
title Voice Extractor
cd /d "%~dp0"
if exist "dist\VoiceExtractor.exe" (
    start "" "dist\VoiceExtractor.exe"
    exit /b 0
)
if exist "VoiceExtractor.exe" (
    start "" "VoiceExtractor.exe"
    exit /b 0
)
echo VoiceExtractor.exe was not found. Build it with scripts\build_launcher.ps1 first.
pause
exit /b 1
