@echo off
chcp 65001 >nul
setlocal
title Voice Extractor
cd /d "%~dp0"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
set "HF_HUB_OFFLINE=1"
set "MODELSCOPE_CACHE=%CD%\models\modelscope_cache"
set "PYTHONUNBUFFERED=1"

echo.
echo  Voice Extractor
echo  ========================================
echo  Checking dependencies and local models...
echo  Missing resources will be downloaded automatically.
echo  CUDA and PyTorch are kept local.
echo.
if not exist "dist\VoiceExtractor.exe" (
    echo  ERROR: dist\VoiceExtractor.exe was not found.
    echo  Build it with scripts\build_launcher.ps1 first.
    echo.
    pause
    exit /b 1
)
"dist\VoiceExtractor.exe"
echo.
echo  Service stopped. Exit code: %ERRORLEVEL%
pause
