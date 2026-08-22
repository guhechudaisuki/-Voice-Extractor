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
echo  Starting local service...
echo  Logs will appear in this window.
echo  Press Ctrl+C to stop.
echo.
"..\GPT-SoVITS-v2pro-20250604\runtime\python.exe" -u app.py
echo.
echo  Service stopped. Exit code: %ERRORLEVEL%
pause
