@echo off
chcp 65001 >nul
setlocal
title Voice Extractor
cd /d "%~dp0"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
set "HF_HUB_OFFLINE=1"
set "MODELSCOPE_CACHE=%CD%\models\modelscope_cache"
set "PYTHONUNBUFFERED=1"
echo Voice Extractor is starting.
echo Logs are shown in this window. Press Ctrl+C or close this window to stop.
echo.
"..\GPT-SoVITS-v2pro-20250604\runtime\python.exe" -u app.py
echo.
echo Service stopped. Exit code: %ERRORLEVEL%
pause
