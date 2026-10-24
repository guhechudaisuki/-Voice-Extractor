@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "VOICE_EXTRACT_INSTALL_ROOT=%%~fI"
for %%I in ("%~dp0..\GPT-SoVITS-v2pro-20250604\runtime\python.exe") do set "VOICE_EXTRACT_PYTHON=%%~fI"
if not exist "%VOICE_EXTRACT_PYTHON%" (
    echo Python runtime was not found.
    pause
    exit /b 1
)
"%VOICE_EXTRACT_PYTHON%" -B "%~dp0launcher\app_entry.py" --ui desktop
set "VOICE_EXTRACT_EXIT_CODE=%ERRORLEVEL%"
if not "%VOICE_EXTRACT_EXIT_CODE%"=="0" pause
exit /b %VOICE_EXTRACT_EXIT_CODE%
