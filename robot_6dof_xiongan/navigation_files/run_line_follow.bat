@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "XIONGAN_PYTHON=C:\Users\Administrator\.virtualenvs\xiongan\Scripts\python.exe"
set "SUPPORT_DIR=%~dp0"
set "PROJECT_ROOT=%~dp0.."

if not exist "%XIONGAN_PYTHON%" (
    echo xiongan Python was not found:
    echo   %XIONGAN_PYTHON%
    pause
    exit /b 1
)

cd /d "%SUPPORT_DIR%"
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\navigation_guide;%PYTHONPATH%"
"%XIONGAN_PYTHON%" "%PROJECT_ROOT%\navigation_guide\run_line_follow.py"
pause
