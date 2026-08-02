@echo off
set "XIONGAN_ENV=C:\Users\Administrator\.virtualenvs\xiongan"
set "PROJECT_ROOT=%~dp0.."

if not exist "%XIONGAN_ENV%\Scripts\activate.bat" (
    echo xiongan environment was not found:
    echo   %XIONGAN_ENV%
    exit /b 1
)

call "%XIONGAN_ENV%\Scripts\activate.bat"
cd /d "%PROJECT_ROOT%"

echo.
echo xiongan environment activated.
echo Project: %CD%
echo Python:  %XIONGAN_ENV%\Scripts\python.exe
echo.
