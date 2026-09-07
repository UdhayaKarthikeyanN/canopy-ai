@echo off
title Canopy AI - Launcher
cd /d "%~dp0"

echo ============================================
echo   Canopy AI - starting local server...
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a local virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create a virtual environment.
        pause
        exit /b 1
    )
)
set PY=.venv\Scripts\python.exe

REM A project-local venv keeps Canopy AI's dependencies isolated from whatever
REM else is installed on this machine's system Python.
"%PY%" -c "import flask, cv2, numpy, reportlab, sklearn, joblib" >nul 2>nul
if errorlevel 1 (
    echo First run - installing dependencies into .venv, this can take a few minutes...
    "%PY%" -m pip install --quiet --upgrade pip
    "%PY%" -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection.
        pause
        exit /b 1
    )
)

echo Starting server in a new window ("Canopy AI Server")...
start "Canopy AI Server" cmd /k "%PY%" app.py

where curl >nul 2>nul
if errorlevel 1 (
    echo Waiting for the server to start...
    ping -n 9 127.0.0.1 >nul
    goto open
)

echo Waiting for the server to come online...
set TRIES=0
:waitloop
set /a TRIES+=1
curl -s -o nul --max-time 1 http://127.0.0.1:7860/
if not errorlevel 1 goto open
if %TRIES% GEQ 40 (
    echo [WARN] Server did not respond in time - opening the browser anyway.
    goto open
)
ping -n 2 127.0.0.1 >nul
goto waitloop

:open
echo Opening http://127.0.0.1:7860 in your browser...
start "" http://127.0.0.1:7860

echo.
echo Canopy AI is running in the "Canopy AI Server" window.
echo To stop it: run stop_canopy.bat, or close that window.
echo This launcher window can be closed safely - it does not run the server.
pause
