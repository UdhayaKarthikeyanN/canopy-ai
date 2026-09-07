@echo off
title Stop Canopy AI
echo Stopping Canopy AI server on port 7860...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>nul
)
taskkill /FI "WINDOWTITLE eq Canopy AI Server*" /F >nul 2>nul
echo Done.
ping -n 3 127.0.0.1 >nul
