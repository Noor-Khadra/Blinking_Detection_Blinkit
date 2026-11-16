@echo off
setlocal
cd /d "%~dp0..\.."
if not exist .venv\live-blink\Scripts\python.exe (
 echo Run src\live_blink\Setup_Windows.cmd first.
 pause
 exit /b 1
)
".venv\live-blink\Scripts\python.exe" src\live_blink\app.py
if errorlevel 1 pause
