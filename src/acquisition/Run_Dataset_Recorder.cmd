@echo off
setlocal
cd /d "%~dp0..\.."
if not exist ".venv\live-webcam\Scripts\python.exe" (
  echo Run Setup_Windows.cmd first.
  pause
  exit /b 1
)
set "MPLBACKEND=Agg"
if not defined MPLCONFIGDIR set "MPLCONFIGDIR=%CD%\.venv\live-webcam\matplotlib"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
".venv\live-webcam\Scripts\python.exe" -u src\acquisition\record_dataset.py %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
