@echo off
setlocal
cd /d "%~dp0..\.."
py -3.12 -c "import tkinter" || goto failed
py -3.12 -m venv .venv\live-blink || goto failed
".venv\live-blink\Scripts\python.exe" -m pip install -r src\live_blink\requirements.txt || goto failed
".venv\live-blink\Scripts\python.exe" -m pip install --no-deps bitalino==1.2.6 || goto failed
echo Ready. Open Run_Live_Blinks.cmd.
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.12 with Python Launcher and Tcl/Tk, then retry.
pause
exit /b 1
