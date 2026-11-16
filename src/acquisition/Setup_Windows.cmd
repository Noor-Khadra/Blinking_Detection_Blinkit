@echo off
setlocal
cd /d "%~dp0..\.."
py -3.12 -c "import tkinter" || goto python_missing
py -3.12 -m venv .venv\live-webcam || goto failed
".venv\live-webcam\Scripts\python.exe" -m pip install -r src\webcam\requirements.txt pyserial==3.5 pillow || goto failed
".venv\live-webcam\Scripts\python.exe" -m pip install --no-deps bitalino==1.2.6 || goto failed
if not exist face_landmarker.task (
  curl.exe -fL "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -o face_landmarker.task.download || goto failed
  move /y face_landmarker.task.download face_landmarker.task > nul || goto failed
)
echo Setup complete. Open Run_Dataset_Recorder.cmd.
pause
exit /b 0

:python_missing
echo Install Python 3.12 for Windows from python.org with the Python Launcher and Tcl/Tk options.
goto failed

:failed
echo Setup did not complete. See the error above.
pause
exit /b 1
