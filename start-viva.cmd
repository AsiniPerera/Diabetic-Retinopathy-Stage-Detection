@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Local Python environment is missing.
  echo Create it and install requirements_viva.txt before starting the demo.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8765 in your browser.
echo Keep this window open during the viva. Press Ctrl+C to stop.
".venv\Scripts\python.exe" viva_server.py
pause
