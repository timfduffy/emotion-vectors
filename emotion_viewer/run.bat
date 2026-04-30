@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat
echo Starting Emotion Viewer...
echo.
echo Once loaded, open http://127.0.0.1:7860 in your browser
echo Press Ctrl+C to stop the server
echo.
python app.py
pause
