@echo off
cd /d "%~dp0"
echo Starting Emotion Viewer...
echo.
echo This viewer does NOT load the model - it only visualizes saved activations.
echo To generate new activations, run generate.bat first.
echo.
echo Opening http://127.0.0.1:7860 in your browser...
echo Press Ctrl+C to stop the server
echo.
python viewer.py
pause
