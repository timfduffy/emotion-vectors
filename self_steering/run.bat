@echo off
cd /d "%~dp0"
echo Self-Steering Chat
echo ==================
echo.
echo Configuration:
echo   Layer range: 19-24 (default)
echo   Decay mode: none (persistent steering)
echo.
echo Starting...
echo.
python app.py
pause
