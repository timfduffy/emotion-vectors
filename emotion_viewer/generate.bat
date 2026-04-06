@echo off
cd /d "%~dp0"
echo.
echo Emotion Activation Generator
echo ============================
echo.
set /p PROMPT="Enter your prompt: "
echo.
echo Generating response and capturing activations...
echo.
python generate_activations.py "%PROMPT%"
echo.
pause
