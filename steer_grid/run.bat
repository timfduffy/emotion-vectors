@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat
echo Steer Grid
echo ==========
echo.
if "%~1"=="" (
  echo Usage: run.bat path\to\config.yaml [extra args]
  echo.
  echo Example:
  echo   run.bat example_config.yaml
  pause
  exit /b 1
)
python run.py --config %*
pause
