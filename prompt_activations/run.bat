@echo off
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat
echo Prompt Activations
echo ==================
echo.
if "%~1"=="" (
  echo Usage: run.bat path\to\config.yaml [extra args]
  echo.
  echo Example:
  echo   run.bat test_config.yaml
  echo   run.bat example_config.yaml --output-dir runs/full
  pause
  exit /b 1
)
python run.py --config %*
pause
