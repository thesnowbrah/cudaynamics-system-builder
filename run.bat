@echo off
setlocal
cd /d "%~dp0"
set NO_ALBUMENTATIONS_UPDATE=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
    if errorlevel 1 pause
    exit /b
)
where py >nul 2>nul
if not errorlevel 1 (
    py app.py
    if errorlevel 1 pause
    exit /b
)
where python >nul 2>nul
if not errorlevel 1 (
    python app.py
    if errorlevel 1 pause
    exit /b
)
echo Python environment not found. Create .venv and run: pip install -r requirements.txt
pause
