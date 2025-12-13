@echo off
REM Breast Cancer Classifier - Windows Build Script
echo ================================================
echo Breast Cancer Classifier - Windows Build
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Install PyInstaller
echo Installing PyInstaller...
pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)

REM Create dist directory if it doesn't exist
if not exist dist mkdir dist

REM Build executable
echo Building executable...
pyinstaller --clean --onefile --windowed --name BreastCancerClassifier --add-data "ml_backend.py;." gui.py
if errorlevel 1 (
    echo ERROR: Failed to build executable
    pause
    exit /b 1
)

echo.
echo ================================================
echo BUILD COMPLETED SUCCESSFULLY!
echo.
echo Executable location: dist\BreastCancerClassifier.exe
echo.
echo You can now distribute this single .exe file
echo ================================================
pause
