#!/bin/bash
# Breast Cancer Classifier - Unix/macOS Build Script

echo "================================================"
echo "Breast Cancer Classifier - Unix/macOS Build"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+"
    exit 1
fi

# Use python3 or python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check Python version
$PYTHON_CMD --version

# Install dependencies
echo "Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Install PyInstaller
echo "Installing PyInstaller..."
$PYTHON_CMD -m pip install pyinstaller
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyInstaller"
    exit 1
fi

# Create dist directory if it doesn't exist
mkdir -p dist

# Build executable
echo "Building executable..."
$PYTHON_CMD -m pyinstaller --clean --onefile --windowed --name BreastCancerClassifier --add-data "ml_backend.py:." gui.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build executable"
    exit 1
fi

echo ""
echo "================================================"
echo "BUILD COMPLETED SUCCESSFULLY!"
echo ""
echo "Executable location: dist/BreastCancerClassifier"
echo ""
echo "You can now distribute this single executable file"
echo "================================================"
