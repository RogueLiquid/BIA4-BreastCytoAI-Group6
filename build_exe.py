#!/usr/bin/env python3
"""
Build script for Breast Cancer Classifier executable
Run with: python build_exe.py
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_pyinstaller():
    """Install PyInstaller if not present"""
    try:
        import PyInstaller
        print("PyInstaller already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def create_spec_file():
    """Create PyInstaller spec file"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(SPEC))

# Collect all model files
model_files = []
for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file.endswith('.pth'):
            model_files.append((os.path.join(root, file), os.path.relpath(root, current_dir)))

a = Analysis(
    ['gui.py'],
    pathex=[current_dir],
    binaries=[],
    datas=model_files + [
        ('ml_backend.py', '.'),
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'skimage',
        'skimage.color',
        'skimage.feature',
        'skimage.exposure',
        'skimage.segmentation',
        'skimage.filters',
        'scipy',
        'scipy.ndimage',
        'sklearn',
        'sklearn.cluster',
        'sklearn.ensemble',
        'sklearn.metrics'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BreastCancerClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for production (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file path if you have one
)
'''
    with open('breast_cancer_classifier.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("Created spec file: breast_cancer_classifier.spec")

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable...")
    cmd = [
        sys.executable, "-m", "pyinstaller",
        "--clean",
        "--onefile",  # Single executable file
        "--windowed",  # No console window (for GUI apps)
        "--name", "BreastCancerClassifier",
        "--add-data", "ml_backend.py;.",
    ]

    # Add all model files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pth'):
                if sys.platform == 'win32':
                    cmd.extend(["--add-data", f"{os.path.join(root, file)};{root}"])
                else:  # macOS/Linux
                    cmd.extend(["--add-data", f"{os.path.join(root, file)}:{root}"])

    cmd.append("gui.py")

    subprocess.check_call(cmd)
    print("Executable built successfully!")

def main():
    """Main build process"""
    print("="*60)
    print("Breast Cancer Classifier - Build Script")
    print("="*60)

    try:
        # Install dependencies
        install_dependencies()

        # Install PyInstaller
        install_pyinstaller()

        # Create spec file
        create_spec_file()

        # Build executable
        build_executable()

        print("\n" + "="*60)
        print("BUILD COMPLETED SUCCESSFULLY!")
        print("Executable location: dist/BreastCancerClassifier")
        print("="*60)

    except Exception as e:
        print(f"\nBuild failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
