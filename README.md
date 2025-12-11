# Breast Cancer Cell Biopsy Classifier

A machine learning-based GUI application for breast cancer cell biopsy classification.

## Features

- 🖼️ **Image Viewer**: Supports PNG, JPG, BMP, TIFF medical images
- ✂️ **Image Cropping**: ROI selection with automatic padding
- 🧠 **Multi-Model Support**:
  - Radiomics feature models
  - Pixel-level CNN models (ResNet, DenseNet, EfficientNet, etc.)
  - Fusion models (Radiomics + CNN)
- 📊 **Visualization Analysis**:
  - GradCAM heatmaps (shows model attention regions)
  - Occlusion sensitivity maps (shows critical regions)
- 📋 **Clinical Reports**: Detailed prediction reports with clinical recommendations
- 🎨 **Modern Interface**: Professional medical application design

## System Requirements

- Python 3.8+
- Recommended: 8GB+ RAM, CUDA-compatible GPU (optional)

## Installation and Running

### Method 1: Run as Python Application

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python gui.py
```

### Method 2: Package as Standalone Executable

#### Cross-Platform Automated Build (Recommended)

```bash
# Use the automated build script
python build_exe.py
```

After building, the executable is located at `dist/BreastCancerClassifier`

#### Windows-Specific Build

```cmd
# Double-click or run in command prompt
build_windows.bat
```

#### macOS/Linux-Specific Build

```bash
# Run the build script
./build_unix.sh
```

#### Manual Build (Advanced Users)

```bash
# 1. Install PyInstaller
pip install pyinstaller

# 2. Build executable
pyinstaller --onefile --windowed --name BreastCancerClassifier gui.py

# 3. Add model files
# Windows:
pyinstaller --onefile --windowed --add-data "ml_backend.py;." --add-data "models;models" gui.py

# macOS/Linux:
pyinstaller --onefile --windowed --add-data "ml_backend.py:." --add-data "models:models" gui.py
```

## Project Structure

```
├── gui.py                 # Main GUI application
├── ml_backend.py          # Machine learning backend
├── requirements.txt       # Python dependencies
├── build_exe.py          # Automated build script
├── build_windows.bat     # Windows build script
├── build_unix.sh         # macOS/Linux build script
├── README.md             # This documentation
└── [model folders]/      # Trained model files
    ├── model (.pth)/
    ├── other info/
    └── ...
```

## Usage Instructions

### Basic Workflow

1. **Launch Application**: Run `python gui.py` or double-click the executable
2. **Select Model**: Choose a pretrained model from the dropdown menu
3. **Load Image**: Click "Open Image" to select a medical image
4. **Optional Operations**:
   - Use "Crop" tool to select region of interest
   - Use "Reset" to restore original image
5. **Predict Classification**: Click "Predict Classification"
6. **Visualization Analysis**:
   - Click "GradCAM" to view model attention regions
   - Click "Occlusion" to view critical region sensitivity

### Model Type Descriptions

- **Radiomics Only**: MLP model based on 39 radiomics features
- **Pixel Only**: CNN model that uses images directly
- **Fusion**: Combined model using both radiomics features and images

## Technical Details

### Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **Pillow**: Image loading and manipulation
- **scikit-image**: Radiomics feature extraction
- **matplotlib**: Visualization
- **tkinter**: GUI framework

### Feature Extraction

- **Radiomics Features**: GLCM, GLRLM, GLSZM statistical features
- **Pixel Features**: RGB channel statistics and texture features

### Model Architectures

- **RadiomicsMLP**: 39-dimensional input → 128 → 64 → 2-class classification
- **CNN Models**: ResNet, DenseNet, EfficientNet pretrained models
- **Fusion Models**: Radiomics features + ResNet feature fusion

## Troubleshooting

### Common Issues

1. **Model Loading Failure**
   - Ensure .pth files are in correct locations
   - Check file permissions

2. **CUDA Errors**
   - Install CUDA version of PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

3. **Memory Insufficient**
   - Use smaller models or reduce batch size
   - Close unnecessary applications

4. **Executable Build Failure**
   - Ensure all dependencies are installed
   - Check disk space (requires 2-3GB)

### Performance Optimization

- Use GPU acceleration (if available)
- Choose appropriate model sizes
- Regularly clean temporary files

## License

This project is for educational and research purposes only.

## Contributing

Issues and improvement suggestions are welcome!

## Version History

- **v2.0**: Added Occlusion visualization, improved interface, enhanced clinical reports
- **v1.0**: Basic functionality implementation
