# Breast Cancer Histopathology Image Classification Software - Group 6

A graphical software tool for classifying breast cancer histopathology staining images as **benign** or **malignant** using **traditional machine learning**, **deep learning**, and **hybrid feature fusion models**.

## Project Overview
This software is designed to assist in the classification of breast cancer histopathology images by combining multiple feature extraction strategies with flexible classification models. Users can interactively upload images, crop regions of interest, select different models, and obtain prediction results with probability estimates and visual explanations.

**Key features include:**
- Image upload and region-of-interest (ROI) cropping
- Multiple feature extraction methods:
  - Radiomics (handcrafted features)
  - SLIC+Kmeans+Birch segmentation and SIFT feature extraction
  - Hybrid feature fusion (manually extracted + DL features)
- Prediction probabilities (benign vs malignant)
- Grad-CAM visualization for model interpretability
- User-friendly graphical user interface (GUI)


## Software Workflow
Image Input → Image Cropping → Model Selection → Prediction → Probability Output → Grad-CAM (if CNN models)

## Feature Extraction Methods

### 1. Radiomics Features
Radiomics is a widely used medical imaging approach that extracts quantitative intensity, texture, and shape features from images to uncover patterns relevant for diagnosis and prognosis.

### 2. SLIC segmentation + SIFT features
This method first applies SLIC superpixel segmentation to partition the image into perceptually homogeneous regions, and then extracts SIFT features within these regions to capture robust local texture and structural information for classification (Manivannan et al., 2025).


### 3. Hybrid Feature Fusion
We integrate manually extracted features such as radiomics and SIFT with deep learning–based features, concatenate them, and perform classification using a multilayer perceptron (MLP).


## Classification Models
Depending on the selected feature extraction method, the software supports:
- Support Vector Machine (SVM)
- Random Forest
- Multilayer Perceptron (MLP)
- ResNet18
- ResNet50
- DesNet121
- EfficientNet-B0
- VGG16
- VGG19
- 

## Graphical User Interface (GUI)
The GUI includes the following main components:
- **Image Upload Panel** for loading histopathology images
- **Cropping Tool** for selecting regions of interest
- **Model Selection Panel** for choosing feature extraction and classification method combinations
- **Result Display Panel** showing classification results and probabilities
- **Grad-CAM Visualization Panel** (available for deep learning models)


## How to Use the Software

### Step 1: Launch the Software
Run the main program file to open the graphical user interface.

### Step 2: Upload an Image
Click the **Upload Image** button and select a breast cancer histopathology staining image.

### Step 3: Crop the Image (Optional)
Use the cropping tool to define the region of interest (ROI) and confirm the selection.

### Step 4: Select models
<img width="633" height="306" alt="Screenshot 2025-12-10 at 19 33 08" src="https://github.com/user-attachments/assets/81bac5e8-5c3a-4b65-ac54-130e9c5d3a51" />

### Step 5: Run Prediction
Click the **Predict** button. The software will output:
- Predicted class (Benign or Malignant)
- Prediction probability

Example output:



### Step 6: Model Interpretability (Grad-CAM only for DL models)

For deep learning–based models, Grad-CAM is available to visualize regions of the image that contribute most to the classification decision. The generated heatmap is overlaid on the original image, where warmer colors indicate higher importance.



## System Requirements
### Hardware
- CPU: Intel i5 / Apple Silicon or higher
- RAM: ≥ 8 GB recommended
- GPU: optional

### Software
- Python 3.10 or above
- PyTorch
- OpenCV
- NumPy


## Limitations

- This software does not replace professional medical diagnosis
- Model performance depends on training data quality and distribution
- Variations in staining protocols may affect predictions
- Insufficient data from the dataset


## Disclaimer

This software is intended **for academic, educational, and research purposes only**.  
It is **not approved for clinical diagnosis or medical decision-making**.


## Authors

Developed by **[Group 6 in BIA4 class]**

## Acknowledgments

- BreakHis dataset
- PyTorch and OpenCV open-source communities

