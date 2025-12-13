import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.exposure import rescale_intensity

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ===========================================

# ================= 模型定义 =================

class RadiomicsMLP(nn.Module):
    """基于影像组学特征的MLP模型"""
    def __init__(self, input_dim=39, num_classes=2, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class SimpleMLP(nn.Module):
    """A simple multilayer perceptron for pixel features"""
    def __init__(self, input_dim=3*224*224, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

class ResNet50RadiomicsFusion(nn.Module):
    """融合 ResNet50 像素特征和 Radiomics 特征的模型"""
    def __init__(self, radiomics_dim=39, resnet_feature_dim=256, radiomics_reduced_dim=64, 
                 fusion_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_reducer = nn.Sequential(
            nn.Linear(2048, resnet_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.radiomics_reducer = nn.Sequential(
            nn.Linear(radiomics_dim, radiomics_reduced_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        fusion_input_dim = resnet_feature_dim + radiomics_reduced_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )
        
    def forward(self, img, radiomics_features):
        resnet_features = self.resnet_backbone(img)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        resnet_features = self.resnet_reducer(resnet_features)
        radiomics_reduced = self.radiomics_reducer(radiomics_features)
        fused_features = torch.cat([resnet_features, radiomics_reduced], dim=1)
        output = self.fusion_classifier(fused_features)
        return output

# CNN models for pixel-only
class ResNet18Empty(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class ResNet18_PT_FT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class ResNet18Freeze(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in base.parameters():
            param.requires_grad = False
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

class ResNet50_PT_FT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class ResNet50Freeze(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in base.parameters():
            param.requires_grad = False
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

class ResNet50Empty(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

class ResNet101_PT_FT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class DenseNet121_PT_FT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        base.classifier = nn.Linear(base.classifier.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class DenseNet201_PT_FT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        base.classifier = nn.Linear(base.classifier.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class EfficientNetB0_PT_FT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class MobileNetV2_PT_FT(nn.Module):
    """Fine-tuned MobileNetV2 with pretrained weights."""
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*28*28, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        base.classifier[6] = nn.Linear(base.classifier[6].in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

# ================= 模型加载管理 =================

_model_instances = {}  # 缓存已加载的模型

# Clean model type mapping based on folder structure
MODEL_TYPE_MAPPING = {
    # Radiomics models
    "Radiomics_MLP": "radiomics",
    "Radiomics_RandomForest": "pixel_ml",  # Same as pixel models: extracts pixel features from image

    # Pixel models (CNN architectures)
    "Pixel_DenseNet121": "pixel",
    "Pixel_EfficientNetB0": "pixel",
    "Pixel_MobileNetV2": "pixel",
    "Pixel_Resnet50": "pixel",
    "Pixel_SimpleCNN": "pixel",
    "Pixel_VGG": "pixel",

    # SIFT models
    "SIFT_RandomForest": "sift_ml",        # Trained on SIFT features (500)

    # Fusion models (Radiomics + Pixel)
    "Radiomics+Pixel_Fusion_DenseNet121": "fusion",
    "Radiomics+Pixel_Fusion_ResNet50": "fusion",

    # SIFT fusion models
    "SIFT+Pixel_Fusion_DenseNet121": "sift_fusion",
    "SIFT+Pixel_Fusion_ResNet50": "sift_fusion",
}

def _get_model_class_name(model_path):
    """从模型路径推断模型类名"""
    dir_name = os.path.basename(os.path.dirname(model_path))
    # 提取模型名称（去掉epoch后缀）
    parts = dir_name.split('_')
    if len(parts) >= 2:
        # 检查是否是数字（epoch）
        try:
            int(parts[-1])
            model_name = '_'.join(parts[:-1])
        except ValueError:
            model_name = dir_name
    else:
        model_name = dir_name

    return model_name

def _get_model_type_from_folder(folder_name):
    """
    Clean model type detection based on folder name mapping

    Args:
        folder_name: Folder name containing the model

    Returns:
        model_type: String indicating model type
    """
    # Direct mapping first
    if folder_name in MODEL_TYPE_MAPPING:
        return MODEL_TYPE_MAPPING[folder_name]

    # Pattern-based fallback for robustness
    if "RandomForest" in folder_name:
        if "sift" in folder_name.lower():
            return "sift_ml"
        else:
            return "pixel_ml"  # All other RandomForest models are pixel-based
    elif "MLP" in folder_name:
        return "radiomics"
    elif folder_name.startswith("Pixel_"):
        return "pixel"
    elif "Fusion" in folder_name:
        if "SIFT" in folder_name:
            return "sift_fusion"
        else:
            return "fusion"
    else:
        # Default fallback
        return "pixel"

def load_model(model_path, model_type="auto"):
    """
    加载模型
    model_path: 模型文件路径 (如 "RadiomicsMLP_30/model.pth")
    model_type: "radiomics", "pixel", "fusion", 或 "auto" (自动检测)
    """
    global _model_instances
    
    if model_path in _model_instances:
        return _model_instances[model_path]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_name = _get_model_class_name(model_path)
    
    # Clean model type detection based on folder name
    if model_type == "auto":
        folder_name = os.path.basename(os.path.dirname(model_path))
        model_type = _get_model_type_from_folder(folder_name)
    
    model = None
    
    # 根据模型类型和名称创建模型
    if model_type == "radiomics":
        model = RadiomicsMLP(input_dim=39, num_classes=2)
    elif model_type == "pixel":
        if "ResNet18Empty" in model_name:
            model = ResNet18Empty(num_classes=2)
        elif "ResNet18" in model_name and "PT" in model_name:
            model = ResNet18_PT_FT(num_classes=2)
        elif "ResNet18Freeze" in model_name:
            model = ResNet18Freeze(num_classes=2)
        elif "ResNet50" in model_name and "PT" in model_name:
            model = ResNet50_PT_FT(num_classes=2)
        elif "ResNet50Freeze" in model_name:
            model = ResNet50Freeze(num_classes=2)
        elif "ResNet50Empty" in model_name:
            model = ResNet50Empty(num_classes=2)
        elif "ResNet101" in model_name:
            model = ResNet101_PT_FT(num_classes=2)
        elif "DenseNet121" in model_name:
            model = DenseNet121_PT_FT(num_classes=2)
        elif "DenseNet201" in model_name:
            model = DenseNet201_PT_FT(num_classes=2)
        elif "EfficientNet" in model_name:
            model = EfficientNetB0_PT_FT(num_classes=2)
        elif "MobileNetV2" in model_name:
            model = MobileNetV2_PT_FT(num_classes=2)
        elif "SimpleCNN" in model_name:
            model = SimpleCNN(num_classes=2)
        elif "VGG16" in model_name:
            model = VGG16(num_classes=2)
        elif "SimpleMLP" in model_name:
            model = SimpleMLP(input_dim=3*224*224, num_classes=2)
        else:
            # 默认使用 ResNet50
            model = ResNet50_PT_FT(num_classes=2)
    elif model_type == "radiomics_ml":
        # For sklearn models, load directly from pickle
        import pickle
        try:
            with open(model_path, 'rb') as f:
                loaded_data = pickle.load(f)

            # Handle models saved as dictionary with 'model' key
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                model = loaded_data['model']
            else:
                model = loaded_data

            print(f"✅ Sklearn model loaded: {model_name}")
            return model, model_type
        except Exception as e:
            print(f"❌ Failed to load sklearn model: {e}")
            raise
    elif model_type == "sift_ml":
        # For sklearn models, load directly from pickle
        import pickle
        try:
            with open(model_path, 'rb') as f:
                loaded_data = pickle.load(f)

            # Handle models saved as dictionary with 'model' key
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                model = loaded_data['model']
            else:
                model = loaded_data

            print(f"✅ Sklearn model loaded: {model_name}")
            return model, model_type
        except Exception as e:
            print(f"❌ Failed to load sklearn model: {e}")
            raise
    elif model_type == "sift_fusion":
        if "DenseNet121" in model_name:
            model = DenseNet121SIFTFusion(num_classes=2)
        elif "ResNet50" in model_name:
            model = ResNet50SIFTFusion(num_classes=2)
        else:
            # Default to DenseNet121SIFTFusion
            model = DenseNet121SIFTFusion(num_classes=2)
    elif model_type == "fusion":
        if "ResNet50" in model_name:
            model = ResNet50RadiomicsFusion(
                radiomics_dim=39,
                resnet_feature_dim=256,
                radiomics_reduced_dim=64,
                fusion_dim=128,
                num_classes=2,
                dropout=0.3
            )
        elif "DenseNet121" in model_name:
            model = ResNet50RadiomicsFusion(
                radiomics_dim=39,
                resnet_feature_dim=256,
                radiomics_reduced_dim=64,
                fusion_dim=128,
                num_classes=2,
                dropout=0.3
        )
    
    if model is None:
        raise ValueError(f"Could not determine model class for: {model_name}")
    
    # 加载权重
    try:
        # Try with weights_only=True first (PyTorch 2.6+ default)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        except Exception:
            # Fallback for older PyTorch models
            print(f"⚠️  Failed to load with weights_only=True, trying fallback...")
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"⚠️  Missing keys in state_dict: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in state_dict: {len(unexpected_keys)} keys")

        model.to(DEVICE)
        model.eval()

        _model_instances[model_path] = (model, model_type)
        print(f"✅ Model loaded: {model_name} ({model_type}) from {model_path}")

        return model, model_type

    except Exception as e:
        print(f"❌ Failed to load model {model_path}: {str(e)}")
        raise

# ================= SIFT特征提取 =================

def apply_slic_segmentation(img, n_segments=100, compactness=10):
    """
    Apply SLIC superpixel segmentation + K-Means + BIRCH refinement

    Args:
        img: RGB image (H x W x 3)
        n_segments: Number of SLIC superpixels
        compactness: SLIC compactness parameter

    Returns:
        segmented_image: Segmented RGB image
    """
    try:
        from sklearn.cluster import KMeans, Birch
        from skimage.segmentation import slic

        # Step 1: SLIC Superpixel Segmentation
        segments = slic(img, n_segments=n_segments, compactness=compactness,
                       sigma=1, start_label=1, channel_axis=-1)

        # Step 2: Extract features from each superpixel
        superpixel_features = []
        segment_ids = np.unique(segments)

        for segment_id in segment_ids:
            mask = segments == segment_id
            region = img[mask]

            # Features: mean RGB (3) + std RGB (3) + area (1) = 7 features
            mean_rgb = np.mean(region, axis=0)
            std_rgb = np.std(region, axis=0)
            area = np.sum(mask)

            features = np.concatenate([mean_rgb, std_rgb, [area]])
            superpixel_features.append(features)

        superpixel_features = np.array(superpixel_features)

        # Step 3: K-Means clustering (n_clusters=5 as in notebook)
        kmeans = KMeans(n_clusters=5, random_state=666, n_init=10)
        kmeans_labels = kmeans.fit_predict(superpixel_features)

        # Step 4: BIRCH refinement
        birch = Birch(n_clusters=5, threshold=0.5)
        final_labels = birch.fit_predict(superpixel_features)

        # Step 5: Create segmented image (keep original pixel values)
        segmented_image = np.zeros_like(img)
        for idx, segment_id in enumerate(segment_ids):
            mask = segments == segment_id
            segmented_image[mask] = img[mask]

        return segmented_image

    except Exception as e:
        print(f"SLIC segmentation error: {e}")
        return img


def extract_sift_features(image, n_features=100):
    """Extract SIFT features from image"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=n_features)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None:
            return np.zeros((n_features, 128))

        if len(descriptors) < n_features:
            padding = np.zeros((n_features - len(descriptors), 128))
            descriptors = np.vstack([descriptors, padding])
        else:
            descriptors = descriptors[:n_features]

        return descriptors

    except Exception as e:
        print(f"SIFT extraction error: {e}")
        return np.zeros((n_features, 128))


def fuzzy_cmeans_clustering(descriptors, n_clusters=500, m=2, max_iter=100):
    """Fuzzy C-Means clustering on SIFT descriptors (from sift concat.ipynb)"""
    try:
        from skfuzzy import cluster

        descriptors_t = descriptors.T
        cntr, u, u0, d, jm, p, fpc = cluster.cmeans(
            descriptors_t, n_clusters, m, error=0.005, maxiter=max_iter, init=None
        )
        cluster_membership = np.argmax(u, axis=0)
        return cntr, cluster_membership
    except Exception as e:
        print(f"FCM clustering error: {e}")
        # Fallback to K-Means if FCM fails
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=666, n_init=10)
        kmeans.fit(descriptors)
        return kmeans.cluster_centers_, kmeans.predict(descriptors)


def create_bof_histogram(descriptors, vocabulary):
    """Create Bag-of-Features histogram (from sift only ml and just dl.ipynb)"""
    try:
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(500)  # Assuming 500 clusters

        # Use Fuzzy C-Means for clustering (from sift concat.ipynb)
        _, membership = fuzzy_cmeans_clustering(descriptors, n_clusters=500)

        # Create histogram
        histogram, _ = np.histogram(membership, bins=np.arange(501))

        # Normalize
        histogram = histogram.astype(float)
        if histogram.sum() > 0:
            histogram /= histogram.sum()

        return histogram

    except Exception as e:
        print(f"BoF histogram creation error: {e}")
        return np.zeros(500)


def extract_sift_bof_features(img_path, vocabulary_path=None, n_features=100, n_clusters=500):
    """
    Extract SIFT+BoF features for an image

    Args:
        img_path: Path to image
        vocabulary_path: Path to saved vocabulary (KMeans model)
        n_features: Number of SIFT features per image
        n_clusters: Number of BoF clusters

    Returns:
        features: BoF histogram (n_clusters dim)
        feature_names: List of feature names
    """
    try:
        # Load vocabulary if provided
        vocabulary = None
        if vocabulary_path and os.path.exists(vocabulary_path):
            import pickle
            with open(vocabulary_path, 'rb') as f:
                vocabulary = pickle.load(f)
        else:
            # Try to find vocabulary relative to this script or in standard locations
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                'vocabulary.pkl',  # Current directory
                os.path.join(script_dir, 'vocabulary.pkl'),  # Script directory
                'models/vocabulary.pkl',  # models subdirectory
                'SIFT_RandomForest/vocabulary.pkl',  # SIFT directory
                os.path.join(script_dir, 'SIFT_RandomForest', 'vocabulary.pkl')  # Script/SIFT directory
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            vocabulary = pickle.load(f)
                        print(f"Found vocabulary at: {path}")
                        break
                    except:
                        continue

            # If still no vocabulary, create a default one for prediction
            if vocabulary is None:
                print("No vocabulary found, creating default vocabulary for prediction")
                # Create a simple default vocabulary (not trained, just for shape compatibility)
                from sklearn.cluster import KMeans
                # Use some dummy data to fit the model
                dummy_data = np.random.rand(100, 128)
                vocabulary = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                vocabulary.fit(dummy_data)

        # Read and process image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Apply SLIC segmentation
        segmented = apply_slic_segmentation(img)

        # Extract SIFT features
        descriptors = extract_sift_features(segmented, n_features)

        # Create BoF histogram
        histogram = create_bof_histogram(descriptors, vocabulary)

        feature_names = [f"BoF_{i}" for i in range(len(histogram))]

        return histogram, feature_names

    except Exception as e:
        print(f"SIFT-BoF extraction error: {e}")
        # Return zeros as fallback
        return np.zeros(n_clusters), [f"BoF_{i}" for i in range(n_clusters)]


# ================= 模型定义 =================

class SIFTBoFMLP(nn.Module):
    """MLP for SIFT+BoF features"""
    def __init__(self, input_dim=500, num_classes=2, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class HybridClassifier(nn.Module):
    """Adaptive classifier for hybrid features (SIFT + CNN) - from sift concat.ipynb"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()

        # Adaptive architecture based on input dimension (from notebook)
        if input_dim >= 4000:  # VGG16/19: 4596
            hidden_dims = [2048, 1024, 512, 256]
        elif input_dim >= 2500:  # ResNet50: 2548
            hidden_dims = [1024, 512, 256]
        elif input_dim >= 1700:  # EfficientNet: 1780
            hidden_dims = [768, 384, 192]
        else:  # DenseNet121: 1524
            hidden_dims = [640, 320, 160]

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class DenseNet121SIFTFusion(nn.Module):
    """融合 DenseNet121 像素特征和 SIFT+BoF 特征的模型 - based on sift concat.ipynb"""
    def __init__(self, num_classes=2):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # Use the same architecture as in the notebook
        self.densenet_backbone = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # This reduces to 1024x1x1
        )
        self.classifier = HybridClassifier(input_dim=500 + 1024, num_classes=num_classes)  # 500 SIFT + 1024 DenseNet

    def forward(self, img, sift_features):
        # Extract DenseNet features
        densenet_features = self.densenet_backbone(img)
        densenet_features = densenet_features.view(densenet_features.size(0), -1)  # Flatten to (batch, 1024)

        # Concatenate with SIFT features
        fused_features = torch.cat([sift_features, densenet_features], dim=1)

        # Classify
        output = self.classifier(fused_features)
        return output


class ResNet50SIFTFusion(nn.Module):
    """融合 ResNet50 像素特征和 SIFT+BoF 特征的模型"""
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = HybridClassifier(input_dim=500 + 2048, num_classes=num_classes)  # 500 SIFT + 2048 ResNet

    def forward(self, img, sift_features):
        # Extract ResNet features
        resnet_features = self.resnet_backbone(img)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)  # Flatten

        # Concatenate with SIFT features
        fused_features = torch.cat([sift_features, resnet_features], dim=1)

        # Classify
        output = self.classifier(fused_features)
        return output

# ================= 特征提取 =================

def extract_radiomics_features(img_path):
    """提取影像组学特征 (39维)"""
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError("Could not read image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        feats = {}
        H, W = 256, 256
        img = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

        img_hsv = (rgb2hsv(img / 255.0) * 255.0).astype(np.uint8)
        for space_name, arr in [("rgb", img), ("hsv", img_hsv)]:
            for c, cname in enumerate(["c1","c2","c3"]):
                ch = arr[..., c].astype(np.float32)
                feats[f"{space_name}_{cname}_mean"] = ch.mean()
                feats[f"{space_name}_{cname}_std"]  = ch.std()

        gray = (rgb2gray(img) * 255).astype(np.uint8)
        feats["gray_mean"] = gray.mean()
        feats["gray_std"]  = gray.std()
        feats["gray_min"]  = float(gray.min())
        feats["gray_max"]  = float(gray.max())
        
        hist_vals, _ = np.histogram(gray, bins=32, range=(0,255), density=True)
        feats["gray_entropy"] = -np.sum((hist_vals + 1e-12) * np.log2(hist_vals + 1e-12))

        q = np.floor(rescale_intensity(gray, in_range="image", out_range=(0,7))).astype(np.uint8)
        distances = [1, 2, 4]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(q, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
        for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
            vals = graycoprops(glcm, prop).ravel()
            feats[f"glcm_{prop}_mean"] = vals.mean()
            feats[f"glcm_{prop}_std"]  = vals.std()

        P, R = 8, 1
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        for i, v in enumerate(hist):
            feats[f"lbp_u{i}"] = float(v)
            
        feature_values = list(feats.values())
        feature_names = list(feats.keys())
        
        return feature_values, feature_names

    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return [], []

def extract_pixel_features(img_path):
    """提取像素特征 (用于SimpleMLP)"""
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        features = img.flatten().tolist()
        names = [f"Px_{i}" for i in range(len(features))]
        return features, names
    except Exception as e:
        print(f"Pixel feature extraction error: {e}")
        return [], []

def prepare_image_tensor(img_path):
    """准备图像张量 (用于CNN模型)"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img_tensor

# ================= 预测函数 =================

def predict_with_model(model_path, features=None, img_path=None, model_type="auto"):
    """
    通用预测函数，支持多种模型类型：
    - Radiomics MLP / RF
    - Pixel CNN / MLP
    - SIFT ML / Fusion
    - Fusion (Radiomics + Pixel / SIFT + Pixel)

    ```
    如果是 Radiomics MLP 或 RF，只需提供 img_path 或 features。

    Args:
        model_path: 模型文件路径
        features: 特征列表 (用于 MLP / Radiomics / Pixel)
        img_path: 图像路径 (用于 CNN / Fusion / SIFT / 自动提取 Radiomics)
        model_type: "radiomics", "pixel", "fusion", "sift_ml", "sift_fusion", 或 "auto"

    Returns:
        (prediction, confidence, details)
    """
    import numpy as np
    import torch

    try:
        model, detected_type = load_model(model_path, model_type)
        classes = ["BENIGN", "MALIGNANT"]

        # =================== Radiomics MLP / RF ===================
        if detected_type in ["radiomics", "radiomics_ml"]:
            # 自动提取特征
            if features is None:
                if img_path is None:
                    return "ERROR", 0.0, {'error': 'Radiomics model requires features or image path'}
                features, _ = extract_radiomics_features(img_path)

            if len(features) != 39:
                return "ERROR", 0.0, {'error': f'Expected 39 radiomics features, got {len(features)}'}

            features_array = np.array(features, dtype=np.float64).reshape(1, -1)

            if detected_type == "radiomics":
                input_tensor = torch.tensor(features_array, dtype=torch.float32).to(DEVICE)
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            else:
                # sklearn RF / MLP
                probs = model.predict_proba(features_array)[0]

            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            details = {'probabilities': {'benign': float(probs[0]), 'malignant': float(probs[1])},
                    'model_type': detected_type}
            prediction = classes[pred_idx]
            return prediction, conf, details

        # =================== Pixel / CNN / MLP ===================
        elif detected_type in ["pixel", "pixel_ml"]:
            if "MLP" in _get_model_class_name(model_path):
                if features is None:
                    return "ERROR", 0.0, {'error': 'Pixel MLP requires features'}
                input_tensor = torch.tensor(np.array(features, dtype=np.float32).reshape(1, -1)).to(DEVICE)
            else:
                if img_path is None:
                    return "ERROR", 0.0, {'error': 'Pixel CNN requires image path'}
                input_tensor = prepare_image_tensor(img_path)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            details = {'probabilities': {'benign': float(probs[0]), 'malignant': float(probs[1])},
                    'model_type': detected_type}
            prediction = classes[pred_idx]
            return prediction, conf, details

        # =================== SIFT ML ===================
        elif detected_type == "sift_ml":
            if features is None or len(features) != 500:
                return "ERROR", 0.0, {'error': 'SIFT ML model requires 500 BoF features'}
            features_array = np.array(features, dtype=np.float64).reshape(1, -1)
            probs = model.predict_proba(features_array)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            details = {'probabilities': {'benign': float(probs[0]), 'malignant': float(probs[1])},
                    'model_type': detected_type}
            prediction = classes[pred_idx]
            return prediction, conf, details

        # =================== SIFT Fusion ===================
        elif detected_type == "sift_fusion":
            if img_path is None:
                return "ERROR", 0.0, {'error': 'SIFT Fusion model requires image path'}
            sift_features, _ = extract_sift_bof_features(img_path)
            if len(sift_features) != 500:
                return "ERROR", 0.0, {'error': f'Failed to extract 500 SIFT-BoF features, got {len(sift_features)}'}
            img_tensor = prepare_image_tensor(img_path)
            sift_tensor = torch.tensor(np.array([sift_features]), dtype=torch.float32).to(DEVICE)
            model.eval()
            with torch.no_grad():
                output = model(img_tensor, sift_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            details = {'probabilities': {'benign': float(probs[0]), 'malignant': float(probs[1])},
                    'model_type': detected_type}
            prediction = classes[pred_idx]
            return prediction, conf, details

        # =================== Pixel + Radiomics Fusion ===================
        elif detected_type == "fusion":
            if img_path is None or features is None or len(features) != 39:
                return "ERROR", 0.0, {'error': 'Fusion model requires image path + 39 radiomics features'}
            img_tensor = prepare_image_tensor(img_path)
            radiomics_tensor = torch.tensor(np.array([features], dtype=np.float32)).to(DEVICE)
            model.eval()
            with torch.no_grad():
                output = model(img_tensor, radiomics_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            details = {'probabilities': {'benign': float(probs[0]), 'malignant': float(probs[1])},
                    'model_type': detected_type}
            prediction = classes[pred_idx]
            return prediction, conf, details

        else:
            return "ERROR", 0.0, {'error': f'Unknown model type: {detected_type}'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "ERROR", 0.0, {'error': str(e)}

# ================= GradCAM =================

class GradCAM:
    """GradCAM实现"""
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer
        self.hook()

    def hook(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def forward_hook(module, input, output):
            self.feature_map = output
        
        # 查找目标层
        named_modules = dict(self.model.named_modules())
        if self.target_layer not in named_modules:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        
        layer = named_modules[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, input_img, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_img)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[:, class_idx]
        score.backward()
        
        if self.gradients is None:
            raise RuntimeError("Gradients are None. Check if target layer is correct.")
        
        grads = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (grads * self.feature_map).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = cam.cpu().detach().numpy()
        return cam

def get_last_conv_layer_name(model):
    """获取最后一个卷积层名称"""
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return name
    raise ValueError("No Conv2d layer found in model")

def generate_gradcam(model_path, img_path, model_type="auto", target_layer=None):
    """
    生成GradCAM热力图
    
    Returns:
        (cam_array, original_image_array, prediction_info)
    """
    try:
        model, detected_type = load_model(model_path, model_type)
        
        # 只有pixel CNN模型支持GradCAM
        if detected_type != "pixel" or "MLP" in _get_model_class_name(model_path):
            raise ValueError("GradCAM only supported for CNN models (not MLP or radiomics-only models)")
        
        # 准备图像
        img_tensor = prepare_image_tensor(img_path)
        
        # 获取目标层
        if target_layer is None:
            target_layer = get_last_conv_layer_name(model)
        
        # 生成GradCAM
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(img_tensor)
        
        # 获取预测结果
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred_idx].item()
        
        classes = ["BENIGN", "MALIGNANT"]
        prediction = classes[pred_idx]
        
        # 读取原始图像用于显示
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return cam, img_rgb, {
            'prediction': prediction,
            'confidence': conf,
            'probabilities': {
                'benign': probs[0][0].item(),
                'malignant': probs[0][1].item()
            }
        }
        
    except Exception as e:
        print(f"GradCAM Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_occlusion(model_path, img_path, model_type="auto", patch_size=32, stride=16):
    """
    Generate occlusion sensitivity map

    Args:
        model_path: Model file path
        img_path: Image path
        model_type: "pixel", "auto"
        patch_size: Size of occlusion patch (default 32)
        stride: Stride for sliding window (default 16)

    Returns:
        (occlusion_map, original_image_array, prediction_info)
    """
    try:
        model, detected_type = load_model(model_path, model_type)

        if detected_type != "pixel" or "MLP" in _get_model_class_name(model_path):
            raise ValueError("Occlusion sensitivity only supported for CNN models")

        # Prepare image
        img_tensor = prepare_image_tensor(img_path)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get baseline prediction
        model.eval()
        with torch.no_grad():
            baseline_output = model(img_tensor)
            baseline_probs = torch.softmax(baseline_output, dim=1)
            pred_class = torch.argmax(baseline_probs, dim=1).item()
            baseline_conf = baseline_probs[0][pred_class].item()

        # Create occlusion map
        h, w = 224, 224  # Assuming 224x224 input
        occlusion_map = np.zeros((h, w), dtype=np.float32)

        # Create occlusion patch (gray)
        occlusion_patch = torch.ones(3, patch_size, patch_size) * 0.5  # Gray patch

        # Slide patch over image
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Create occluded image
                occluded_img = img_tensor.clone()
                occluded_img[:, :, y:y+patch_size, x:x+patch_size] = occlusion_patch

                # Predict
                with torch.no_grad():
                    output = model(occluded_img)
                    probs = torch.softmax(output, dim=1)
                    conf = probs[0][pred_class].item()

                # Confidence drop (higher drop = more important)
                conf_drop = baseline_conf - conf
                occlusion_map[y:y+patch_size, x:x+patch_size] = np.maximum(
                    occlusion_map[y:y+patch_size, x:x+patch_size], conf_drop
                )

        # Normalize to 0-1
        if occlusion_map.max() > 0:
            occlusion_map = occlusion_map / occlusion_map.max()

        classes = ["BENIGN", "MALIGNANT"]
        prediction = classes[pred_class]

        return occlusion_map, img_rgb, {
            'prediction': prediction,
            'confidence': baseline_conf,
            'probabilities': {
                'benign': baseline_probs[0][0].item(),
                'malignant': baseline_probs[0][1].item()
            }
        }

    except Exception as e:
        print(f"Occlusion Error: {e}")
        import traceback
        traceback.print_exc()
        raise

# ================= 图像处理功能 =================

def apply_enhancement(img_array, contrast=1.0, brightness=0):
    """
    Apply enhancement to image (contrast and brightness adjustment)

    Args:
        img_array: Input image as numpy array (RGB)
        contrast: Contrast factor (0.0-3.0, 1.0 = no change)
        brightness: Brightness offset (-100 to 100, 0 = no change)

    Returns:
        Enhanced image as numpy array
    """
    try:
        # Convert to float for processing
        img_float = img_array.astype(np.float32)

        # Apply contrast
        img_float = img_float * contrast

        # Apply brightness
        img_float = img_float + brightness

        # Clip to valid range and convert back to uint8
        img_float = np.clip(img_float, 0, 255)
        return img_float.astype(np.uint8)

    except Exception as e:
        print(f"Enhancement Error: {e}")
        return img_array

def apply_blur(img_array, kernel_size=5, sigma=0):
    """
    Apply Gaussian blur to image

    Args:
        img_array: Input image as numpy array (RGB)
        kernel_size: Kernel size (odd number, 1-15)
        sigma: Gaussian sigma (0 = auto, >0 = manual)

    Returns:
        Blurred image as numpy array
    """
    try:
        # Ensure kernel size is odd and within bounds
        kernel_size = max(1, min(15, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply Gaussian blur
        if sigma > 0:
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
        else:
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)

        return blurred

    except Exception as e:
        print(f"Blur Error: {e}")
        return img_array

def apply_sharpen(img_array, strength=1.0):
    """
    Apply sharpening to image using unsharp mask

    Args:
        img_array: Input image as numpy array (RGB)
        strength: Sharpening strength (0.0-3.0)

    Returns:
        Sharpened image as numpy array
    """
    try:
        # Convert to float
        img_float = img_array.astype(np.float32)

        # Create Gaussian blur for unsharp mask
        blurred = cv2.GaussianBlur(img_float, (0, 0), 3.0)

        # Create unsharp mask
        unsharp_mask = img_float - blurred

        # Apply sharpening
        sharpened = img_float + strength * unsharp_mask

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255)

        return sharpened.astype(np.uint8)

    except Exception as e:
        print(f"Sharpen Error: {e}")
        return img_array

def process_image(img_path, operation, **params):
    """
    Apply image processing operation to image

    Args:
        img_path: Path to input image
        operation: "enhance", "blur", or "sharpen"
        **params: Parameters for the operation

    Returns:
        Processed image as numpy array (RGB)
    """
    try:
        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError("Could not read image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply operation
        if operation == "enhance":
            contrast = params.get('contrast', 1.0)
            brightness = params.get('brightness', 0)
            processed = apply_enhancement(img_rgb, contrast, brightness)
        elif operation == "blur":
            kernel_size = params.get('kernel_size', 5)
            sigma = params.get('sigma', 0)
            processed = apply_blur(img_rgb, kernel_size, sigma)
        elif operation == "sharpen":
            strength = params.get('strength', 1.0)
            processed = apply_sharpen(img_rgb, strength)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return processed

    except Exception as e:
        print(f"Image Processing Error: {e}")
        # Return original image if processing fails
        img_bgr = cv2.imread(img_path)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
