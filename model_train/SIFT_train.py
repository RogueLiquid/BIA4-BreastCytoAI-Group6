import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import List, Iterable, Tuple
from PIL import Image
import pickle

# Make sure we can import helpers under final/
REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = Path(__file__).resolve().parents[1]
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from split_utils import load_splits

# For radiomics features
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from sklearn.cluster import KMeans, Birch
import skfuzzy as fuzz

import random

SEED = 124

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(SEED)

SIFT_N_FEATURES = 100
BOF_CLUSTERS = 500
SEGMENT_PARAMS = dict(n_segments=250, compactness=10, n_clusters=5)
CNN_NORM_MEAN = [0.485, 0.456, 0.406]
CNN_NORM_STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = (224, 224)

print(f"Random seed set to {SEED} for reproducibility")

device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

class ImageRecordDataset(Dataset):
    """Dataset backed by a list of records with file paths/labels/patient ids."""
    def __init__(self, records, transform=None, class_names=None):
        self.records = records
        self.transform = transform
        self.classes = class_names or ["benign", "malignant"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(str(rec.path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = rec.label
        if self.transform:
            img = self.transform(img)
        return img, label

def safe_read_rgb(path):
    img = cv2.imread(path)
    if img is None: 
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_radiomics_features(img_rgb):
    feats = {}
    # Resize for consistency (keeps enough detail for stats)
    H, W = 256, 256
    img = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    # Color stats (RGB + HSV)
    img_hsv = (rgb2hsv(img / 255.0) * 255.0).astype(np.uint8)
    for space_name, arr in [("rgb", img), ("hsv", img_hsv)]:
        for c, cname in enumerate(["c1","c2","c3"]):
            ch = arr[..., c].astype(np.float32)
            feats[f"{space_name}_{cname}_mean"] = ch.mean()
            feats[f"{space_name}_{cname}_std"]  = ch.std()

    # --- First-order intensity stats
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    feats["gray_mean"] = gray.mean()
    feats["gray_std"]  = gray.std()
    feats["gray_min"]  = gray.min()
    feats["gray_max"]  = gray.max()
    feats["gray_entropy"] = -np.sum(
        (np.histogram(gray, bins=32, range=(0,255), density=True)[0] + 1e-12) *
        np.log2(np.histogram(gray, bins=32, range=(0,255), density=True)[0] + 1e-12)
    )

    # --- GLCM texture (quantize to 8 levels for speed/robustness)
    q = np.floor(rescale_intensity(gray, in_range="image", out_range=(0,7))).astype(np.uint8)
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(q, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).ravel()
        feats[f"glcm_{prop}_mean"] = vals.mean()
        feats[f"glcm_{prop}_std"]  = vals.std()

    # --- LBP histogram (uniform)
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    for i, v in enumerate(hist):
        feats[f"lbp_u{i}"] = float(v)

    return feats

class RadiomicsDataset(Dataset):
    def __init__(self, root_dir, feature_cache_path=None):
        self.img_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for i, c in enumerate(self.classes):
            imgs = glob(os.path.join(root_dir, c, "*.png"))
            self.img_paths.extend(imgs)
            self.labels.extend([i]*len(imgs))
        
        if feature_cache_path and os.path.exists(feature_cache_path):
            print(f"Loading cached features from {feature_cache_path}")
            cache_data = np.load(feature_cache_path, allow_pickle=True)
            self.features = cache_data['features']
            self.labels = cache_data['labels'].tolist()
            print(f"Loaded {len(self.features)} cached features")
        else:
            print("Extracting radiomics features...")
            features_list = []
            valid_indices = []
            for idx, img_path in enumerate(tqdm(self.img_paths, desc="Extracting features")):
                img = safe_read_rgb(img_path)
                if img is None:
                    continue
                feats = extract_radiomics_features(img)
                features_list.append(list(feats.values()))
                valid_indices.append(idx)
            
            self.img_paths = [self.img_paths[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
            self.features = np.array(features_list, dtype=np.float32)
            
            if feature_cache_path:
                os.makedirs(os.path.dirname(feature_cache_path) if os.path.dirname(feature_cache_path) else '.', exist_ok=True)
                np.savez(feature_cache_path, features=self.features, labels=np.array(self.labels))
                print(f"Saved features cache to {feature_cache_path}")
        
        print(f"Dataset size: {len(self.features)}, Feature dimension: {self.features.shape[1]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = self.labels[idx]
        return feature, label

class FusionDataset(Dataset):
    def __init__(self, root_dir, transform=None, feature_cache_path=None):
        self.img_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for i, c in enumerate(self.classes):
            imgs = glob(os.path.join(root_dir, c, "*.png"))
            self.img_paths.extend(imgs)
            self.labels.extend([i]*len(imgs))
        
        self.transform = transform
        
        if feature_cache_path and os.path.exists(feature_cache_path):
            print(f"Loading cached radiomics features from {feature_cache_path}")
            cache_data = np.load(feature_cache_path, allow_pickle=True)
            self.radiomics_features = cache_data['features']
            cached_labels = cache_data['labels'].tolist()
            if len(self.radiomics_features) != len(self.img_paths):
                print("Warning: Cached features count doesn't match image count. Re-extracting...")
                self._extract_features(feature_cache_path)
            else:
                print(f"Loaded {len(self.radiomics_features)} cached features")
        else:
            self._extract_features(feature_cache_path)
        
        print(f"FusionDataset size: {len(self.img_paths)}, Radiomics feature dimension: {self.radiomics_features.shape[1]}")
    
    def _extract_features(self, feature_cache_path=None):
        print("Extracting radiomics features for fusion dataset...")
        features_list = []
        valid_indices = []
        
        for idx, img_path in enumerate(tqdm(self.img_paths, desc="Extracting features")):
            img = safe_read_rgb(img_path)
            if img is None:
                continue
            feats = extract_radiomics_features(img)
            features_list.append(list(feats.values()))
            valid_indices.append(idx)
        
        self.img_paths = [self.img_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        self.radiomics_features = np.array(features_list, dtype=np.float32)
        
        if feature_cache_path:
            os.makedirs(os.path.dirname(feature_cache_path) if os.path.dirname(feature_cache_path) else '.', exist_ok=True)
            np.savez(feature_cache_path, features=self.radiomics_features, labels=np.array(self.labels))
            print(f"Saved features cache to {feature_cache_path}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        
        radiomics_feature = torch.FloatTensor(self.radiomics_features[idx])
        
        label = self.labels[idx]
        
        return img, radiomics_feature, label

class RadiomicsMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ResNet50RadiomicsFusion(nn.Module):
    def __init__(
        self,
        radiomics_dim: int,
        resnet_feature_dim: int = 256,
        radiomics_reduced_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ResNet50 backbone
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet50(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, resnet_feature_dim)
        self.backbone = backbone

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        self.radiomics_branch = nn.Sequential(
            nn.Linear(radiomics_dim, radiomics_reduced_dim),
            nn.BatchNorm1d(radiomics_reduced_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        fusion_input_dim = resnet_feature_dim + radiomics_reduced_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, imgs, radiomics_features):
        img_feat = self.backbone(imgs)
        rad_feat = self.radiomics_branch(radiomics_features)
        fused = torch.cat([img_feat, rad_feat], dim=1)
        logits = self.fusion_head(fused)
        return logits

class RadiomicsRecordDataset(Dataset):
    """Dataset that computes radiomics features for records on the fly."""
    def __init__(self, records, feature_cache=None):
        self.records = records
        self.feature_cache = {} if feature_cache is None else feature_cache
        self.classes = ["benign", "malignant"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = str(rec.path)
        if path in self.feature_cache:
            feats = self.feature_cache[path]
        else:
            img = safe_read_rgb(path)
            if img is None:
                raise RuntimeError(f"Failed to read image at {path}")
            feats = np.array(list(extract_radiomics_features(img).values()), dtype=np.float32)
            self.feature_cache[path] = feats
        feature = torch.from_numpy(feats)
        return feature, rec.label

class FusionRecordDataset(Dataset):
    """Dataset that returns (image, radiomics_feature, label) for split records."""
    def __init__(self, records, transform=None, feature_cache=None, class_names=None):
        self.records = records
        self.transform = transform
        self.feature_cache = {} if feature_cache is None else feature_cache
        self.classes = class_names or ["benign", "malignant"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = safe_read_rgb(str(rec.path))
        if img is None:
            raise RuntimeError(f"Failed to read image at {rec.path}")
        if self.transform:
            img = self.transform(img)
        if rec.path in self.feature_cache:
            feats = self.feature_cache[rec.path]
        else:
            # use original RGB image for radiomics
            feats = np.array(list(extract_radiomics_features(img.cpu().permute(1,2,0).numpy() if torch.is_tensor(img) else img).values()), dtype=np.float32)
            self.feature_cache[rec.path] = feats
        rad_feature = torch.from_numpy(feats.astype(np.float32))
        return img, rad_feature, rec.label


class CNNFeatureExtractor(nn.Module):
    """Base class for CNN feature extractors."""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.features = None
        self.feature_dim = None

    def forward(self, x):
        with torch.no_grad():
            feats = self.features(x)
            if len(feats.shape) > 2:
                feats = feats.view(feats.size(0), -1)
        return feats


class ResNet50Extractor(CNNFeatureExtractor):
    def __init__(self):
        super().__init__("ResNet50")
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.features.eval()


class VGG16Extractor(CNNFeatureExtractor):
    def __init__(self):
        super().__init__("VGG16")
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
            *list(backbone.classifier.children())[:-1],
        )
        self.feature_dim = 4096
        self.features.eval()


class VGG19Extractor(CNNFeatureExtractor):
    def __init__(self):
        super().__init__("VGG19")
        backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
            *list(backbone.classifier.children())[:-1],
        )
        self.feature_dim = 4096
        self.features.eval()


class EfficientNetExtractor(CNNFeatureExtractor):
    def __init__(self):
        super().__init__("EfficientNet-B0")
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(backbone.features, backbone.avgpool)
        self.feature_dim = 1280
        self.features.eval()


class DenseNet121Extractor(CNNFeatureExtractor):
    def __init__(self):
        super().__init__("DenseNet121")
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 1024
        self.features.eval()


cnn_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CNN_NORM_MEAN, std=CNN_NORM_STD),
])


class HybridFeatureDataset(Dataset):
    """Dataset that combines SIFT-BoF features with CNN global features."""
    def __init__(self, images, sift_features, labels, transform, cnn_extractor, device, class_names=None):
        self.images = images
        self.sift_features = torch.FloatTensor(sift_features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.cnn_extractor = cnn_extractor.to(device)
        self.device = device
        self.classes = class_names or ["benign", "malignant"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sift_feat = self.sift_features[idx]
        image = Image.fromarray(self.images[idx])
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_feat = self.cnn_extractor(image_tensor).cpu().squeeze(0)
        hybrid_feat = torch.cat([sift_feat, cnn_feat], dim=0)
        label = self.labels[idx]
        return hybrid_feat, label


class HybridClassifier(nn.Module):
    """Classifier for concatenated SIFT+CNN features."""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        if input_dim >= 4000:
            hidden_dims = [2048, 1024, 512, 256]
            dropout_rates = [0.4, 0.3, 0.3, 0.2]
        elif input_dim >= 2500:
            hidden_dims = [1024, 512, 256]
            dropout_rates = [0.3, 0.3, 0.2]
        elif input_dim >= 1700:
            hidden_dims = [768, 384, 192]
            dropout_rates = [0.3, 0.2, 0.2]
        else:
            hidden_dims = [640, 320, 160]
            dropout_rates = [0.3, 0.2, 0.2]

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h, dr in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dr),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


SIFT_EXTRACTORS = {
    "SIFT_ResNet50": ResNet50Extractor,
    "SIFT_VGG16": VGG16Extractor,
    "SIFT_EfficientNetB0": EfficientNetExtractor,
    "SIFT_DenseNet121": DenseNet121Extractor,
}


SUPPORTED_SIFT_MODELS = set(SIFT_EXTRACTORS.keys())

def build_model(model_name, num_classes=2, radiomics_dim=None):
    """
    Only SIFT-hybrid models are supported in this file.
    They are constructed inside run_training (HybridClassifier + extractor),
    so callers should invoke run_training with one of SIFT_* names.
    """
    if model_name in SUPPORTED_SIFT_MODELS:
        raise ValueError(
            f"{model_name} is a SIFT hybrid; use run_training to build it (supported: {sorted(SUPPORTED_SIFT_MODELS)})"
        )
    raise ValueError(f"Unsupported model_name: {model_name}. Use one of {sorted(SUPPORTED_SIFT_MODELS)}")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        if len(batch) == 2:
            inputs, labels = batch
            if inputs.ndim == 4:  # image tensor
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            else:  # radiomics features only
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
        elif len(batch) == 3:
            imgs, rad_feats, labels = batch
            imgs, rad_feats, labels = imgs.to(device), rad_feats.to(device), labels.to(device)
            outputs = model(imgs, rad_feats)
        else:
            raise ValueError("Unexpected batch structure")

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                inputs, labels = batch
                if inputs.ndim == 4:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
            elif len(batch) == 3:
                imgs, rad_feats, labels = batch
                imgs, rad_feats, labels = imgs.to(device), rad_feats.to(device), labels.to(device)
                outputs = model(imgs, rad_feats)
            else:
                raise ValueError("Unexpected batch structure")
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, class_weights=None, weight_decay=1e-4, log_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_info = f" | LR: {new_lr:.2e}"
        if new_lr < old_lr:
            lr_info += " ⬇️"

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{lr_info}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses, train_accs, val_accs, best_val_loss

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score

def evaluate_on_testset(model, test_loader, dataset, model_name=None, epoch=None, save=True, save_dir=".", save_suffix=""):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if len(batch) == 2:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
            elif len(batch) == 3:
                imgs, rad_feats, labels = batch
                imgs, rad_feats, labels = imgs.to(device), rad_feats.to(device), labels.to(device)
                outputs = model(imgs, rad_feats)
            else:
                raise ValueError("Unexpected batch structure in test_loader")

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    class_report_str = classification_report(all_labels, all_preds, target_names=dataset.classes)
    print(class_report_str)

    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        "accuracy": acc,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": class_report_str,
    }

    # confusion matrix counts
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)
    cm_counts = {
        "tp": int(tp) if tp is not None else None,
        "fp": int(fp) if fp is not None else None,
        "tn": int(tn) if tn is not None else None,
        "fn": int(fn) if fn is not None else None,
        "matrix": cm.tolist(),
    }

    return metrics, cm_counts



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def compute_mean_std_from_records(records: List, resize: Tuple[int, int] = (224, 224)) -> Tuple[List[float], List[float]]:
    """Compute dataset mean/std over given records (RGB, 0-1 range)."""
    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    count = 0
    for rec in records:
        img = safe_read_rgb(str(rec.path))
        if img is None:
            continue
        img = cv2.resize(img, resize)
        img = img.astype(np.float32) / 255.0
        # H, W, C
        sum_c += img.sum(axis=(0, 1))
        sum_sq_c += (img ** 2).sum(axis=(0, 1))
        count += img.shape[0] * img.shape[1]
    mean = sum_c / count
    var = (sum_sq_c / count) - mean ** 2
    std = np.sqrt(var)
    return mean.tolist(), std.tolist()


# === SIFT + CNN hybrid helpers (from sift_concat.ipynb, adapted to split-based pipeline) ===


def slic_kmeans_birch_segmentation(image, n_segments=250, compactness=10, n_clusters=5):
    """
    Three-stage segmentation from the notebook:
      1) SLIC superpixels
      2) KMeans on superpixel descriptors
      3) BIRCH refinement
    """
    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )

    superpixel_features = []
    segment_ids = np.unique(segments)
    for segment_id in segment_ids:
        mask = segments == segment_id
        region = image[mask]
        mean_rgb = np.mean(region, axis=0)
        std_rgb = np.std(region, axis=0)
        area = np.sum(mask)
        features = np.concatenate([mean_rgb, std_rgb, [area]])
        superpixel_features.append(features)

    superpixel_features = np.array(superpixel_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    kmeans_labels = kmeans.fit_predict(superpixel_features)

    birch = Birch(n_clusters=n_clusters, threshold=0.5)
    final_labels = birch.fit_predict(superpixel_features)

    segmented_image = np.zeros_like(image)
    for idx, segment_id in enumerate(segment_ids):
        mask = segments == segment_id
        segmented_image[mask] = image[mask]

    return segmented_image


def extract_sift_features(image, n_features=SIFT_N_FEATURES):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros((n_features, 128), dtype=np.float32)
    if len(descriptors) < n_features:
        padding = np.zeros((n_features - len(descriptors), 128), dtype=np.float32)
        descriptors = np.vstack([descriptors, padding])
    else:
        descriptors = descriptors[:n_features]
    return descriptors.astype(np.float32)


def fuzzy_cmeans_clustering(descriptors, n_clusters=BOF_CLUSTERS, m=2, max_iter=100):
    """Return cluster centers from fuzzy C-means."""
    descriptors_t = descriptors.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        descriptors_t, n_clusters, m, error=0.005, maxiter=max_iter, init=None
    )
    cluster_membership = np.argmax(u, axis=0)
    return cntr, cluster_membership


def create_bof_histogram(membership, n_clusters=BOF_CLUSTERS):
    histogram, _ = np.histogram(membership, bins=np.arange(n_clusters + 1))
    histogram = histogram.astype(float)
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    return histogram


def load_and_segment_records(records, img_size=DEFAULT_IMG_SIZE, seg_params=SEGMENT_PARAMS):
    """Read, resize, and segment images for a list of split records."""
    images = []
    labels = []
    for rec in records:
        img = safe_read_rgb(str(rec.path))
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = slic_kmeans_birch_segmentation(img, **seg_params)
        images.append(img)
        labels.append(rec.label)
    return np.array(images), np.array(labels)


def build_bof_features(images, centers=None, n_clusters=BOF_CLUSTERS):
    """Build BoF histograms for images, optionally using provided FCM centers."""
    descriptors_list = [extract_sift_features(img) for img in images]
    if centers is None:
        all_descriptors = np.vstack(descriptors_list) if len(descriptors_list) else np.zeros((0, 128))
        centers, _ = fuzzy_cmeans_clustering(all_descriptors, n_clusters=n_clusters)

    bof_features = []
    for descriptors in descriptors_list:
        # use precomputed centers to assign memberships
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            descriptors.T, centers, m=2, error=0.005, maxiter=100
        )
        membership = np.argmax(u, axis=0)
        bof_features.append(create_bof_histogram(membership, n_clusters))

    return np.array(bof_features), centers


def prepare_sift_sets(train_records, val_records, test_records, img_size=DEFAULT_IMG_SIZE, cache_path: Path = None):
    """Prepare segmented images and BoF features for train/val/test splits with optional caching."""
    if cache_path and cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            print(f"Loaded cached SIFT features from {cache_path}")
            return {
                "train": {"images": data["train_images"], "labels": data["train_labels"], "bof": data["train_bof"]},
                "val": {"images": data["val_images"], "labels": data["val_labels"], "bof": data["val_bof"]},
                "test": {"images": data["test_images"], "labels": data["test_labels"], "bof": data["test_bof"]},
                "vocab": data["vocab"],
            }
        except Exception as e:
            print(f"Failed to load cache at {cache_path}, recomputing. Error: {e}")

    train_images, train_labels = load_and_segment_records(train_records, img_size)
    val_images, val_labels = load_and_segment_records(val_records, img_size)
    test_images, test_labels = load_and_segment_records(test_records, img_size)

    train_bof, vocab = build_bof_features(train_images)
    val_bof, _ = build_bof_features(val_images, centers=vocab)
    test_bof, _ = build_bof_features(test_images, centers=vocab)

    result = {
        "train": {"images": train_images, "labels": train_labels, "bof": train_bof},
        "val": {"images": val_images, "labels": val_labels, "bof": val_bof},
        "test": {"images": test_images, "labels": test_labels, "bof": test_bof},
        "vocab": vocab,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train_images=train_images,
            train_labels=train_labels,
            train_bof=train_bof,
            val_images=val_images,
            val_labels=val_labels,
            val_bof=val_bof,
            test_images=test_images,
            test_labels=test_labels,
            test_bof=test_bof,
            vocab=vocab,
        )
        print(f"Saved SIFT cache to {cache_path}")

    return result


def run_training(
    model_name,
    *,
    splits_path,
    epochs=20,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    train_transforms=None,
    val_transforms=None,
    save_root="."
):
    """Main entry point for training a model using a precomputed splits JSON."""
    repo_root = REPO_ROOT
    splits_file = Path(splits_path)
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    records, splits, meta = load_splits(splits_file, repo_root=repo_root)
    class_names = meta.get("class_names", ["benign", "malignant"])
    split_mode = meta.get("split_mode", "unknown")
    magnification = meta.get("magnification", "unknown")

    print(f"Using splits from: {splits_file}")
    print(f"Split mode: {split_mode} | magnification: {magnification} | total folds: {len(splits)}")
    sift_extractor_cls = SIFT_EXTRACTORS.get(model_name)
    is_sift_hybrid = sift_extractor_cls is not None
    sift_cache_root = FINAL_ROOT / "SIFT_cache" / splits_file.stem

    # Determine radiomics dim if needed (use first training record of first split)
    radiomics_dim = None
    if model_name in ["RadiomicsMLP", "ResNet50RadiomicsFusion"]:
        first_train_idx = splits[0]["train"][0]
        sample_path = str(records[first_train_idx].path)
        img = safe_read_rgb(sample_path)
        if img is None:
            raise RuntimeError(f"Failed to read sample image for radiomics dim at {sample_path}")
        radiomics_dim = len(extract_radiomics_features(img))

    results = []

    # decide normalization strategy
    pretrained_models = {
        "VGG16_PT_FT",
        "ResNet50_PT_FT",
        "DenseNet121_PT_FT",
        "MobileNetV2_PT_FT",
        "EfficientNetB0_PT_FT",
        "ViT_B16_PT_FT",
        "AlexNet_PT_FT",
        "GoogLeNet_PT_FT",
        "ResNet50RadiomicsFusion",
    }
    use_imagenet_norm = model_name in pretrained_models

    stats_path = FINAL_ROOT / "splits" / f"{splits_file.stem}_stats.json"
    mean_used = IMAGENET_MEAN
    std_used = IMAGENET_STD
    if is_sift_hybrid:
        print("Using SIFT+CNN hybrid pipeline (ImageNet normalization for CNN branch)")
        print(f"SIFT features: {SIFT_N_FEATURES} per image | BoF clusters: {BOF_CLUSTERS}")
    elif not use_imagenet_norm and model_name != "RadiomicsMLP":
        # compute or load dataset mean/std using all train records across folds
        train_indices = set()
        for split in splits:
            train_indices.update(split["train"])
        train_records_for_stats = [records[i] for i in sorted(train_indices)]
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                mean_used = stats["mean"]
                std_used = stats["std"]
                print(f"Loaded cached mean/std from {stats_path}")
            except Exception:
                mean_used, std_used = compute_mean_std_from_records(train_records_for_stats)
        else:
            mean_used, std_used = compute_mean_std_from_records(train_records_for_stats)
            stats_payload = {
                "mean": mean_used,
                "std": std_used,
                "num_images": len(train_records_for_stats),
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "splits_file": str(splits_file),
            }
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
            print(f"Saved mean/std to {stats_path}")

    if not is_sift_hybrid:
        norm_label = "ImageNet" if use_imagenet_norm else "Dataset"
        print(f"Using {norm_label} normalization")
        print("Mean:", mean_used)
        print("Std:", std_used)

    results = []

    for fold_index, split in enumerate(splits):
        train_records = [records[i] for i in split["train"]]
        val_records = [records[i] for i in split["val"]]
        test_records = [records[i] for i in split["test"]]

        print(f"\n=== Fold {fold_index+1}/{len(splits)} ===")

        if is_sift_hybrid:
            cache_path = sift_cache_root / f"fold{fold_index+1}.npz"
            sift_data = prepare_sift_sets(
                train_records,
                val_records,
                test_records,
                img_size=DEFAULT_IMG_SIZE,
                cache_path=cache_path,
            )
            extractor = sift_extractor_cls()
            extractor.eval()
            input_dim = BOF_CLUSTERS + extractor.feature_dim
            model = HybridClassifier(input_dim=input_dim, num_classes=len(class_names))
            print(f"SIFT data shapes - train: {sift_data['train']['bof'].shape}, val: {sift_data['val']['bof'].shape}, test: {sift_data['test']['bof'].shape}")
        else:
            model = build_model(model_name, num_classes=2, radiomics_dim=radiomics_dim)

        if train_transforms is not None:
            train_tfms = train_transforms
        else:
            train_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_used, std=std_used),
            ])

        if val_transforms is not None:
            test_tfms = val_transforms
        else:
            test_tfms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_used, std=std_used),
            ])

        feature_cache = {}
        if model_name == "RadiomicsMLP":
            train_ds = RadiomicsRecordDataset(train_records, feature_cache=feature_cache)
            val_ds = RadiomicsRecordDataset(val_records, feature_cache=feature_cache)
            test_ds = RadiomicsRecordDataset(test_records, feature_cache=feature_cache)
        elif model_name == "ResNet50RadiomicsFusion":
            train_ds = FusionRecordDataset(train_records, transform=train_tfms, feature_cache=feature_cache, class_names=class_names)
            val_ds = FusionRecordDataset(val_records, transform=test_tfms, feature_cache=feature_cache, class_names=class_names)
            test_ds = FusionRecordDataset(test_records, transform=test_tfms, feature_cache=feature_cache, class_names=class_names)
        elif is_sift_hybrid:
            extractor.eval()
            train_ds = HybridFeatureDataset(
                sift_data["train"]["images"],
                sift_data["train"]["bof"],
                sift_data["train"]["labels"],
                cnn_transform,
                extractor,
                device,
                class_names=class_names,
            )
            val_ds = HybridFeatureDataset(
                sift_data["val"]["images"],
                sift_data["val"]["bof"],
                sift_data["val"]["labels"],
                cnn_transform,
                extractor,
                device,
                class_names=class_names,
            )
            test_ds = HybridFeatureDataset(
                sift_data["test"]["images"],
                sift_data["test"]["bof"],
                sift_data["test"]["labels"],
                cnn_transform,
                extractor,
                device,
                class_names=class_names,
            )
        else:
            train_ds = ImageRecordDataset(train_records, transform=train_tfms, class_names=class_names)
            val_ds = ImageRecordDataset(val_records, transform=test_tfms, class_names=class_names)
            test_ds = ImageRecordDataset(test_records, transform=test_tfms, class_names=class_names)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        num_classes = len(class_names)
        if num_classes != 2:
            raise ValueError(f"Expected binary classification, got {num_classes} classes: {class_names}")

        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for rec in train_records:
            class_counts[rec.label] += 1

        if (class_counts == 0).any():
            raise ValueError(f"Class count contains zero for classes {class_names}: {class_counts.tolist()}")

        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        print("\nClass distribution in training set:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {int(class_counts[i])} samples")
        print(f"\nClass weights: {class_weights.tolist()}")
        print(f"Total training samples: {len(train_ds)} (records source: {len(train_records)})")

        save_suffix = ""
        print("Model save suffix: '' (no offline augmentation)")

        # paths and logging
        base_dir = Path(FINAL_ROOT) / "SIFT_weights" / model_name / split_mode
        base_dir.mkdir(parents=True, exist_ok=True)
        model_path = base_dir / f"{model_name}_{split_mode}_fold{fold_index+1}.pth"
        log_path = base_dir / f"{model_name}_{split_mode}_fold{fold_index+1}_train_log.txt"
        cm_json_path = base_dir / "confusion_matrices.json"
        best_meta_path = base_dir / "best_test_metrics.json"
        overall_best_path = base_dir / "best_overall.json"
        overall_best_model_path = base_dir / f"{model_name}_{split_mode}_best.pth"

        # write training log header
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"model: {model_name}\n")
            f.write(f"fold: {fold_index+1}\n")
            f.write(f"split_mode: {split_mode}\n")
            f.write(f"magnification: {magnification}\n")
            f.write(f"timestamp: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"epochs: {epochs}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"lr: {lr}\n")
            f.write(f"weight_decay: {weight_decay}\n")
            f.write("\nEpoch,TrainLoss,TrainAcc,ValLoss,ValAcc\n")

        train_losses, val_losses, train_accs, val_accs, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            class_weights=class_weights,
            weight_decay=weight_decay,
            log_path=log_path
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\nbest_val_loss: {best_val_loss:.6f}\n")

        metrics, cm = evaluate_on_testset(
            model,
            test_loader,
            test_ds,
            model_name=model_name,
            epoch=epochs,
            save=False,
            save_dir=save_root,
            save_suffix=f"{save_suffix}_fold{fold_index+1}"
        )

        # confusion matrices aggregation
        cm_entry = {
            "fold": fold_index + 1,
            "split_mode": split_mode,
            "magnification": magnification,
            "accuracy": metrics["accuracy"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "best_val_loss": best_val_loss,
            "confusion_matrix": cm["matrix"],
            "tp": cm["tp"],
            "fp": cm["fp"],
            "tn": cm["tn"],
            "fn": cm["fn"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if cm_json_path.exists():
            try:
                cm_data = json.loads(cm_json_path.read_text(encoding="utf-8"))
                if not isinstance(cm_data, list):
                    cm_data = []
            except Exception:
                cm_data = []
        else:
            cm_data = []
        cm_data.append(cm_entry)
        cm_json_path.write_text(json.dumps(cm_data, indent=2), encoding="utf-8")

        # save model only if best test accuracy for this fold
        if best_meta_path.exists():
            try:
                best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            except Exception:
                best_meta = {}
        else:
            best_meta = {}
        fold_key = str(fold_index + 1)
        prev_best = best_meta.get(fold_key, {}).get("accuracy", -1)
        if metrics["accuracy"] > prev_best:
            torch.save(model.state_dict(), model_path)
            best_meta[fold_key] = {
                "accuracy": metrics["accuracy"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "best_val_loss": best_val_loss,
                "path": str(model_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            best_meta_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")
            print(f"Saved best model for fold {fold_index+1} to {model_path}")
        else:
            print(f"Model not saved (test accuracy {metrics['accuracy']:.4f} <= best {prev_best:.4f})")

        # track overall best across folds for this split_mode
        if overall_best_path.exists():
            try:
                overall_best = json.loads(overall_best_path.read_text(encoding="utf-8"))
            except Exception:
                overall_best = {}
        else:
            overall_best = {}
        overall_acc = overall_best.get("accuracy", -1)
        if metrics["accuracy"] > overall_acc:
            torch.save(model.state_dict(), overall_best_model_path)
            overall_best.update(
                {
                    "accuracy": metrics["accuracy"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "best_val_loss": best_val_loss,
                    "fold": fold_index + 1,
                    "path": str(overall_best_model_path),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "split_mode": split_mode,
                    "confusion_matrix": cm["matrix"],
                    "tp": cm["tp"],
                    "fp": cm["fp"],
                    "tn": cm["tn"],
                    "fn": cm["fn"],
                }
            )
            overall_best_path.write_text(json.dumps(overall_best, indent=2), encoding="utf-8")
            print(f"Updated overall best ({split_mode}) model to fold {fold_index+1} at {overall_best_model_path}")

        results.append(
            {
                "fold": fold_index + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "class_weights": class_weights.tolist(),
                "save_dir": str(base_dir),
                "test_metrics": metrics,
                "confusion_counts": cm,
            }
        )

    return results

if __name__ == "__main__":
    splits_to_use = [
        FINAL_ROOT / "splits" / "splits_patient_400X_5fold_seed124.json",
        FINAL_ROOT / "splits" / "splits_image_400X_5fold_seed124.json",
    ]
    model_names = sorted(SIFT_EXTRACTORS.keys())

    for splits_file in splits_to_use:
        for mname in model_names:
            print(f"\n==== Running {mname} on splits {splits_file.name} ====")
            try:
                run_training(mname, splits_path=splits_file)
            except Exception as e:
                print(f"Skipped {mname} on {splits_file.name} due to error: {e}")
