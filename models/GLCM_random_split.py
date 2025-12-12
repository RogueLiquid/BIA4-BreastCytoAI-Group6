#!/usr/bin/env python3
"""
glcm_densenet201_full_pipeline.py

Single script that:

  1) Reads original BreaKHis_v1 histology_slides/breast structure.
  2) Uses one magnification (MAGNIFICATION = "40X" by default).
  3) Computes / caches 3-channel GLCM features.
  4) Does 5 patient-based 70/30 splits (GroupShuffleSplit, as in the paper).
  5) For EACH split:
       - Trains DenseNet201 baseline with EXACT hyper-params you provided:
           * ImageNet pretrained
           * input size 224x224
           * SGD(lr=1e-4, momentum=0.9)
           * batch size 10, epochs 6
           * CrossEntropyLoss
           * fine-tune ALL layers
       - Builds a conv1 multi-block extractor on the trained model.
       - Extracts fused [deep_128 || glcm_66] features.
       - Trains an RBF SVM (grid-search C, gamma) and evaluates.

IMPORTANT:
  - Training code & hyperparameters are copied from your densenet201_finetune.py
    with minimal necessary changes (only to accept arbitrary train/val record
    lists instead of pre-made train/validation folders).

Run from repository root:

    python -m models.glcm_densenet201_full_pipeline
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from skimage.feature import graycomatrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    GroupShuffleSplit,
)

# ============================================================
# CONFIG
# ============================================================

# main_folder/models/...py -> main_folder
REPO_ROOT = Path(__file__).resolve().parent.parent

# Original BreaKHis_v1 root (update if your path is slightly different)
IMAGE_ROOT = (
    REPO_ROOT
    / "BreakHis"
    / "BreaKHis_v1"
    / "BreaKHis_v1"
    / "histology_slides"
    / "breast"
)

# Magnification to use ("40X", "100X", "200X", "400X")
MAGNIFICATION = "400X"

# Where to cache GLCM features
GLCM_FEATURE_ROOT = REPO_ROOT / "BreakHis_400X_glcm"

# Binary class names correspond to top-level folders
CLASS_NAMES = ["benign", "malignant"]  # 0 -> benign, 1 -> malignant

# GLCM parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
GLCM_LEVELS = 256

# SVM kernel
SVM_KERNEL = "rbf"

# ImageNet normalization (shared by training & feature extraction)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training hyperparameters (EXACTLY as in your densenet201_finetune.py)
BATCH_SIZE = 10
EPOCHS = 5
LR = 5e-5
MOMENTUM = 0.9

# Random seed
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)


# ============================================================
# Data structures
# ============================================================

@dataclass
class ImageRecord:
    img_path: Path
    label: int         # 0: benign, 1: malignant
    patient_id: str    # patient folder name, e.g. SOB_B_A_14-22549AB


# ============================================================
# Collect ALL images for a given magnification
# ============================================================

def collect_all_image_records() -> List[ImageRecord]:
    """
    Walk the original BreaKHis_v1 structure and collect all images for
    the chosen magnification (MAGNIFICATION).

    Expected structure under IMAGE_ROOT:

      benign/
        SOB/
          adenosis/
            SOB_B_A_14-22549AB/
              40X/*.png
              100X/*.png
              200X/*.png
              400X/*.png
          fibroadenoma/...
      malignant/
        SOB/
          ductal_carcinoma/...
    """
    records: List[ImageRecord] = []

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"IMAGE_ROOT not found: {IMAGE_ROOT}")

    for class_name in ("benign", "malignant"):
        class_root = IMAGE_ROOT / class_name / "SOB"
        if not class_root.exists():
            print(f"[WARN] Class root does not exist: {class_root}")
            continue

        label = CLASS_NAMES.index(class_name)

        for subtype_dir in class_root.iterdir():
            if not subtype_dir.is_dir():
                continue

            for patient_dir in subtype_dir.iterdir():
                if not patient_dir.is_dir():
                    continue

                patient_id = patient_dir.name

                mag_dir = patient_dir / MAGNIFICATION
                if not mag_dir.exists():
                    # Patient may not have this magnification
                    continue

                for img_path in mag_dir.glob("*.png"):
                    records.append(
                        ImageRecord(
                            img_path=img_path,
                            label=label,
                            patient_id=patient_id,
                        )
                    )

    if not records:
        raise RuntimeError(
            f"No images found under {IMAGE_ROOT} for MAGNIFICATION={MAGNIFICATION}"
        )

    return records


# ============================================================
# GLCM 22 Haralick-like features
# ============================================================

def _compute_glcm_22_features(p: np.ndarray) -> np.ndarray:
    """
    Compute 22 Haralick-like features for a normalized GLCM p (shape [Ng, Ng]).
    """
    p = p.astype(np.float64)
    eps = 1e-12

    Ng = p.shape[0]
    assert p.shape == (Ng, Ng)
    p = p / (p.sum() + eps)

    i_idx, j_idx = np.indices((Ng, Ng))

    px = p.sum(axis=1)
    py = p.sum(axis=0)

    i_mean = (i_idx * p).sum()
    j_mean = (j_idx * p).sum()
    i_std = np.sqrt(((i_idx - i_mean) ** 2 * p).sum())
    j_std = np.sqrt(((j_idx - j_mean) ** 2 * p).sum())

    # px+py and px-py
    k_range_sum = np.arange(2, 2 * Ng + 1)
    px_plus_y = np.zeros(2 * Ng + 1)
    for i in range(Ng):
        for j in range(Ng):
            px_plus_y[i + j + 2] += p[i, j]

    k_range_diff = np.arange(0, Ng)
    px_minus_y = np.zeros(Ng)
    for i in range(Ng):
        for j in range(Ng):
            px_minus_y[abs(i - j)] += p[i, j]

    def _entropy(x: np.ndarray) -> float:
        x = x[x > 0]
        return -(x * np.log2(x + eps)).sum()

    HXY = _entropy(p.flatten())
    HX = _entropy(px)
    HY = _entropy(py)

    px_py = px[:, None] * py[None, :]
    HXY1 = -((p + eps) * np.log2(px_py + eps)).sum()
    HXY2 = -(px_py * np.log2(px_py + eps)).sum()

    features: List[float] = []

    # 1) Autocorrelation
    autoc = (i_idx * j_idx * p).sum()
    features.append(autoc)

    # 2) Contrast
    contrast = 0.0
    for n in range(Ng):
        mask = np.abs(i_idx - j_idx) == n
        contrast += n**2 * p[mask].sum()
    features.append(contrast)

    # 3) Correlation (Haralick)
    if i_std > 0 and j_std > 0:
        corr1 = ((i_idx * j_idx * p).sum() - i_mean * j_mean) / (i_std * j_std)
    else:
        corr1 = 0.0
    features.append(corr1)

    # 4) Correlation (alternative) – same as corr1 here
    corr2 = corr1
    features.append(corr2)

    # 5) Cluster prominence
    cluster_prom = (((i_idx + j_idx - i_mean - j_mean) ** 4) * p).sum()
    features.append(cluster_prom)

    # 6) Cluster shade
    cluster_shade = (((i_idx + j_idx - i_mean - j_mean) ** 3) * p).sum()
    features.append(cluster_shade)

    # 7) Dissimilarity
    dissim = (np.abs(i_idx - j_idx) * p).sum()
    features.append(dissim)

    # 8) Energy (angular second moment)
    energy = (p ** 2).sum()
    features.append(energy)

    # 9) Entropy
    entropy = HXY
    features.append(entropy)

    # 10) Homogeneity (1) – inverse difference moment
    homog1 = (p / (1.0 + (i_idx - j_idx) ** 2)).sum()
    features.append(homog1)

    # 11) Homogeneity (2) – 1/(1+|i-j|)
    homog2 = (p / (1.0 + np.abs(i_idx - j_idx))).sum()
    features.append(homog2)

    # 12) Maximum probability
    max_prob = p.max()
    features.append(max_prob)

    # 13) Sum of squares (variance)
    sum_squares = (((i_idx - i_mean) ** 2) * p).sum()
    features.append(sum_squares)

    # 14) Sum average
    sum_avg = (k_range_sum * px_plus_y[2:2 * Ng + 2]).sum()
    features.append(sum_avg)

    # 16) Sum entropy (compute first)
    sum_entropy = _entropy(px_plus_y[2:2 * Ng + 2])

    # 15) Sum variance
    sum_var = ((k_range_sum - sum_entropy) ** 2 * px_plus_y[2:2 * Ng + 2]).sum()
    features.append(sum_var)

    # 16) Sum entropy
    features.append(sum_entropy)

    # 17) Difference variance
    diff_var = np.sum((np.arange(Ng) ** 2) * px_minus_y)
    features.append(diff_var)

    # 18) Difference entropy
    diff_entropy = _entropy(px_minus_y)
    features.append(diff_entropy)

    # 19) Normalized inverse difference (NID)
    denom = 1.0 + (np.abs(i_idx - j_idx) / (Ng**2))
    nid = (p / denom).sum()
    features.append(nid)

    # 20) Normalized inverse difference moment (NIDM)
    denom2 = 1.0 + ((i_idx - j_idx) ** 2 / Ng**2)
    nidm = (p / denom2).sum()
    features.append(nidm)

    # 21) Information measure of correlation 1
    if max(HX, HY) > 0:
        icorr1 = (HXY - HXY1) / max(HX, HY)
    else:
        icorr1 = 0.0
    features.append(icorr1)

    # 22) Information measure of correlation 2
    tmp = max(0.0, 1.0 - math.exp(-2.0 * (HXY2 - HXY)))
    icorr2 = math.sqrt(tmp)
    features.append(icorr2)

    return np.array(features, dtype=np.float64)


def compute_3channel_glcm_features(image: np.ndarray) -> np.ndarray:
    """
    3-channel GLCM (R,G,B) with 4 directions, distance=1, Ng=256.
    22 Haralick-like features per angle, averaged over angles.
    Total 66-dim vector.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3) for GLCM")

    img_uint8 = image.astype(np.uint8)
    feats_all_channels: List[float] = []

    for ch in range(3):  # R, G, B
        channel = img_uint8[..., ch]

        feats_per_angle: List[np.ndarray] = []
        for angle in GLCM_ANGLES:
            glcm = graycomatrix(
                channel,
                distances=GLCM_DISTANCES,
                angles=[angle],
                levels=GLCM_LEVELS,
                symmetric=True,
                normed=True,
            )
            p = glcm[:, :, 0, 0]
            feats_22 = _compute_glcm_22_features(p)
            feats_per_angle.append(feats_22)

        feats_per_angle = np.stack(feats_per_angle, axis=0)
        feats_mean = feats_per_angle.mean(axis=0)
        feats_all_channels.extend(feats_mean.tolist())

    return np.array(feats_all_channels, dtype=np.float64)


# ============================================================
# GLCM caching utilities
# ============================================================

def get_glcm_feature_path(img_path: Path) -> Path:
    """
    Map image path under IMAGE_ROOT to cached GLCM path under GLCM_FEATURE_ROOT.
    """
    rel = img_path.relative_to(IMAGE_ROOT)
    feat_path = GLCM_FEATURE_ROOT / rel.parent / f"{img_path.stem}_glcm.npy"
    return feat_path


def get_glcm_features_cached(img_path: Path, img_np: np.ndarray | None = None) -> np.ndarray:
    """
    Load GLCM features from cache if present; otherwise compute and save.
    """
    feat_path = get_glcm_feature_path(img_path)

    if feat_path.exists():
        return np.load(feat_path)

    if img_np is None:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

    glcm_feats = compute_3channel_glcm_features(img_np)

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, glcm_feats)

    return glcm_feats


# ============================================================
# DenseNet201 conv1 multi-block extractor (no weight loading)
# ============================================================

class DenseNet201MultiBlockConv1Extractor(nn.Module):
    """
    Wraps an ALREADY-TRAINED DenseNet201 model's .features and exposes
    GAP-pooled conv1 outputs from selected denselayers in denseblock4.

    IMPORTANT: this class DOES NOT load weights; it uses the features module
    passed in (so each fold uses its own fine-tuned model).
    """

    BLOCK_INDEX = {
        "block4": 4,
        "block6": 6,
        "block14": 14,
        "block19": 19,
        "block22": 22,
        "block23": 23,
    }

    def __init__(self, features: nn.Sequential, device: torch.device | None = None):
        super().__init__()

        # Use the features from the trained DenseNet201
        self.features = features

        self.block_names = list(self.BLOCK_INDEX.keys())
        self._feat_dict: dict[str, torch.Tensor | None] = {
            name: None for name in self.block_names
        }

        denseblock4 = self.features.denseblock4
        for name, idx in self.BLOCK_INDEX.items():
            layer = getattr(denseblock4, f"denselayer{idx}")
            conv1 = layer.conv1
            conv1.register_forward_hook(self._make_hook(name))

        # Preprocessing (must match training)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        if device is not None:
            self.to(device)

        self.eval()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._feat_dict[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        for k in self._feat_dict:
            self._feat_dict[k] = None

        _ = self.features(x)

        out: dict[str, torch.Tensor] = {}
        for name, feat in self._feat_dict.items():
            if feat is None:
                raise RuntimeError(f"No feature captured for {name}")
            pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            out[name] = pooled
        return out


# ============================================================
# BreakHisDataset (from your training script, minimal change)
# ============================================================

class BreakHisDataset(Dataset):
    """
    Dataset wrapping a list of ImageRecord objects.

    Transforms:
      - Resize to 224 x 224 (as required by DenseNet201)
      - ToTensor
      - Normalize with ImageNet mean/std
    """

    def __init__(self, records: List[ImageRecord]):
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img = Image.open(rec.img_path).convert("RGB")
        x = self.transform(img)
        y = rec.label
        pid = rec.patient_id
        return x, y, pid


# ============================================================
# Patient-level accuracy (same definition as before)
# ============================================================

def patient_level_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
) -> float:
    assert y_true.shape == y_pred.shape == patient_ids.shape

    acc_per_patient: Dict[str, float] = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        acc_per_patient[pid] = correct / total if total > 0 else 0.0

    return float(np.mean(list(acc_per_patient.values())))


# ============================================================
# Training / evaluation loops (copied from your script)
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        if step % 20 == 0:
            print(
                f"    [Epoch {epoch} | Step {step}/{len(loader)}] "
                f"loss={loss.item():.4f}"
            )

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_pids: List[str] = []

    for images, labels, pids in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_pids.extend(pids)

    val_loss = running_loss / running_total
    img_acc = running_correct / running_total

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    pids_arr = np.array(all_pids)

    pat_acc = patient_level_accuracy(y_true, y_pred, pids_arr)

    return val_loss, img_acc, pat_acc


def train_densenet201_for_split(
    train_records: List[ImageRecord],
    val_records: List[ImageRecord],
    device: torch.device,
) -> nn.Module:
    """
    Train DenseNet201 on THIS split using exactly your original settings.
    Returns the model loaded with the best validation (image-level) weights.
    """
    train_dataset = BreakHisDataset(train_records)
    val_dataset = BreakHisDataset(val_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("    Building DenseNet201 (ImageNet pretrained) ...")
    try:
        from torchvision.models import DenseNet201_Weights
        weights = DenseNet201_Weights.IMAGENET1K_V1
        model = models.densenet201(weights=weights)
    except Exception:
        model = models.densenet201(pretrained=True)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
    )

    best_val_img_acc = 0.0
    best_state_dict = None

    print("    Starting training ...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_img_acc, val_pat_acc = evaluate_classifier(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - start_time
        print(
            f"    Epoch {epoch}/{EPOCHS} ({epoch_time:.1f}s) "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, img_acc={val_img_acc:.4f}, "
            f"pat_acc={val_pat_acc:.4f}"
        )

        if val_img_acc > best_val_img_acc:
            best_val_img_acc = val_img_acc
            best_state_dict = model.state_dict().copy()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"    Loaded best DenseNet201 weights (val img acc={best_val_img_acc:.4f})")
    else:
        print("    WARNING: no best state dict recorded.")

    model.eval()
    return model


# ============================================================
# Feature extraction for a split (using trained DenseNet201)
# ============================================================

def extract_fused_features_multi_block(
    records: List[ImageRecord],
    device: torch.device,
    densenet_fe: DenseNet201MultiBlockConv1Extractor,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract fused features (deep + GLCM) for each record for ALL blocks from
    the given conv1 extractor (which uses trained DenseNet features).
    """
    X_blocks_lists: Dict[str, List[np.ndarray]] = {
        name: [] for name in densenet_fe.block_names
    }
    ys: List[int] = []
    pids: List[str] = []

    for idx, rec in enumerate(records, start=1):
        img = Image.open(rec.img_path).convert("RGB")
        img_np = np.array(img)

        glcm_feats = get_glcm_features_cached(rec.img_path, img_np)

        x_tensor = densenet_fe.preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            deep_feats_dict = densenet_fe(x_tensor)

        for name in densenet_fe.block_names:
            deep_vec = deep_feats_dict[name].cpu().numpy().reshape(-1)
            fused = np.concatenate([deep_vec, glcm_feats], axis=0)  # (194,)
            X_blocks_lists[name].append(fused)

        ys.append(rec.label)
        pids.append(rec.patient_id)

        if idx % 50 == 0:
            print(f"      [feat] processed {idx}/{len(records)} images")

    X_blocks: Dict[str, np.ndarray] = {}
    for name, vec_list in X_blocks_lists.items():
        X_blocks[name] = np.stack(vec_list, axis=0)

    y = np.array(ys, dtype=np.int64)
    pids_arr = np.array(pids)
    return X_blocks, y, pids_arr


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"IMAGE_ROOT        = {IMAGE_ROOT}")
    print(f"MAGNIFICATION     = {MAGNIFICATION}")
    print(f"GLCM_FEATURE_ROOT = {GLCM_FEATURE_ROOT}")

    # --------------------------------------------------------
    # 1) Collect all records for this magnification
    # --------------------------------------------------------
    print("\n[1] Collecting image records ...")
    records = collect_all_image_records()
    print(f"  Total images: {len(records)}")

    class_counts = Counter(r.label for r in records)
    print(
        "  Class counts (0=benign,1=malignant):",
        {k: class_counts[k] for k in sorted(class_counts)},
    )

    per_class_patients: Dict[int, set[str]] = defaultdict(set)
    for r in records:
        per_class_patients[r.label].add(r.patient_id)
    print("  Patients per class:")
    for label in sorted(per_class_patients):
        print(
            f"    {CLASS_NAMES[label]}: {len(per_class_patients[label])} patients"
        )

    # --------------------------------------------------------
    # 2) Precompute GLCM for all images (so splits share cache)
    # --------------------------------------------------------
    print("\n[2] Ensuring GLCM features are cached ...")
    for idx, rec in enumerate(records, start=1):
        _ = get_glcm_features_cached(rec.img_path)
        if idx % 100 == 0:
            print(f"  [GLCM] {idx}/{len(records)} images cached")
    print("  GLCM caching done.")

    # --------------------------------------------------------
    # 3) Prepare patient-based splits
    # --------------------------------------------------------
    print("\n[3] Preparing patient-based GroupShuffleSplit (70/30, 5 splits) ...")
    labels = np.array([r.label for r in records], dtype=np.int64)
    groups = np.array([r.patient_id for r in records])

    gss = GroupShuffleSplit(
        n_splits=5,
        test_size=0.20,
        random_state=GLOBAL_SEED,
    )

    splits = list(gss.split(labels, labels, groups=groups))
    print(f"  Unique patients: {len(np.unique(groups))}")
    for fold_idx, (tr, te) in enumerate(splits, start=1):
        train_pats = len(np.unique(groups[tr]))
        test_pats = len(np.unique(groups[te]))
        print(
            f"  Fold {fold_idx}: train images={len(tr)}, test images={len(te)}, "
            f"train patients={train_pats}, test patients={test_pats}"
        )

    # --------------------------------------------------------
    # 4) For each block, keep lists of fold accuracies
    # --------------------------------------------------------
    all_block_img_accs: Dict[str, List[float]] = defaultdict(list)
    all_block_pat_accs: Dict[str, List[float]] = defaultdict(list)

    # --------------------------------------------------------
    # 5) Run folds
    # --------------------------------------------------------
    print("\n[4] Running 5 folds ...")

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n====================== Fold {fold_idx} ======================")

        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        print(f"  Train images: {len(train_records)}")
        print(f"  Test images:  {len(test_records)}")

        # --- DenseNet201 training for this fold (exact settings) ---
        model = train_densenet201_for_split(train_records, test_records, device=device)

        # --- Build conv1 extractor on the trained model ---
        densenet_fe = DenseNet201MultiBlockConv1Extractor(
            features=model.features,
            device=device,
        )

        # --- Extract fused features for train & test ---
        print("  Extracting fused features (train) ...")
        X_train_blocks, y_train, pids_train = extract_fused_features_multi_block(
            train_records, device=device, densenet_fe=densenet_fe
        )
        print("  Extracting fused features (test) ...")
        X_test_blocks, y_test, pids_test = extract_fused_features_multi_block(
            test_records, device=device, densenet_fe=densenet_fe
        )

        print("  Train class counts:", Counter(y_train))
        print("  Test class counts: ", Counter(y_test))

        # --- SVM training per block ---
        param_grid = {
            "svm__C":     [0.5, 1, 2, 5, 10],
            "svm__gamma": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2],
        }

        for block_name in sorted(X_train_blocks.keys()):
            print(f"\n  --- Block: {block_name} (conv1 + GLCM) ---")

            Xtr = X_train_blocks[block_name]
            Xte = X_test_blocks[block_name]

            base_clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel=SVM_KERNEL, class_weight="balanced")),
                ]
            )

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=GLOBAL_SEED + fold_idx,
            )

            grid = GridSearchCV(
                estimator=base_clf,
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )

            grid.fit(Xtr, y_train)
            print(
                f"    Best params: {grid.best_params_}, "
                f"best CV acc={grid.best_score_:.4f}"
            )

            clf = grid.best_estimator_
            y_pred = clf.predict(Xte)

            img_acc = accuracy_score(y_test, y_pred)
            pat_acc = patient_level_accuracy(y_test, y_pred, pids_test)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, target_names=CLASS_NAMES, digits=4
            )

            all_block_img_accs[block_name].append(img_acc)
            all_block_pat_accs[block_name].append(pat_acc)

            print(f"    Image-level accuracy:  {img_acc:.4f}")
            print(f"    Patient-level accuracy:{pat_acc:.4f}")
            print("    Confusion matrix (rows=true, cols=pred):")
            print(cm)
            print("    Classification report:")
            print(report)

    # --------------------------------------------------------
    # 6) Summary across folds for each block
    # --------------------------------------------------------
    print("\n====================== Summary over 5 folds ======================")
    for block_name in sorted(all_block_img_accs.keys()):
        img_accs = np.array(all_block_img_accs[block_name])
        pat_accs = np.array(all_block_pat_accs[block_name])

        print(
            f"\nBlock {block_name}:"
            f"\n  Image-level acc:   mean={img_accs.mean():.4f}, std={img_accs.std():.4f}"
            f"\n  Patient-level acc: mean={pat_accs.mean():.4f}, std={pat_accs.std():.4f}"
        )


if __name__ == "__main__":
    main()
