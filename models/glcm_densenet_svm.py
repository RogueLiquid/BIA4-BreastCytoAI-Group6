#!/usr/bin/env python3
"""
glcm_densenet_svm.py

BreakHis 400X binary (benign vs malignant) classification using the pipeline
described in:

    "Breast cancer histopathological images classification based on deep semantic
     features and three-channel GLCM features" (PLOS ONE 2022)

Key points:

- Three-channel GLCM (R,G,B), distance=1, 4 directions
  (0, pi/4, pi/2, 3pi/4), Ng=256.
  -> For each channel, average GLCM over directions, compute 22 Haralick
     features -> 22 * 3 = 66-D GLCM feature vector.

- DenseNet201 (ImageNet-pretrained) deep semantic features from the 1x1 conv
  layers (conv1) of selected blocks in DenseBlock4:
    denselayer4, denselayer6, denselayer14,
    denselayer19, denselayer22, denselayer23

  For each conv1 output, we apply global average pooling -> 128-D deep vector.

- Feature fusion: concatenate [deep_128 || glcm_66] -> 194-D fused feature.

- Classifier: SVM with RBF kernel; in this script we grid-search C & gamma.

IMPORTANT: We DO NOT re-split the dataset. We trust your existing split:

    BreakHis_400X_full/
        train/<subtype>/*.png
        validation/<subtype>/*.png

We:
  - Train on all images under train/
  - Evaluate on all images under validation/

Save this file as: main_folder/models/glcm_densenet_svm.py

Run from main_folder:

    python -m models.glcm_densenet_svm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from PIL import Image

from skimage.feature import graycomatrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import densenet201

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# ================================================
# CONFIG
# ================================================

# main_folder/models/glcm_densenet_svm.py -> main_folder
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where your BreakHis 400X data lives
IMAGE_ROOT = REPO_ROOT / "BreakHis_400X_full"

# Where to cache GLCM features
GLCM_FEATURE_ROOT = REPO_ROOT / "BreakHis_400X_glcm"

# We will use these splits as-is (no resplitting)
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"

# Subtype -> class mapping (benign vs malignant)
SUBTYPE_TO_CLASS: Dict[str, str] = {
    # benign
    "adenosis": "benign",
    "fibroadenoma": "benign",
    "phyllodes_tumor": "benign",
    "tubular_adenoma": "benign",
    # malignant
    "ductal_carcinoma": "malignant",
    "lobular_carcinoma": "malignant",
    "mucinous_carcinoma": "malignant",
    "papillary_carcinoma": "malignant",
}
CLASS_NAMES = ["benign", "malignant"]  # 0->benign, 1->malignant

# GLCM parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
GLCM_LEVELS = 256

# SVM parameters (initial defaults; we grid-search around them)
SVM_KERNEL = "rbf"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Random seed (for reproducibility where relevant)
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# ================================================
# Data structures
# ================================================

@dataclass
class ImageRecord:
    img_path: Path
    label: int         # 0: benign, 1: malignant
    patient_id: str


# ================================================
# Utility: parse patient ID from BreakHis filename
# ================================================

def extract_patient_id_from_filename(path: Path) -> str:
    """
    BreakHis 400X filenames look like:
        SOB_B_A-14-22549AB-400-001.png

    We will treat '14-22549AB' as the patient ID, i.e. parts[1] + "-" + parts[2].

    If the pattern is different, we fall back to the stem.
    """
    stem = path.stem  # e.g. "SOB_B_A-14-22549AB-400-001"
    parts = stem.split("-")
    if len(parts) >= 3:
        return parts[1] + "-" + parts[2]
    else:
        return stem


# ================================================
# Load records from folder structure
# ================================================

def collect_image_records(split: str) -> List[ImageRecord]:
    """
    Collect ImageRecord for a given split (train or validation) using:

    BreakHis_400X_full/{split}/{subtype}/*.png
    """
    split_root = IMAGE_ROOT / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    records: List[ImageRecord] = []

    for subtype_dir in split_root.iterdir():
        if not subtype_dir.is_dir():
            continue
        subtype = subtype_dir.name
        if subtype not in SUBTYPE_TO_CLASS:
            print(f"[WARN] Unknown subtype folder '{subtype}', skipping.")
            continue

        class_name = SUBTYPE_TO_CLASS[subtype]
        label = CLASS_NAMES.index(class_name)

        for img_path in subtype_dir.glob("*.png"):
            patient_id = extract_patient_id_from_filename(img_path)
            records.append(
                ImageRecord(
                    img_path=img_path,
                    label=label,
                    patient_id=patient_id,
                )
            )

    return records


# ================================================
# GLCM 22 Haralick-like features
# ================================================

def _compute_glcm_22_features(p: np.ndarray) -> np.ndarray:
    """
    Compute 22 Haralick-like features for a normalized GLCM p (shape [Ng, Ng]).
    This follows standard Haralick definitions.
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

    # px+py and px-py distributions
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

    # 4) Correlation (alternative) – here same as corr1
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

    # 15) Sum variance (Haralick: variance around sum entropy)
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
    Compute three-channel GLCM features for an RGB image exactly as described:

    - Three-channel features (R, G, B separately).
    - For each channel:
        * GLCM at 4 directions: 0, pi/4, pi/2, 3pi/4
        * distance (step) = 1
        * gray level (Ng) = 256
        * For each direction's GLCM, compute 22 Haralick features.
        * Then average the 22 features across the 4 directions.

    -> 22 features per channel * 3 channels = 66-dim GLCM feature vector.

    image: HxWx3 RGB array in [0,255] (uint8 or convertible).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3) for GLCM")

    img_uint8 = image.astype(np.uint8)
    feats_all_channels: List[float] = []

    for ch in range(3):  # R, G, B
        channel = img_uint8[..., ch]

        # For each direction, compute a GLCM and then 22 features
        feats_per_angle: List[np.ndarray] = []

        for angle in GLCM_ANGLES:
            glcm = graycomatrix(
                channel,
                distances=GLCM_DISTANCES,  # [1]
                angles=[angle],            # single angle
                levels=GLCM_LEVELS,        # 256
                symmetric=True,
                normed=True,
            )
            # glcm shape: (levels, levels, 1, 1)
            p = glcm[:, :, 0, 0]  # [Ng, Ng]
            feats_22 = _compute_glcm_22_features(p)  # (22,)
            feats_per_angle.append(feats_22)

        # Stack over angles and average features across directions
        feats_per_angle = np.stack(feats_per_angle, axis=0)  # (4, 22)
        feats_mean = feats_per_angle.mean(axis=0)            # (22,)

        feats_all_channels.extend(feats_mean.tolist())

    return np.array(feats_all_channels, dtype=np.float64)  # length 66


# ================================================
# GLCM caching utilities
# ================================================

def get_glcm_feature_path(img_path: Path) -> Path:
    """
    Map image path under IMAGE_ROOT to its cached GLCM feature path under
    GLCM_FEATURE_ROOT.

    Example:
        IMAGE_ROOT/train/adenosis/img.png
    ->  GLCM_FEATURE_ROOT/train/adenosis/img_glcm.npy
    """
    rel = img_path.relative_to(IMAGE_ROOT)
    feat_path = GLCM_FEATURE_ROOT / rel.parent / f"{img_path.stem}_glcm.npy"
    return feat_path


def get_glcm_features_cached(img_path: Path, img_np: np.ndarray | None = None) -> np.ndarray:
    """
    Load GLCM features for this image if cached; otherwise compute and cache.

    This avoids recomputing GLCM every time we run the script.
    """
    feat_path = get_glcm_feature_path(img_path)

    if feat_path.exists():
        return np.load(feat_path)

    # If not cached, compute and save
    if img_np is None:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

    glcm_feats = compute_3channel_glcm_features(img_np)

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, glcm_feats)

    return glcm_feats


# ================================================
# DenseNet201 deep semantic features (multi-block conv1)
# ================================================

class DenseNet201MultiBlockConv1Extractor(nn.Module):
    """
    DenseNet201 feature extractor that returns GAP-pooled outputs from
    the 1x1 conv ("conv1") of selected blocks in DenseBlock4, as described in
    the paper:

        Dense Block4_block4_1, block6_1, block14_1,
        block19_1, block22_1, block23_1

    Blocks mapped to PyTorch DenseNet201 as:
        denseblock4.denselayer4.conv1,
        denseblock4.denselayer6.conv1,
        denseblock4.denselayer14.conv1,
        denseblock4.denselayer19.conv1,
        denseblock4.denselayer22.conv1,
        denseblock4.denselayer23.conv1
    """

    BLOCK_INDEX = {
        "block4": 4,
        "block6": 6,
        "block14": 14,
        "block19": 19,
        "block22": 22,
        "block23": 23,
    }

    def __init__(
        self,
        device: torch.device | None = None,
        ckpt_path: str | Path | None = None,
    ):
        super().__init__()

        # -----------------------------
        # 1) Build same architecture as in fine-tuning script
        # -----------------------------
        from torchvision import models

        # Default checkpoint path (relative to repo root) if not provided
        if ckpt_path is None:
            ckpt_path = REPO_ROOT / "checkpoints" / "densenet201_breakhis400x_finetuned.pth"
        ckpt_path = Path(ckpt_path)

        # Start with uninitialized DenseNet201, then load our fine-tuned weights
        base = models.densenet201(weights=None)
        num_features = base.classifier.in_features
        # Same change as in densenet201_finetune.py (2 classes)
        base.classifier = nn.Linear(num_features, 2)

        if ckpt_path.is_file():
            print(f"[DenseNet201MultiBlockConv1Extractor] Loading fine-tuned weights from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            base.load_state_dict(state_dict, strict=True)
        else:
            # Fallback: ImageNet pretrained if checkpoint is missing
            print(f"[WARN] Checkpoint not found at {ckpt_path}, falling back to ImageNet pretrained weights.")
            try:
                from torchvision.models import DenseNet201_Weights
                weights = DenseNet201_Weights.IMAGENET1K_V1
                base = models.densenet201(weights=weights)
            except Exception:
                base = models.densenet201(pretrained=True)

        # -----------------------------
        # 2) Store feature trunk
        # -----------------------------
        self.features = base.features  # conv0 -> norm5

        # which blocks we want
        self.block_names = list(self.BLOCK_INDEX.keys())

        # storage for activations
        self._feat_dict: dict[str, torch.Tensor | None] = {
            name: None for name in self.block_names
        }

        # register hooks on DenseBlock4 conv1 layers
        denseblock4 = self.features.denseblock4
        for name, idx in self.BLOCK_INDEX.items():
            layer = getattr(denseblock4, f"denselayer{idx}")
            conv1 = layer.conv1
            conv1.register_forward_hook(self._make_hook(name))

        # -----------------------------
        # 3) Preprocessing (must match training)
        # -----------------------------
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
            # output: [B, C, H, W]
            self._feat_dict[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [B, 3, H, W] ImageNet-normalized tensor

        returns:
            dict {block_name: [B, C]} where each is GAP-pooled conv1 output
            (C should be 128 for DenseNet201 conv1 in dense layers).
        """
        # reset stored features
        for k in self._feat_dict:
            self._feat_dict[k] = None

        _ = self.features(x)  # run full DenseNet201 features

        out: dict[str, torch.Tensor] = {}
        for name, feat in self._feat_dict.items():
            if feat is None:
                raise RuntimeError(f"No feature captured for {name}. Hook failed?")
            # feat: [B, C, H, W] -> [B, C]
            pooled = F.adaptive_avg_pool2d(feat, output_size=1).squeeze(-1).squeeze(-1)
            out[name] = pooled
        return out


# ================================================
# Feature extraction for a split (multi-block)
# ================================================

def extract_fused_features_multi_block(
    records: List[ImageRecord],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract fused features (deep + GLCM) for each ImageRecord in a split, for
    all selected conv1 blocks.

    Returns:
        X_blocks: dict {block_name: (N, 194)}  # 128 deep + 66 GLCM
        y:        (N,) labels (0 or 1)
        pids:     (N,) patient IDs
    """
    densenet_fe = DenseNet201MultiBlockConv1Extractor(device=device)

    # For each block, we keep a list of fused feature vectors
    X_blocks_lists: Dict[str, List[np.ndarray]] = {
        name: [] for name in densenet_fe.block_names
    }
    ys: List[int] = []
    pids: List[str] = []

    for idx, rec in enumerate(records, start=1):
        img = Image.open(rec.img_path).convert("RGB")
        img_np = np.array(img)

        # 1) GLCM features (cached)
        glcm_feats = get_glcm_features_cached(rec.img_path, img_np)  # (66,)

        # 2) DenseNet conv1 features (all six blocks)
        x_tensor = densenet_fe.preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            deep_feats_dict = densenet_fe(x_tensor)  # {block_name: [1, C]}

        for name in densenet_fe.block_names:
            deep_vec = deep_feats_dict[name].cpu().numpy().reshape(-1)  # (C=128,)
            fused = np.concatenate([deep_vec, glcm_feats], axis=0)      # (194,)
            X_blocks_lists[name].append(fused)

        ys.append(rec.label)
        pids.append(rec.patient_id)

        if idx % 50 == 0:
            print(f"[feat] Processed {idx}/{len(records)} images")

    # Stack into arrays
    X_blocks: Dict[str, np.ndarray] = {}
    for name, vec_list in X_blocks_lists.items():
        X_blocks[name] = np.stack(vec_list, axis=0)

    y = np.array(ys, dtype=np.int64)
    pids_arr = np.array(pids)
    return X_blocks, y, pids_arr


# ================================================
# Patient-level accuracy (no resplit)
# ================================================

def patient_level_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
) -> float:
    """
    Spanhol-style patient-level accuracy:
      For each patient: (#correct images) / (#images)
      Overall: mean across patients
    """
    assert y_true.shape == y_pred.shape == patient_ids.shape

    acc_per_patient: Dict[str, float] = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        acc_per_patient[pid] = correct / total if total > 0 else 0.0

    return float(np.mean(list(acc_per_patient.values())))


# ================================================
# Main: train on train/, evaluate on validation/
# ================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"IMAGE_ROOT = {IMAGE_ROOT}")
    print(f"GLCM_FEATURE_ROOT = {GLCM_FEATURE_ROOT}")

    # 1) Collect records
    print("\n[1] Collecting records from train split ...")
    train_records = collect_image_records(TRAIN_SPLIT)
    print(f"  Train images: {len(train_records)}")

    print("[1] Collecting records from validation split ...")
    val_records = collect_image_records(VAL_SPLIT)
    print(f"  Validation images: {len(val_records)}")

    if len(train_records) == 0 or len(val_records) == 0:
        raise RuntimeError("Train or validation set is empty. Check folder structure.")

    # 2) Extract fused features for each block
    print("\n[2] Extracting fused features (deep + GLCM) for train ...")
    X_train_blocks, y_train, pids_train = extract_fused_features_multi_block(
        train_records, device=device
    )
    print("  Train feature shapes per block:")
    for name, X in X_train_blocks.items():
        print(f"    {name}: {X.shape}")

    print("\n[2] Extracting fused features (deep + GLCM) for validation ...")
    X_val_blocks, y_val, pids_val = extract_fused_features_multi_block(
        val_records, device=device
    )
    print("  Val feature shapes per block:")
    for name, X in X_val_blocks.items():
        print(f"    {name}: {X.shape}")

    print("Train class counts:", Counter(r.label for r in train_records))
    print("Val class counts:", Counter(r.label for r in val_records))

    # 3) Train SVM per block and evaluate on validation
    print("\n[3] Training SVM (RBF) for each conv1 block ...")
    for block_name in sorted(X_train_blocks.keys()):
        print(f"\n=== Block: {block_name} (conv1 + GLCM) ===")
        Xtr = X_train_blocks[block_name]
        Xva = X_val_blocks[block_name]

        param_grid = {
            "svm__C":     [0.5, 1, 2, 5, 10],
            "svm__gamma": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2],
        }

        base_clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel=SVM_KERNEL, class_weight="balanced")),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid.fit(Xtr, y_train)
        print(f"  Best params for {block_name}: {grid.best_params_}, best CV acc={grid.best_score_:.4f}")

        clf = grid.best_estimator_

        y_pred = clf.predict(Xva)

        img_acc = accuracy_score(y_val, y_pred)
        pat_acc = patient_level_accuracy(y_val, y_pred, pids_val)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(
            y_val, y_pred, target_names=CLASS_NAMES, digits=4
        )

        print(f"Image-level accuracy (validation):  {img_acc:.4f}")
        print(f"Patient-level accuracy (validation):{pat_acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification report:")
        print(report)


if __name__ == "__main__":
    main()
