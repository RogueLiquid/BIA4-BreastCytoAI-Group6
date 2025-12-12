#!/usr/bin/env python3
"""
densenet201_finetune.py

DenseNet201 baseline training on BreakHis 400X as described in:

    "Breast cancer histopathological images classification based on deep semantic
     features and three-channel GLCM features" (PLOS ONE 2022)

Key training settings (mirroring the paper):

- Input size for DenseNet201: 224 x 224
- Pretrained on ImageNet
- Optimizer: SGD
    - learning rate = 1e-4
    - momentum = 0.9
- Loss: Cross-Entropy
- Batch size: 10
- Epochs: 6
- Fine-tune the *entire* network (no layers are frozen)

We assume the dataset is already split into train/validation:

    BreakHis_400X_full/
        train/<subtype>/*.png
        validation/<subtype>/*.png

Subtypes are mapped to binary labels (benign vs malignant) by SUBTYPE_TO_CLASS.

Run from the repository root:

    python -m models.densenet201_finetune
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ================================================
# CONFIG
# ================================================

# main_folder/models/densenet201_finetune.py -> main_folder
REPO_ROOT = Path(__file__).resolve().parent.parent

# Data root
IMAGE_ROOT = REPO_ROOT / "BreakHis_400X_full"

TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"

# Where to save trained model
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "densenet201_breakhis400x_finetuned.pth"

# Mapping BreakHis subtypes -> binary labels
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
CLASS_NAMES = ["benign", "malignant"]  # 0 -> benign, 1 -> malignant

# Training hyperparameters (as in the paper)
BATCH_SIZE = 10
EPOCHS = 6
LR = 1e-4
MOMENTUM = 0.9

# ImageNet normalization (for DenseNet201)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Random seed for reproducibility
GLOBAL_SEED = 42


# ================================================
# Small helper dataclass
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
    BreakHis 400X filenames typically look like:
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
# Collect image records from folder structure
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
# PyTorch Dataset
# ================================================

class BreakHisDataset(Dataset):
    """
    Simple Dataset wrapping a list of ImageRecord objects.

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


# ================================================
# Patient-level accuracy (for validation)
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
        mask = (patient_ids == pid)
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        acc_per_patient[pid] = correct / total if total > 0 else 0.0

    return float(np.mean(list(acc_per_patient.values())))


# ================================================
# Training / evaluation loops
# ================================================

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
        outputs = model(images)           # [B, 2]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        if step % 20 == 0:
            print(
                f"  [Epoch {epoch} | Step {step}/{len(loader)}] "
                f"loss={loss.item():.4f}"
            )

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
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

    # Patient-level accuracy
    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    pids_arr = np.array(all_pids)

    pat_acc = patient_level_accuracy(y_true, y_pred, pids_arr)

    return val_loss, img_acc, pat_acc


# ================================================
# Main
# ================================================

def main():
    # Reproducibility
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"IMAGE_ROOT = {IMAGE_ROOT}")
    print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}")

    # 1) Collect records
    print("\n[1] Collecting records ...")
    train_records = collect_image_records(TRAIN_SPLIT)
    val_records = collect_image_records(VAL_SPLIT)

    print(f"  Train images:      {len(train_records)}")
    print(f"  Validation images: {len(val_records)}")

    print("  Train class counts:", Counter(r.label for r in train_records))
    print("  Val class counts:  ", Counter(r.label for r in val_records))

    if len(train_records) == 0 or len(val_records) == 0:
        raise RuntimeError("Train or validation set is empty. Check folder structure.")

    # 2) Datasets and loaders
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

    # 3) Model: DenseNet201 pretrained on ImageNet, fine-tune all layers
    print("\n[2] Building DenseNet201 (ImageNet pretrained) ...")
    try:
        from torchvision.models import DenseNet201_Weights
        weights = DenseNet201_Weights.IMAGENET1K_V1
        model = models.densenet201(weights=weights)
    except Exception:
        # Older torchvision fallback
        model = models.densenet201(pretrained=True)

    # Replace classifier with 2-class output
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)

    model = model.to(device)

    # 4) Loss and optimizer (as in the paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
    )

    # 5) Training loop
    best_val_img_acc = 0.0
    best_state_dict = None

    print("\n[3] Starting training ...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )

        val_loss, val_img_acc, val_pat_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = time.time() - start_time

        print(
            f"\nEpoch {epoch}/{EPOCHS} "
            f"({epoch_time:.1f} s) "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} "
            f"| Val loss={val_loss:.4f}, img_acc={val_img_acc:.4f}, "
            f"pat_acc={val_pat_acc:.4f}"
        )

        # Track best image-level accuracy
        if val_img_acc > best_val_img_acc:
            best_val_img_acc = val_img_acc
            best_state_dict = model.state_dict().copy()

    # 6) Save best model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if best_state_dict is not None:
        torch.save(
            {
                "state_dict": best_state_dict,
                "best_val_img_acc": best_val_img_acc,
                "num_classes": 2,
            },
            CHECKPOINT_PATH,
        )
        print(f"\nBest model saved with val image-level acc={best_val_img_acc:.4f}")
    else:
        print("\nWARNING: No best model state was recorded (something went wrong).")


if __name__ == "__main__":
    main()
