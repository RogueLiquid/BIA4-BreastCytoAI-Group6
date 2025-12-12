#!/usr/bin/env python3
"""EfficientNet-B0 training script for BreakHis 400X augmented dataset (patient-level split).

- Uses ImageNet-pretrained EfficientNet-B0.
- Uses ImageNet mean/std normalization (no dataset z-score).
- Freezes most backbone parameters; only classifier head is trainable.
- Expects input folders under BreakHis_400X_full_augmented/{train,validation}/<subtype>.
- Maps subtypes -> {benign, malignant}.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ----------------------------------------------------------------------
# Reproducibility & paths
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parent.parent

# ImageNet normalization for EfficientNet-B0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_SIZE = 224  # standard input size

# ----------------------------------------------------------------------
# Subtype -> binary class mapping
# ----------------------------------------------------------------------
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
CLASS_NAMES: List[str] = ["benign", "malignant"]


# ----------------------------------------------------------------------
# Dataset (no in-pipeline augmentation; just resize + ImageNet norm)
# ----------------------------------------------------------------------
class BreastCancerDataset(Dataset):
    """Dataset for BreakHis subtype folders; resize + ImageNet normalization."""

    def __init__(
        self,
        root_dir: str | Path,
        resize: Tuple[int, int] = (IMAGENET_SIZE, IMAGENET_SIZE),
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        print_stats: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.resize = resize
        self.img_paths: List[Path] = []
        self.labels: List[int] = []

        self.classes = CLASS_NAMES

        # Walk subtype folders and map to binary labels
        for subtype_dir in self.root_dir.iterdir():
            if not subtype_dir.is_dir():
                continue
            subtype = subtype_dir.name
            if subtype not in SUBTYPE_TO_CLASS:
                raise ValueError(f"Unknown subtype folder '{subtype}' in {self.root_dir}")
            class_name = SUBTYPE_TO_CLASS[subtype]
            label = self.classes.index(class_name)
            paths = list(subtype_dir.glob("*.png"))
            self.img_paths.extend(paths)
            self.labels.extend([label] * len(paths))

        if print_stats:
            print(f"[Dataset] {self.root_dir} -> {len(self.img_paths)} images")

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


# ----------------------------------------------------------------------
# EfficientNet-B0 model wrapper (pretrained + frozen backbone)
# ----------------------------------------------------------------------
class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 backbone with a binary classifier head.

    - Loads ImageNet-pretrained weights.
    - Optionally freezes backbone features.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_p: float | None = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Replace final classifier layer
        in_features = self.backbone.classifier[1].in_features

        # Optionally override dropout p in the classifier
        if dropout_p is not None and hasattr(self.backbone.classifier[0], "p"):
            self.backbone.classifier[0].p = dropout_p

        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Optionally freeze most of the backbone (only classifier is trainable)
        if freeze_backbone:
            for name, param in self.backbone.features.named_parameters():
                param.requires_grad = False
            # You could also freeze more (or less) by adjusting this.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ----------------------------------------------------------------------
# Dataloaders
# ----------------------------------------------------------------------
def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    resize: Tuple[int, int] = (IMAGENET_SIZE, IMAGENET_SIZE),
):
    train_ds = BreastCancerDataset(
        train_dir,
        resize=resize,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        print_stats=True,
    )
    val_ds = BreastCancerDataset(
        val_dir,
        resize=resize,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        print_stats=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes


# ----------------------------------------------------------------------
# Training & validation loops
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion, num_classes: int):
    model.eval()
    running_loss = 0.0
    correct = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()
            for p, t in zip(preds_cpu.view(-1), labels_cpu.view(-1)):
                confusion[t.long(), p.long()] += 1

    avg_loss = running_loss / len(loader)
    acc = correct / len(loader.dataset)
    return avg_loss, acc, confusion


def format_confusion_matrix(cm: torch.Tensor, class_names: Sequence[str]) -> str:
    cm_np = cm.cpu().numpy()
    header = [""] + [f"pred_{name}" for name in class_names]
    lines = ["\t".join(header)]
    for idx, row in enumerate(cm_np):
        lines.append("\t".join([f"true_{class_names[idx]}"] + [str(int(x)) for x in row]))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# High-level training entrypoint
# ----------------------------------------------------------------------
def run_training(
    train_dir: Path | str = REPO_ROOT / "BreakHis_400X_full_augmented" / "train",
    val_dir: Path | str = REPO_ROOT / "BreakHis_400X_full_augmented" / "validation",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    save_dir: Path | str = REPO_ROOT / "weights",
    save_prefix: str = "EffNetB0_pretrained_frozen",
    resize: Tuple[int, int] = (IMAGENET_SIZE, IMAGENET_SIZE),
):
    train_loader, val_loader, classes = create_dataloaders(
        Path(train_dir),
        Path(val_dir),
        batch_size=batch_size,
        resize=resize,
    )

    model = EfficientNetB0Classifier(
        num_classes=len(classes),
        pretrained=True,
        dropout_p=0.5,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients (classifier head if backbone is frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_cm = validate(model, val_loader, criterion, num_classes=len(classes))
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        print(format_confusion_matrix(val_cm, classes))

        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_path = Path(save_dir) / f"{save_prefix}_best.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    return model

run_training()
