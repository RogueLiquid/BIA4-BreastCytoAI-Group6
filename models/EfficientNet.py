#!/usr/bin/env python3
"""EfficientNet-B0 training script for BreakHis 400X augmented dataset (no in-pipeline augmentation).

- Uses z-score normalization with mean/std computed from the training split.
- Expects input folders under BreakHis_400X_full_augmented/{train,validation}/<subtype>.
- Maps subtypes -> {benign, malignant}.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0  # and optionally EfficientNet_B0_Weights

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
STATS_PATH = REPO_ROOT / "weights" / "BreakHis_400X_full_augmented2_stats.json"

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
# Dataset
# ----------------------------------------------------------------------
class BreastCancerDataset(Dataset):
    """Dataset for BreakHis subtype folders; no augmentation, only resize + z-score norm."""

    def __init__(
        self,
        root_dir: str | Path,
        resize: Tuple[int, int] = (224, 224),
        compute_stats: bool = False,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
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

        # Base transforms for stats (no normalize); ToTensor() -> [0,1]
        self._base_tfms = transforms.Compose(
            [transforms.Resize(self.resize), transforms.ToTensor()]
        )

        if compute_stats:
            self.mean, self.std = self._compute_mean_std(print_stats=print_stats)
        else:
            if mean is None or std is None:
                raise ValueError("Provide mean/std or set compute_stats=True.")
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
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

    def _compute_mean_std(self, print_stats: bool = True):
        n_use = len(self.img_paths)
        sum_c = torch.zeros(3, dtype=torch.float64)
        sum_sq_c = torch.zeros(3, dtype=torch.float64)
        count = 0

        if print_stats:
            print(f"[Stats] Computing mean/std on {n_use} images ...")

        for i in range(n_use):
            from PIL import Image

            img = Image.open(self.img_paths[i]).convert("RGB")
            t = self._base_tfms(img)  # [C,H,W] in [0,1]
            sum_c += t.sum(dim=[1, 2]).double()
            sum_sq_c += (t ** 2).sum(dim=[1, 2]).double()
            count += t.shape[1] * t.shape[2]

        mean = sum_c / count
        var = (sum_sq_c / count) - mean ** 2
        std = torch.sqrt(var)

        if print_stats:
            print("Mean:", mean.tolist())
            print("Std:", std.tolist())

        return mean.float(), std.float()


# ----------------------------------------------------------------------
# EfficientNet-B0 model wrapper
# ----------------------------------------------------------------------
class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 backbone with a binary classifier head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout_p: float | None = None):
        super().__init__()
        if pretrained:
            # If you want ImageNet pretrained, you can enable this and
            # ALSO swap your normalization to ImageNet stats.
            # from torchvision.models import EfficientNet_B0_Weights
            # weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            # self.backbone = efficientnet_b0(weights=weights)
            raise NotImplementedError(
                "Pretrained=True is not wired to change normalization; "
                "either keep pretrained=False or add ImageNet normalization."
            )
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Replace final classifier layer
        in_features = self.backbone.classifier[1].in_features
        # Optionally override dropout if requested
        if dropout_p is not None and hasattr(self.backbone.classifier[0], "p"):
            self.backbone.classifier[0].p = dropout_p
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

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
    resize: Tuple[int, int] = (224, 224),
    mean_override: Sequence[float] | None = None,
    std_override: Sequence[float] | None = None,
):
    train_ds = BreastCancerDataset(
        train_dir,
        resize=resize,
        compute_stats=(mean_override is None or std_override is None),
        mean=mean_override,
        std=std_override,
        print_stats=True,
    )
    val_ds = BreastCancerDataset(
        val_dir,
        resize=resize,
        compute_stats=False,
        mean=train_ds.mean.tolist(),
        std=train_ds.std.tolist(),
        print_stats=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.mean, train_ds.std, train_ds.classes


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
# Mean/std cache helpers
# ----------------------------------------------------------------------
def save_stats(mean: torch.Tensor, std: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"mean": mean.tolist(), "std": std.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_stats(path: Path) -> Tuple[List[float], List[float]] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mean"), data.get("std")


# ----------------------------------------------------------------------
# High-level training entrypoint
# ----------------------------------------------------------------------
def run_training(
    train_dir: Path | str = REPO_ROOT / "BreakHis_400X_full_augmented2" / "train",
    val_dir: Path | str = REPO_ROOT / "BreakHis_400X_full_augmented2" / "validation",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: Path | str = REPO_ROOT / "weights",
    save_prefix: str = "EffNetB0_aug",
    resize: Tuple[int, int] = (224, 224),
):
    stats = load_stats(STATS_PATH)
    mean_override, std_override = (stats if stats is not None else (None, None))

    # If stats are present, avoid recomputing; otherwise compute and save.
    if mean_override is not None and std_override is not None:
        compute_train_stats = False
    else:
        compute_train_stats = True

    train_loader, val_loader, mean, std, classes = create_dataloaders(
        Path(train_dir),
        Path(val_dir),
        batch_size=batch_size,
        resize=resize,
        mean_override=mean_override,
        std_override=std_override,
    )

    if compute_train_stats:
        save_stats(mean, std, STATS_PATH)

    model = EfficientNetB0Classifier(num_classes=len(classes), pretrained=False, dropout_p=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_cm = validate(model, val_loader, criterion, num_classes=len(classes))
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        print(format_confusion_matrix(val_cm, classes))

        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_path = Path(save_dir) / f"{save_prefix}_best.pth"
            torch.save(
                {"model_state": model.state_dict(), "mean": mean, "std": std, "classes": classes},
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    return model


# Example usage inside VS Code / notebook:
# from models.EfficientNetB0 import run_training
model = run_training()
