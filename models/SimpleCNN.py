#!/usr/bin/env python3
"""SimpleCNN training script for BreakHis 400X augmented dataset (no in-pipeline augmentation).

- Uses z-score normalization with mean/std computed from the training split.
- Expects input folders under BreakHis_400X_full_augmented/{train,validation}/<class_subtype>.
- Provides a `run_training` helper to train/validate and save checkpoints.
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = REPO_ROOT / "weights" / "BreakHis_400X_full_augmented_stats.json"

# Map subtypes into binary classes
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

        # Base transforms for stats (no normalize); input images are already PIL.
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


class SimpleCNN(nn.Module):
    """Same architecture as in test_code.py."""

    def __init__(self, num_classes: int = 2, base_channels: int = 16, dropout_p: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


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


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    avg_loss = running_loss / len(loader)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


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


def run_training(
    train_dir: Path | str = REPO_ROOT / "BreakHis_400X_full_augmented" / "train",
    val_dir: Path | str = REPO_ROOT / "BreakHis_400X_full_augmented" / "validation",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: Path | str = REPO_ROOT / "weights",
    save_prefix: str = "SimpleCNN_aug",
    resize: Tuple[int, int] = (224, 224),
):
    stats = load_stats(STATS_PATH)
    mean_override, std_override = (stats if stats is not None else (None, None))

    train_loader, val_loader, mean, std, classes = create_dataloaders(
        Path(train_dir),
        Path(val_dir),
        batch_size=batch_size,
        resize=resize,
        mean_override=mean_override,
        std_override=std_override,
    )

    # Save stats if we just computed them
    if mean_override is None or std_override is None:
        save_stats(mean, std, STATS_PATH)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_path = Path(save_dir) / f"{save_prefix}_best.pth"
            torch.save({"model_state": model.state_dict(), "mean": mean, "std": std, "classes": classes}, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    return model


# Example usage inside VS Code / notebook:
# from models.SimpleCNN import run_training
model = run_training()
