#!/usr/bin/env python3
from __future__ import annotations
"""
SimpleCNN training on BreakHis 400X with GA-like search over image augmentation hyperparameters.

- Dataset expected at:
    BreakHis_400X_full/
        train/<subtype>/*.png
        validation/<subtype>/*.png

- Subtypes are mapped to binary classes {benign, malignant}.
- Training-time augmentation only uses:

    transforms.RandomHorizontalFlip(p=p_hflip),
    transforms.RandomVerticalFlip(p=p_vflip),
    transforms.RandomRotation(degrees=rotation_degrees),
    transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

- GA-like search:
    * Generation 0: random configs from wide ranges.
    * Generation g>0: sample configs in a shrinking window around the best config from previous generations.
"""

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------------------------------------------------
# Reproducibility & device
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] Using device: {device}")

# ----------------------------------------------------------------------
# Paths (adjust if needed)
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "BreakHis_400X_full"
TRAIN_DIR_DEFAULT = DATA_ROOT / "train"
VAL_DIR_DEFAULT = DATA_ROOT / "validation"

WEIGHTS_DIR = REPO_ROOT / "weights"
STATS_PATH = WEIGHTS_DIR / "BreakHis_400X_simplecnn_stats.json"
AUG_SEARCH_LOG_PATH = WEIGHTS_DIR / "augmentation_ga_results.json"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

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
# SimpleCNN model
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Baseline SimpleCNN with classifier dropout."""

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

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class BreastCancerDataset(Dataset):
    """
    Dataset for BreakHis subtype folders.

    - Computes mean/std on raw images (resize + ToTensor, no augmentation) when compute_stats=True.
    - During training/validation, uses:
        [extra_tfms (if provided)] -> Resize -> ToTensor -> Normalize(mean,std)
    """

    def __init__(
        self,
        root_dir: str | Path,
        resize: Tuple[int, int] = (224, 224),
        compute_stats: bool = False,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        print_stats: bool = True,
        extra_tfms: transforms.Compose | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.resize = resize
        self.img_paths: List[Path] = []
        self.labels: List[int] = []
        self.extra_tfms = extra_tfms

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

        # Build main transform: (optional augment) -> resize -> tensor -> normalize
        core_tfms = [
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
        ]
        if self.extra_tfms is not None:
            self.transform = transforms.Compose([self.extra_tfms] + core_tfms)
        else:
            self.transform = transforms.Compose(core_tfms)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
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
            print(f"[Stats] Computing mean/std on {n_use} images (no aug) ...")

        for i in range(n_use):
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
# Dataloaders
# ----------------------------------------------------------------------
def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    resize: Tuple[int, int] = (224, 224),
    mean_override: Sequence[float] | None = None,
    std_override: Sequence[float] | None = None,
    train_extra_tfms: transforms.Compose | None = None,
):
    train_ds = BreastCancerDataset(
        train_dir,
        resize=resize,
        compute_stats=(mean_override is None or std_override is None),
        mean=mean_override,
        std=std_override,
        print_stats=True,
        extra_tfms=train_extra_tfms,
    )
    val_ds = BreastCancerDataset(
        val_dir,
        resize=resize,
        compute_stats=False,
        mean=train_ds.mean.tolist(),
        std=train_ds.std.tolist(),
        print_stats=False,
        extra_tfms=None,  # no augmentation in validation
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.mean, train_ds.std, train_ds.classes


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


def save_aug_search_results(history: List[Dict], best_cfg: AugConfig | None, best_val_loss: float) -> None:
    """Persist GA search history and best config for later inspection."""
    AUG_SEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "meta": {
            "num_generations": max((item["generation"] for item in history), default=-1) + 1,
            "population_size": max((item["individual"] for item in history), default=0),
            "search_epochs": history[0].get("search_epochs") if history else None,
            "train_dir": str(history[0].get("train_dir")) if history and "train_dir" in history[0] else None,
            "val_dir": str(history[0].get("val_dir")) if history and "val_dir" in history[0] else None,
        },
        "history": history,
        "best_config": asdict(best_cfg) if best_cfg is not None else None,
        "best_val_loss": best_val_loss,
    }
    with open(AUG_SEARCH_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_last_best_config() -> Tuple[AugConfig | None, float | None]:
    """Load the last saved GA run (if present) and return best AugConfig and its val_loss."""
    if not AUG_SEARCH_LOG_PATH.exists():
        return None, None
    try:
        with open(AUG_SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        best_cfg_dict = data.get("best_config")
        best_loss = data.get("best_val_loss")
        if best_cfg_dict is None or best_loss is None:
            return None, None
        return AugConfig(**best_cfg_dict), float(best_loss)
    except Exception:
        return None, None


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


# ----------------------------------------------------------------------
# Augmentation config & GA-like search
# ----------------------------------------------------------------------
@dataclass
class AugConfig:
    p_hflip: float
    p_vflip: float
    rotation_degrees: float
    brightness: float
    contrast: float
    saturation: float
    hue: float


# Global bounds for each parameter
PARAM_BOUNDS = {
    # Flips/rotation fixed; keep bounds for clarity.
    "p_hflip": (0.5, 0.5),
    "p_vflip": (0.5, 0.5),
    "rotation_degrees": (90.0, 90.0),
    "brightness": (0.0, 1.0),
    "contrast": (0.0, 1.0),
    "saturation": (0.0, 1.0),
    "hue": (0.0, 0.5),  # torchvision requires hue <= 0.5
}


def sample_initial_config() -> AugConfig:
    """Sample an augmentation config uniformly from the full bounds (Generation 0)."""
    return AugConfig(
        p_hflip=0.5,
        p_vflip=0.5,
        rotation_degrees=90.0,
        brightness=float(np.random.uniform(*PARAM_BOUNDS["brightness"])),
        contrast=float(np.random.uniform(*PARAM_BOUNDS["contrast"])),
        saturation=float(np.random.uniform(*PARAM_BOUNDS["saturation"])),
        hue=float(np.random.uniform(*PARAM_BOUNDS["hue"])),
    )


def mutate_config_around_best(best: AugConfig, generation_idx: int) -> AugConfig:
    """
    Sample a new config around 'best' with a shrinking window as generation_idx increases.

    generation_idx: 1,2,3,... (0 is reserved for initial random generation).
    """
    def mutate(param_name: str, current: float) -> float:
        low, high = PARAM_BOUNDS[param_name]
        full_range = high - low
        # window shrinks as 1 / (generation_idx + 1)
        base_width = 0.5 * full_range  # initial window = half of full range
        width = base_width / (generation_idx + 1)

        new_low = max(low, current - width)
        new_high = min(high, current + width)
        if new_high <= new_low:
            return current  # degenerate window; just return current
        return float(np.random.uniform(new_low, new_high))

    return AugConfig(
        p_hflip=0.5,
        p_vflip=0.5,
        rotation_degrees=90.0,
        brightness=mutate("brightness", best.brightness),
        contrast=mutate("contrast", best.contrast),
        saturation=mutate("saturation", best.saturation),
        hue=mutate("hue", best.hue),
    )


def build_augment_from_config(cfg: AugConfig) -> transforms.Compose | None:
    """
    Build transforms.Compose([
        RandomHorizontalFlip,
        RandomVerticalFlip,
        RandomRotation,
        ColorJitter,
    ]) from an AugConfig.
    """
    ops: List[transforms.Transform] = []

    if cfg.p_hflip > 0.0:
        ops.append(transforms.RandomHorizontalFlip(p=cfg.p_hflip))
    if cfg.p_vflip > 0.0:
        ops.append(transforms.RandomVerticalFlip(p=cfg.p_vflip))
    if cfg.rotation_degrees > 0.0:
        ops.append(transforms.RandomRotation(degrees=cfg.rotation_degrees))

    cj_kwargs = {}
    if cfg.brightness > 0.0:
        cj_kwargs["brightness"] = cfg.brightness
    if cfg.contrast > 0.0:
        cj_kwargs["contrast"] = cfg.contrast
    if cfg.saturation > 0.0:
        cj_kwargs["saturation"] = cfg.saturation
    if cfg.hue > 0.0:
        cj_kwargs["hue"] = cfg.hue

    if cj_kwargs:
        ops.append(transforms.ColorJitter(**cj_kwargs))

    if not ops:
        return None

    return transforms.Compose(ops)


# ----------------------------------------------------------------------
# Training wrapper (one config)
# ----------------------------------------------------------------------
def train_with_aug_config(
    cfg: AugConfig,
    train_dir: Path,
    val_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    base_channels: int = 16,
) -> float:
    """
    Train SimpleCNN for a few epochs with a given augmentation config and
    return the validation loss from the final epoch (selection based on last-epoch val loss).
    """
    print(f"\n=== Training with config ===")
    print(cfg)

    # Load stats (compute once globally)
    stats = load_stats(STATS_PATH)
    mean_override, std_override = (stats if stats is not None else (None, None))
    compute_train_stats = stats is None

    aug_tfms = build_augment_from_config(cfg)

    train_loader, val_loader, mean, std, classes = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        resize=(224, 224),
        mean_override=mean_override,
        std_override=std_override,
        train_extra_tfms=aug_tfms,
    )

    if compute_train_stats:
        save_stats(mean, std, STATS_PATH)

    model = SimpleCNN(num_classes=len(classes), base_channels=base_channels, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    last_val_loss = float("inf")
    last_val_acc = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _ = validate(model, val_loader, criterion, num_classes=len(classes))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        last_val_loss = val_loss
        last_val_acc = val_acc

    print(f"Final epoch Val Loss for this config: {last_val_loss:.4f} (Val Acc: {last_val_acc:.4f})")
    return last_val_loss


# ----------------------------------------------------------------------
# GA-like augmentation hyperparameter search
# ----------------------------------------------------------------------
def ga_like_aug_search(
    num_generations: int = 3,
    population_size: int = 6,
    search_epochs: int = 3,
    train_dir: Path | str = TRAIN_DIR_DEFAULT,
    val_dir: Path | str = VAL_DIR_DEFAULT,
    start_config: AugConfig | None = None,
    start_best_loss: float | None = None,
    start_generation: int = 0,
) -> Tuple[AugConfig, float]:
    """
    Genetic-Algorithm-like search:

    - Generation 0:
        * population_size random configs sampled from full ranges.
    - Generations g >= 1:
        * population_size configs sampled around the current best config using mutate_config_around_best.
    - For each config:
        * Train SimpleCNN for `search_epochs` epochs.
        * Use final-epoch validation loss as the score (lower is better).
    - Return best_config and its final-epoch val_loss.

    NOTE: This can be compute-intensive if you increase generations/population_size/search_epochs.
    """

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    best_config: AugConfig | None = start_config
    best_val_loss: float = start_best_loss if start_best_loss is not None else float("inf")
    run_history: List[Dict] = []

    for gen in range(num_generations):
        print("\n" + "#" * 80)
        print(f"Generation {start_generation + gen} / {start_generation + num_generations - 1}")
        print("#" * 80)

        configs: List[AugConfig] = []

        if gen == 0 and best_config is not None:
            # Warm start: include provided best_config plus mutated neighbors.
            configs.append(best_config)
            for _ in range(population_size - 1):
                configs.append(
                    mutate_config_around_best(
                        best_config,
                        generation_idx=max(1, start_generation + gen),
                    )
                )
        elif gen == 0 or best_config is None:
            # Initial random population
            for _ in range(population_size):
                configs.append(sample_initial_config())
        else:
            # Mutated population around best_config
            for _ in range(population_size):
                configs.append(
                    mutate_config_around_best(
                        best_config,
                        generation_idx=max(1, start_generation + gen),
                    )
                )

        for idx, cfg in enumerate(configs):
            print(f"\n--- Gen {gen} | Individual {idx+1}/{population_size} ---")
            val_loss = train_with_aug_config(
                cfg,
                train_dir=train_dir,
                val_dir=val_dir,
                epochs=search_epochs,
                batch_size=64,
                lr=1e-3,
                weight_decay=1e-4,
                base_channels=16,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = cfg
                print(f"*** New best config found! Val Loss = {best_val_loss:.4f} ***")

            run_history.append(
                {
                    "generation": start_generation + gen,
                    "individual": idx + 1,
                    "config": asdict(cfg),
                    "val_loss": val_loss,
                    "search_epochs": search_epochs,
                    "train_dir": str(train_dir),
                    "val_dir": str(val_dir),
                }
            )

    print("\n" + "=" * 80)
    print("GA-like augmentation search finished.")
    print(f"Best (lowest) validation loss: {best_val_loss:.4f}")
    print("Best augmentation config:")
    print(best_config)
    print("=" * 80)

    save_aug_search_results(run_history, best_config, best_val_loss)

    return best_config, best_val_loss


# ----------------------------------------------------------------------
# Optional: retrain SimpleCNN with best augmentation config
# ----------------------------------------------------------------------
def retrain_with_best_config(
    best_cfg: AugConfig,
    train_dir: Path | str = TRAIN_DIR_DEFAULT,
    val_dir: Path | str = VAL_DIR_DEFAULT,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    base_channels: int = 16,
):
    """
    Retrain SimpleCNN for more epochs using the best augmentation config found
    in the GA-like search, and save a checkpoint.
    """
    print("\n" + "#" * 80)
    print("Retraining with best augmentation config")
    print(best_cfg)
    print("#" * 80)

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    stats = load_stats(STATS_PATH)
    mean_override, std_override = (stats if stats is not None else (None, None))
    compute_train_stats = stats is None

    aug_tfms = build_augment_from_config(best_cfg)

    train_loader, val_loader, mean, std, classes = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        resize=(224, 224),
        mean_override=mean_override,
        std_override=std_override,
        train_extra_tfms=aug_tfms,
    )
    if compute_train_stats:
        save_stats(mean, std, STATS_PATH)

    model = SimpleCNN(num_classes=len(classes), base_channels=base_channels, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _ = validate(model, val_loader, criterion, num_classes=len(classes))

        print(
            f"[Retrain] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = WEIGHTS_DIR / "SimpleCNN_best_aug_config.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "classes": classes,
                },
                ckpt_path,
            )
            print(f"[Retrain] Saved best checkpoint to {ckpt_path}")

    print(f"[Retrain] Final best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


# Convenience: resume-aware search runner
def run_aug_search_with_resume(
    num_generations: int = 3,
    population_size: int = 6,
    search_epochs: int = 3,
    train_dir: Path | str = TRAIN_DIR_DEFAULT,
    val_dir: Path | str = VAL_DIR_DEFAULT,
    use_last_best: bool = True,
):
    """
    Run GA-like search, optionally seeding from the last saved best config.

    Returns: (best_cfg, best_val_loss)
    """
    start_cfg, start_loss = (load_last_best_config() if use_last_best else (None, None))
    start_gen = 0
    if AUG_SEARCH_LOG_PATH.exists():
        try:
            with open(AUG_SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # infer last generation index from history
            if "history" in data and data["history"]:
                start_gen = max(item.get("generation", -1) for item in data["history"]) + 1
        except Exception:
            start_gen = 0
    return ga_like_aug_search(
        num_generations=num_generations,
        population_size=population_size,
        search_epochs=search_epochs,
        train_dir=train_dir,
        val_dir=val_dir,
        start_config=start_cfg,
        start_best_loss=start_loss,
        start_generation=start_gen,
    )

run_aug_search_with_resume(
    num_generations = 3,
    population_size = 12,
    search_epochs = 5,
    use_last_best = True,
)
