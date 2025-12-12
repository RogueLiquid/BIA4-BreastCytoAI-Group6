import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from skimage.feature import graycomatrix, graycoprops

# ====================== 路径配置 ======================
DATA_DIR = r"C:\Users\86188\Desktop\BIA\ica\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

SAVE_DIR = r"C:\Users\86188\Desktop\BIA\ica"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vit_radiomics_best.pt")
RESULTS_PATH    = os.path.join(SAVE_DIR, "vit_radiomics_results.pkl")

IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES: List[str] = ["benign", "malignant"]  # 0,1


# ====================== 工具函数 ======================
def set_seed(seed: int = 666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device

def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ====================== Radiomics（GLCM + First-order） ======================
def extract_radiomics(img: Image.Image) -> np.ndarray:
    gray = np.asarray(img.convert("L"), dtype=np.uint8).copy()

    glcm = graycomatrix(
        gray,
        distances=[1, 2, 4],
        angles=[0, np.pi/4, np.pi/2],
        levels=256,
        symmetric=True,
        normed=True
    )

    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    features = []
    for p in props:
        features.extend(graycoprops(glcm, p).flatten())

    gray_f = gray.astype(np.float32)
    features.extend([
        gray_f.mean(),
        gray_f.std(),
        gray_f.min(),
        gray_f.max(),
        np.percentile(gray_f, 25),
        np.percentile(gray_f, 75),
    ])

    return np.array(features, dtype=np.float32)

class RadiomicsAlignedDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.paths: List[Path] = []
        self.labels: List[int] = []

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])

        root_path = Path(root)
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = root_path / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")

            for p in sorted(class_dir.glob("*.png")):
                self.paths.append(p)
                self.labels.append(idx)

        print(f"[Dataset] {root} -> {len(self.paths)} samples.")

        # Radiomics 缓存（只做一次）
        print(f"[Radiomics] Extracting {len(self.paths)} radiomics features (only once)...")
        rad_list = []
        for p in self.paths:
            img = Image.open(p).convert("RGB")
            rad_list.append(extract_radiomics(img))
        self.rad_cache = np.asarray(rad_list, dtype=np.float32)
        print("[Radiomics] Done.\n")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = cast(torch.Tensor, self.transform(img))

        rad = torch.tensor(self.rad_cache[idx], dtype=torch.float32)
        return img_tensor, label, rad

class HybridModel(nn.Module):
    def __init__(self, backbone="vit_base_patch16_224", rad_dim=60, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,  
            num_classes=0    
        )
        self.feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + rad_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, rad):
        feat = self.backbone.forward_features(x)
        if feat.dim() == 3:  # (B,197,768)
            feat = feat[:, 0]
        return self.classifier(torch.cat([feat, rad], dim=1))

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0.0, path=BEST_MODEL_PATH):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int, model: nn.Module):
        if val_loss + self.delta < self.best_score:
            self.best_score = val_loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"  ✔ Saved new best model to {self.path}")
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp=True):
    model.train()
    total_loss = 0.0
    for imgs, labels, rad in loader:
        imgs = imgs.to(device)
        rad = rad.to(device)
        labels_t = torch.tensor(labels, device=device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(imgs, rad)
            loss = criterion(outputs, labels_t)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels, rad in loader:
            imgs = imgs.to(device)
            rad = rad.to(device)
            labels_t = torch.tensor(labels, device=device)

            with autocast(enabled=use_amp):
                outputs = model(imgs, rad)
                loss = criterion(outputs, labels_t)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)

            correct += preds.eq(labels_t).sum().item()
            total += labels_t.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(labels_t.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total
    f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
    return avg_loss, acc, f1


def split_indices(n: int, val_ratio: float = 0.2, seed: int = 666):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


# ====================== main ======================
def main():
    set_seed(666)
    device = get_device()
    use_amp = device.type == "cuda"

    full_train_ds = RadiomicsAlignedDataset(TRAIN_DIR, train=True)

    train_idx, val_idx = split_indices(len(full_train_ds), val_ratio=0.2, seed=666)
    train_ds = Subset(full_train_ds, train_idx)

    full_train_ds_eval = RadiomicsAlignedDataset(TRAIN_DIR, train=False)
    val_ds = Subset(full_train_ds_eval, val_idx)


    test_ds = RadiomicsAlignedDataset(TEST_DIR, train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(full_train_ds.labels).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    rad_dim = int(full_train_ds.rad_cache.shape[1])
    print("Radiomics dim:", rad_dim)

    model = HybridModel(rad_dim=rad_dim, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(patience=7, verbose=True, path=BEST_MODEL_PATH)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_epoch = 0

    print("Start training ...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"[Epoch {epoch:02d}/{num_epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_f1(pos=mal)={val_f1:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6e}"
        )

        early_stopping(val_loss, epoch, model)
        if early_stopping.best_epoch:
            best_epoch = early_stopping.best_epoch
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    if best_epoch == 0:
        best_epoch = len(history["train_loss"])
    print(f"Best VAL epoch: {best_epoch}")

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        print("WARNING: best checkpoint not found, using final weights.")

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, use_amp)
    print(f"\nFinal TEST (on data/test): loss={test_loss:.4f}, acc={test_acc:.4f}, f1(pos=mal)={test_f1:.4f}")

    results = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "val_f1": history["val_f1"],  
        "best_val_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "class_names": CLASS_NAMES,
        "rad_dim": rad_dim,
        "val_ratio": 0.2,
        "seed": 666,
    }

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Saved training results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
