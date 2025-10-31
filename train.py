import os, re, glob, random
import pandas as pd

DATA_DIR = "BreaKHis 400X" 

def collect_breakhis_400x(data_dir):
    paths = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    rows = []
    for p in paths:
        lp = p.lower()
        if not lp.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        if "400x" not in lp:
            continue
        if "benign" in lp:
            label = "benign"
        elif "malignant" in lp:
            label = "malignant"
        else:
            continue
        rows.append({"path": p, "label": label})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

df = collect_breakhis_400x(DATA_DIR)
print(df.head(), "\n")
print("Counts:\n", df['label'].value_counts())
print("Total images:", len(df))

import cv2
import matplotlib.pyplot as plt

def show_samples(df, n=8):
    subset = df.sample(min(n, len(df)), random_state=1).reset_index(drop=True)
    cols = 4
    rows = (len(subset) + cols - 1)//cols
    plt.figure(figsize=(12, 3*rows))
    for i, row in subset.iterrows():
        img = cv2.imread(row['path'])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, (i % (rows*cols)) + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(row['label'])
    plt.tight_layout()
    plt.show()

show_samples(df, n=12)

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.exposure import rescale_intensity


def safe_read_rgb(path):
    img = cv2.imread(path)
    if img is None: 
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_features(img_rgb):
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

# Build feature table
features, labels, keep_paths = [], [], []
for i, row in df.iterrows():
    img = safe_read_rgb(row["path"])
    if img is None:
        continue
    features.append(extract_features(img))
    labels.append(1 if row["label"] == "malignant" else 0)
    keep_paths.append(row["path"])

X = pd.DataFrame(features)
y = pd.Series(labels, name="label")
print(X.shape, y.value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --- SVM pipeline
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale", random_state=42))
])

svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_proba = svm_clf.predict_proba(X_test)[:,1]

print("=== SVM Results ===")
print(classification_report(y_test, svm_pred, target_names=["benign","malignant"]))
print("ROC-AUC:", roc_auc_score(y_test, svm_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, svm_pred))

# --- Random Forest
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]

print("\n=== RandomForest Results ===")
print(classification_report(y_test, rf_pred, target_names=["benign","malignant"]))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, rf_pred))

import torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm
import numpy as np

class BreakHisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.targets = (self.df['label'] == "malignant").astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert("RGB")
        if self.transform: img = self.transform(img)
        label = 1 if row['label']=="malignant" else 0
        return img, label

# Train/val split for CNN (keep previous test set for fair comparison)
df_ = df.loc[[p in set(keep_paths) for p in df['path']]]
df_trainval, df_test_cnn = train_test_split(df_, test_size=0.20, random_state=42, stratify=df_['label'])

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_ds = BreakHisDataset(df_trainval, transform=train_tfms)
val_ds   = BreakHisDataset(df_test_cnn, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Simple CNN (fast to train, fine to start)
class SmallCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28x28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.net(x)
        return self.head(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallCNN(n_classes=2).to(device)

# Class weights (handle imbalance)
cnt = Counter((df_trainval['label']=="malignant").astype(int).values)
total = cnt[0] + cnt[1]
weights = torch.tensor([total/(2*cnt[0]), total/(2*cnt[1])], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, loader):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    preds = all_logits.argmax(1).numpy()
    probs = torch.softmax(all_logits, dim=1)[:,1].numpy()
    return preds, probs, all_y.numpy()

EPOCHS = 8
best_val_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # val
    preds, probs, ytrue = evaluate(model, val_loader)
    acc = (preds == ytrue).mean()
    print(f"Epoch {epoch} | TrainLoss {running_loss/len(train_loader):.4f} | ValAcc {acc:.4f}")

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

preds, probs, ytrue = evaluate(model, val_loader)
print("=== CNN Results ===")
print(classification_report(ytrue, preds, target_names=["benign","malignant"]))
try:
    print("ROC-AUC:", roc_auc_score(ytrue, probs))
except:
    pass
print("Confusion matrix:\n", confusion_matrix(ytrue, preds))