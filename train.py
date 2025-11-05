import os, re, glob, random
import pandas as pd
import cv2
import numpy as np
import torch, torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm

from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.exposure import rescale_intensity
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# --- Custom Transform for Aspect-Ratio-Preserving Resize + Pad ---
class ResizeWithPad:
    """
    A custom PyTorch transform that resizes an image to fit within a
    target size while maintaining aspect ratio, then pads the rest with 0.
    This replicates tf.image.resize_with_pad.
    """
    def __init__(self, target_size, fill=0):
        # target_size is (width, height)
        if isinstance(target_size, int):
            self.target_w, self.target_h = target_size, target_size
        else:
            self.target_w, self.target_h = target_size
        self.fill = fill

    def __call__(self, img):
        # img is a PIL Image
        # img.size is (width, height)
        w, h = img.size

        # Calculate ratio
        ratio = min(self.target_w / w, self.target_h / h)
        
        # New size
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Resize
        # Use LANCZOS (formerly ANTIALIAS) for high-quality downsampling
        img = img.resize((new_w, new_h), Image.LANCZOS) 
        
        # Create a new image with padding
        new_img = Image.new("RGB", (self.target_w, self.target_h), self.fill)
        
        # Paste the resized image into the center
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
# --- END Custom Transform ---


DATA_DIR = "BreaKHis 400X" 

# --- Data Collection Functions ---
def collect_breakhis_400x(data_dir):
    """
    Scans the data directory, filters for 400X images, 
    and extracts the path, label, and a unique patient_id.
    """
    paths = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    rows = []
    print(f"Scanning {len(paths)} total files...")
    
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
            
        fname = os.path.basename(p)
        parts = fname.split('-')
        
        if len(parts) >= 3: 
            patient_id = '-'.join(parts[:-2]) 
        else:
            print(f"Warning: Skipping file with unexpected format: {fname}")
            continue

        rows.append({"path": p, "label": label, "patient_id": patient_id})
            
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"Found {len(df)} images belonging to {df['patient_id'].nunique()} unique patients/biopsies.")
    return df

def safe_read_rgb(path):
    img = cv2.imread(path)
    if img is None: 
        print(f"Warning: Could not read {path}, skipping.")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Feature Extraction Function ---
def extract_features(img_rgb):
    feats = {}
    H, W = 256, 256
    img = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    img_hsv = (rgb2hsv(img / 255.0) * 255.0).astype(np.uint8)
    for space_name, arr in [("rgb", img), ("hsv", img_hsv)]:
        for c, cname in enumerate(["c1","c2","c3"]):
            ch = arr[..., c].astype(np.float32)
            feats[f"{space_name}_{cname}_mean"] = ch.mean()
            feats[f"{space_name}_{cname}_std"]  = ch.std()

    gray = (rgb2gray(img) * 255).astype(np.uint8)
    feats["gray_mean"] = gray.mean()
    feats["gray_std"]  = gray.std()
    feats["gray_min"]  = gray.min()
    feats["gray_max"]  = gray.max()
    
    hist_prob = np.histogram(gray, bins=32, range=(0, 255), density=True)[0] + 1e-12
    feats["gray_entropy"] = -np.sum(hist_prob * np.log2(hist_prob))

    q = np.floor(rescale_intensity(gray, in_range="image", out_range=(0,7))).astype(np.uint8)
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(q, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).ravel()
        feats[f"glcm_{prop}_mean"] = vals.mean()
        feats[f"glcm_{prop}_std"]  = vals.std()

    P, R = 8, 1
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    for i, v in enumerate(hist):
        feats[f"lbp_u{i}"] = float(v)

    return feats

# --- 1. Data Loading and Splitting ---
df = collect_breakhis_400x(DATA_DIR)
print(df.head(), "\n")
print("Counts:\n", df['label'].value_counts())
print("Total images:", len(df))

features, labels, keep_indices = [], [], []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    img = safe_read_rgb(row["path"])
    if img is None:
        continue
    features.append(extract_features(img))
    labels.append(1 if row["label"] == "malignant" else 0)
    keep_indices.append(i)

X = pd.DataFrame(features)
y = pd.Series(labels, name="label")
df_clean = df.loc[keep_indices].reset_index(drop=True)
groups = df_clean['patient_id'] 

print(f"\nProcessed {len(df_clean)} images successfully.")
print(f"X shape: {X.shape}, y counts:\n{y.value_counts()}")
print(f"Group series length: {len(groups)}, Unique groups: {groups.nunique()}")

print("\nSplitting data based on patient ID to prevent leakage...")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
df_trainval = df_clean.iloc[train_idx]
df_test_cnn = df_clean.iloc[test_idx]

print(f"Total unique patients: {groups.nunique()}")
print(f"Train set: {len(X_train)} images from {df_trainval['patient_id'].nunique()} patients.")
print(f"Test set:  {len(X_test)} images from {df_test_cnn['patient_id'].nunique()} patients.")
train_patients = set(df_trainval['patient_id'])
test_patients = set(df_test_cnn['patient_id'])
leakage = train_patients.intersection(test_patients)
if len(leakage) == 0:
    print("✅ Data leakage check passed: No patients overlap between train and test.")
else:
    print(f"❌ CRITICAL ERROR: {len(leakage)} patients leaked into test set!")


# --- 2. Traditional ML Pipeline ---
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, gamma="scale", random_state=42))
])
print("\nTraining SVM...")
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_proba = svm_clf.predict_proba(X_test)[:,1]
print("=== SVM Results ===")
print(classification_report(y_test, svm_pred, target_names=["benign","malignant"]))
print("ROC-AUC:", roc_auc_score(y_test, svm_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, svm_pred))

rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]
print("\n=== RandomForest Results ===")
print(classification_report(y_test, rf_pred, target_names=["benign","malignant"]))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, rf_pred))


# --- 3. Deep Learning Pipeline ---

class BreakHisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.targets = (self.df['label'] == "malignant").astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert("RGB")
        except Exception as e:
            print(f"Error loading image {row['path']}: {e}")
            return torch.randn(3, 224, 224), 0 
            
        if self.transform: 
            img = self.transform(img)
            
        label = self.targets[idx]
        return img, label

# --- Use ResizeWithPad in transforms ---
train_tfms = transforms.Compose([
    ResizeWithPad((224, 224)), # <-- CORRECTED RESIZING
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    ResizeWithPad((224, 224)), # <-- CORRECTED RESIZING
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\nCreating CNN DataLoaders...")
train_ds = BreakHisDataset(df_trainval, transform=train_tfms)
val_ds   = BreakHisDataset(df_test_cnn, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

class SmallCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),    # 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),   # 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2), # 28x28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),                  # 28x28
            nn.AdaptiveAvgPool2d((1,1)),
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
print(f"Training CNN model on {device}")
model = SmallCNN(n_classes=2).to(device)

cnt = Counter(train_ds.targets)
if cnt[0] == 0 or cnt[1] == 0:
    print("Warning: Training set has only one class. Weights not applied.")
    weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
else:
    total = cnt[0] + cnt[1]
    weights = torch.tensor([total/(2*cnt[0]), total/(2*cnt[1])], dtype=torch.float32).to(device)
    print(f"Using class weights: {weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

def evaluate(model, loader):
    model.eval()
    all_logits, all_y = [], []
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb.long() # <-- FIX: Cast labels to long
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    preds = all_logits.argmax(1).numpy()
    probs = torch.softmax(all_logits, dim=1)[:,1].numpy()
    return preds, probs, all_y.numpy(), val_loss / len(loader)

# --- 4. CNN Training & Evaluation ---
WEIGHTS_DIR = "weights"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "best_model_weights.pth")
os.makedirs(WEIGHTS_DIR, exist_ok=True) 

needs_training = True
if os.path.exists(WEIGHTS_FILE):
    print(f"\nFound existing weights at {WEIGHTS_FILE}. Attempting to load...")
    try:
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
        print("Weights loaded successfully. Skipping training.")
        needs_training = False
    except Exception as e:
        print(f"Warning: Could not load weights ({e}). Training from scratch.")

if needs_training:
    EPOCHS = 10
    best_val_auc = 0.0
    print("\nTraining CNN...")

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            yb = yb.long() # <-- FIX: Cast labels to long
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        preds, probs, ytrue, val_loss = evaluate(model, val_loader)
        acc = (preds == ytrue).mean()
        try:
            auc = roc_auc_score(ytrue, probs) # <-- FIX: Corrected typo
        except ValueError:
            auc = 0.0
            
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch} | TrainLoss {running_loss/len(train_loader):.4f} | ValLoss {val_loss:.4f} | ValAcc {acc:.4f} | ValAUC {auc:.4f}")
        
        if auc > best_val_auc:
            print(f"New best ValAUC: {auc:.4f}. Saving model to {WEIGHTS_FILE}...")
            best_val_auc = auc
            torch.save(model.state_dict(), WEIGHTS_FILE)

print("\nEvaluating final CNN model on test set...")
preds, probs, ytrue, _ = evaluate(model, val_loader)

print("=== CNN Results ===")
print(classification_report(ytrue, preds, target_names=["benign","malignant"]))

try:
    print("ROC-AUC:", roc_auc_score(ytrue, probs))
except ValueError as e:
    print(f"Could not calculate ROC-AUC. Reason: {e}")

print("Confusion matrix:\n", confusion_matrix(ytrue, preds))