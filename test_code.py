import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class BreastCancerDataset(Dataset):
    def __init__(
        self,
        root_dir,
        resize=(224, 224),
        compute_stats=False,
        mean=None,
        std=None,
        max_stats_samples=None,
        print_stats=True,
    ):
        """
        root_dir:           path to class subfolders (e.g. ./BreaKHis_400X/train)
        resize:             target size for Resize
        compute_stats:      if True, compute dataset mean/std in [0,1]
        mean, std:          optional precomputed mean/std (lists or tensors of len 3)
        max_stats_samples:  limit number of images for stats (None = all)
        print_stats:        if True, print stats after computing (for mean/std)
        """
        self.img_paths = []
        self.labels = []

        # Only use subdirectories as classes
        self.classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        for i, c in enumerate(self.classes):
            imgs = glob(os.path.join(root_dir, c, "*.png"))
            self.img_paths.extend(imgs)
            self.labels.extend([i] * len(imgs))

        self.resize = resize

        # Base transform for "input" stats: Resize + ToTensor (no Normalize)
        self._base_tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize),
            transforms.ToTensor(),  # --> [0,1]
        ])

        # Decide mean/std for normalization
        if compute_stats:
            self.mean, self.std = self._compute_mean_std(
                max_samples=max_stats_samples,
                print_stats=print_stats,
            )
        else:
            if mean is None or std is None:
                raise ValueError(
                    "Either set compute_stats=True or provide mean and std."
                )
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)

        # Final transform: Resize -> ToTensor -> Normalize(dataset_mean, dataset_std)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        img = self.transform(img)
        return img, label

    def _compute_mean_std(self, max_samples=None, print_stats=True):
        """
        Compute per-channel mean and std in [0,1] on resized images
        using self._base_tfms (Resize + ToTensor), i.e. BEFORE Normalize.
        """
        n_total = len(self.img_paths)
        n_use = n_total if max_samples is None else min(max_samples, n_total)

        sum_c = torch.zeros(3, dtype=torch.float64)
        sum_sq_c = torch.zeros(3, dtype=torch.float64)
        count = 0

        if print_stats:
            print(f"\n[BreastCancerDataset] Computing mean/std on {n_use}/{n_total} images ...")

        for i in range(n_use):
            p = self.img_paths[i]
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = self._base_tfms(img)  # [C,H,W], float in [0,1]

            sum_c += t.sum(dim=[1, 2]).double()
            sum_sq_c += (t ** 2).sum(dim=[1, 2]).double()
            count += t.shape[1] * t.shape[2]

        mean = sum_c / count
        var = (sum_sq_c / count) - mean ** 2
        std = torch.sqrt(var)

        if print_stats:
            print("Mean per channel (in [0,1]):", mean.tolist())
            print("Std  per channel (in [0,1]):", std.tolist())
            print("============================================\n")

        return mean.float(), std.float()

    def compute_input_and_normalized_stats(self, max_samples=None, print_stats=True):
        """
        Compute per-channel (R,G,B) min, max, mean for:
          - 'input'   : after Resize+ToTensor (self._base_tfms), in [0,1]
          - 'normalized': after full self.transform (includes Normalize)

        max_samples:   number of images to use (None = all)
        print_stats:   if True, print results

        Returns:
            {
              "input":      {"min": [..], "max": [..], "mean": [..]},
              "normalized": {"min": [..], "max": [..], "mean": [..]},
            }
        """
        n_total = len(self.img_paths)
        n_use = n_total if max_samples is None else min(max_samples, n_total)

        # Input stats (after Resize+ToTensor, before Normalize)
        in_min = torch.full((3,), float("inf"), dtype=torch.float64)
        in_max = torch.full((3,), float("-inf"), dtype=torch.float64)
        in_sum = torch.zeros(3, dtype=torch.float64)
        in_count = 0

        # Normalized stats (after full transform)
        norm_min = torch.full((3,), float("inf"), dtype=torch.float64)
        norm_max = torch.full((3,), float("-inf"), dtype=torch.float64)
        norm_sum = torch.zeros(3, dtype=torch.float64)
        norm_count = 0

        if print_stats:
            print(f"\n[BreastCancerDataset] Computing input & normalized stats on {n_use}/{n_total} images ...")

        for i in range(n_use):
            p = self.img_paths[i]
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ----- input (pre-normalization) -----
            inp = self._base_tfms(img)  # [C,H,W], in [0,1]
            in_min = torch.minimum(in_min, inp.amin(dim=[1, 2]).double())
            in_max = torch.maximum(in_max, inp.amax(dim=[1, 2]).double())
            in_sum += inp.sum(dim=[1, 2]).double()
            in_count += inp.shape[1] * inp.shape[2]

            # ----- normalized (post-normalization) -----
            out = self.transform(img)   # [C,H,W], normalized
            norm_min = torch.minimum(norm_min, out.amin(dim=[1, 2]).double())
            norm_max = torch.maximum(norm_max, out.amax(dim=[1, 2]).double())
            norm_sum += out.sum(dim=[1, 2]).double()
            norm_count += out.shape[1] * out.shape[2]

        in_mean = in_sum / in_count
        norm_mean = norm_sum / norm_count

        stats = {
            "input": {
                "min": in_min.tolist(),
                "max": in_max.tolist(),
                "mean": in_mean.tolist(),
            },
            "normalized": {
                "min": norm_min.tolist(),
                "max": norm_max.tolist(),
                "mean": norm_mean.tolist(),
            },
        }

        if print_stats:
            print("=== INPUT (after Resize+ToTensor, in [0,1]) ===")
            print("min  per channel:", stats["input"]["min"])
            print("max  per channel:", stats["input"]["max"])
            print("mean per channel:", stats["input"]["mean"])

            print("\n=== NORMALIZED (after full transform) ===")
            print("min  per channel:", stats["normalized"]["min"])
            print("max  per channel:", stats["normalized"]["max"])
            print("mean per channel:", stats["normalized"]["mean"])
            print("=========================================\n")

        return stats

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*28*28, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VGGInspired(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*28*28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class ResNetLike(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    save_dir: str | None = None,
    save_prefix: str | None = None,
    save_every: int | None = None,
):
    """
    Train a classification model and (optionally) save checkpoints.

    Args:
        model:        nn.Module
        train_loader: DataLoader for training set
        val_loader:   DataLoader for validation set
        epochs:       total number of epochs
        lr:           learning rate for Adam

        save_dir:     directory to save model checkpoints (e.g. "./weights")
        save_prefix:  filename prefix (e.g. "SimpleCNN") -> file name will be
                      f"{save_prefix}_epoch{epoch}.pth"
        save_every:   save checkpoint every N epochs. If None, don't save
                      inside this function.

    Returns:
        train_losses: list of training loss (per epoch)
        val_losses:   list of validation loss (per epoch)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    # Prepare saving directory if requested
    do_save = (
        save_dir is not None and
        save_prefix is not None and
        save_every is not None and
        save_every > 0
    )
    if do_save:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Validation -----
        model.eval()
        val_loss, correct = 0.0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                correct += (out.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ----- Save checkpoint if requested -----
        if do_save and ((epoch + 1) % save_every == 0 or (epoch + 1) == epochs):
            filename = f"{save_prefix}_epoch{epoch+1}.pth"
            path = os.path.join(save_dir, filename)
            torch.save(model.state_dict(), path)
            print(f"💾 Saved checkpoint: {path}")

    return train_losses, val_losses

def evaluate_on_testset(model, test_loader, dataset, model_name=None, epoch=None, save=True, save_dir="."):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix on Test Set", fontsize=14)
    plt.tight_layout()

    if save and model_name is not None and epoch is not None:
        save_path = os.path.join(save_dir, f"{model_name}_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        out_path = f"{save_path}/confusion_matrix.pdf"
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        print(f"✅ Saved confusion matrix to {out_path}")
    elif not save:
        print("⚠️ Skipped saving confusion matrix (save=False)")

    plt.show()
    plt.close()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer
        self.hook()

    def hook(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def forward_hook(module, input, output):
            self.feature_map = output
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate(self, input_img, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_img)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[:, class_idx]
        score.backward()
        grads = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (grads * self.feature_map).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = cam.cpu().detach().numpy()
        return cam

def show_gradcam(img_tensor, cam):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    if cam.ndim > 2:
        cam = cam[0]

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = 0.5 * heatmap + 0.5 * img
    overlay = np.clip(overlay, 0, 1)

    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

def show_gradcam_side_by_side(img_tensor, cam, model, dataset, label_true):
    """
    显示原图与 Grad-CAM 热力图并排，并在标题标注预测与真实标签
    """
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        pred_idx = output.argmax(1).item()
    pred_label = dataset.classes[pred_idx]
    true_label = dataset.classes[label_true]

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    if cam.ndim > 2:
        cam = cam[0]
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    color = "green" if pred_label == true_label else "red"

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(f"Predicted: {pred_label} | True: {true_label}",
                 color=color, fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

def plot_curves(train_losses, val_losses, model_name=None, epoch=None, save=True, save_dir="."):
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss Curves', fontsize=16, pad=12, weight='bold')
    plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    if save and model_name is not None and epoch is not None:
        save_path = os.path.join(save_dir, f"{model_name}_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        out_path = f"{save_path}/loss_curves.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved loss curve to {out_path}")
    elif not save:
        print("⚠️ Skipped saving loss curve (save=False)")

    plt.show()
    plt.close()

def visualize_predictions(model, dataset, num_images=9):
    model.eval()
    indices = np.random.choice(len(dataset), num_images, replace=False)
    n_cols = int(np.ceil(np.sqrt(num_images)))
    n_rows = int(np.ceil(num_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.3 * n_rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for ax in axes[num_images:]:
        ax.axis('off')

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(1).item()
        img_np = np.transpose(img.cpu().numpy(), (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)  # normalize to [0,1]

        axes[i].imshow(img_np)
        # Academic professional: show class with full name, pred/true, color highlight for mistakes.
        color = "forestgreen" if pred == label else "crimson"
        axes[i].set_title(
            f"Pred: {dataset.classes[pred]}\nTrue: {dataset.classes[label]}",
            fontsize=11,
            color=color,
            fontweight='bold',
            loc='center'
        )
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[i].spines['top'].set_color(color)
        axes[i].spines['right'].set_color(color)
        axes[i].spines['left'].set_color(color)
        axes[i].spines['bottom'].set_color(color)
        axes[i].spines['top'].set_linewidth(2 if pred != label else 1)
        axes[i].spines['right'].set_linewidth(2 if pred != label else 1)
        axes[i].spines['left'].set_linewidth(2 if pred != label else 1)
        axes[i].spines['bottom'].set_linewidth(2 if pred != label else 1)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle("Model Predictions vs Ground Truth", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def build_model(
    model_name: str,
    num_classes: int,
    train_new: bool,
    weight_path: str | None = None,
):
    """
    Create a model of the given type.
    If train_new is False, initialize from weight_path instead of random weights.
    Does NOT run training.
    """
    # 1) Instantiate architecture
    if model_name == 'SimpleCNN':
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == 'VGGInspired':
        model = VGGInspired(num_classes=num_classes)
    elif model_name == 'ResNetLike':
        model = ResNetLike(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # 2) Optionally load weights
    if not train_new:
        if weight_path is None:
            raise ValueError("weight_path must be provided when train_new=False.")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Loaded weights from {weight_path}")
    else:
        print(f"✅ Initialized new {model_name} with random weights")

    return model.to(device)

data_dir = "./BreaKHis_400X_patient_split"

train_ds = BreastCancerDataset(os.path.join(data_dir, "train"), compute_stats=True, max_stats_samples=100)
test_ds = BreastCancerDataset(os.path.join(data_dir, "test"), compute_stats=True)

# this code shows the normalization of our data
train_stats = train_ds.compute_input_and_normalized_stats(
    max_samples=100,
    print_stats=True
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# this code might be used if we want to preserve the mean and std from train dataset
'''
test_ds = BreastCancerDataset(
    os.path.join(data_dir, "test"),
    resize=(224, 224),
    compute_stats=False,          # don't recompute
    mean=train_ds.mean.tolist(),  # reuse train statistics
    std=train_ds.std.tolist(),
    print_stats=False,
)
'''

model_name='SimpleCNN'

model = build_model(
    model_name=model_name,
    num_classes=2,
    train_new=True,
    weight_path=None,
)

epoch = 30
train_losses = []
val_losses = []

train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    epochs=epoch,
    lr=1e-4,
    save_dir="./weights",
    save_prefix=model_name,
    save_every=10,
)

plot_curves(train_losses, val_losses, model_name=model_name, epoch=epoch, save=True)
evaluate_on_testset(model, test_loader, test_ds, model_name=model_name, epoch=epoch, save=True)
visualize_predictions(model, test_ds, num_images=9)

def get_last_conv_name(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return name
target_layer = get_last_conv_name(model)
print("Last conv layer:", target_layer)
gradcam = GradCAM(model, target_layer=target_layer)

idx = np.random.randint(len(test_ds))
img, label_true = test_ds[idx]
cam = gradcam.generate(img.unsqueeze(0).to(device))
show_gradcam(img, cam)
show_gradcam_side_by_side(img, cam, model, test_ds, label_true)
