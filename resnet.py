import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt

# ================== 配置区域 ==================
# 你的“按病人划分好的”数据目录：
# 结构：
#   BreakHis 400X_patient_split/
#       train/benign, train/malignant
#       test/benign,  test/malignant
data_root = r"./data_split"  # <<< 改成你的路径

img_size = 224          # ResNet 推荐 224x224
batch_size = 16
num_epochs = 20
learning_rate = 1e-4
val_ratio = 0.2
random_seed = 42

output_state_dict = "resnet50_breakhis_state.pth"
output_pickle = "resnet50_breakhis_results.pkl"
curves_fig_path = "resnet50_training_curves.png"
cm_fig_path = "resnet50_confusion_matrix_test.png"

# 设备选择：优先用 MPS（Apple 芯片），然后 CUDA，再不行用 CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
# =====================================================


# =============== 数据集定义 ==================
class TumorDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root 例如：data_root/train 或 data_root/test
        """
        self.transform = transform
        self.samples = []

        for cls_name, label in [("benign", 0), ("malignant", 1)]:
            cls_folder = os.path.join(root, cls_name)
            if not os.path.isdir(cls_folder):
                print(f"[警告] 找不到目录: {cls_folder}")
                continue

            for r, _, fnames in os.walk(cls_folder):
                for fname in fnames:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(r, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# =============== 数据增强 / 预处理 ==================
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # 简单标准化，也可以改成 ImageNet 的均值/方差
])

test_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# =============== DataLoader ==================
train_full = TumorDataset(os.path.join(data_root, "train"), transform=train_tf)
test_set  = TumorDataset(os.path.join(data_root, "test"), transform=test_tf)

# 按比例从 train_full 中划分出 val
val_size = int(len(train_full) * val_ratio)
train_size = len(train_full) - val_size

g = torch.Generator().manual_seed(random_seed)
train_set, val_set = random_split(train_full, [train_size, val_size], generator=g)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f"Train size: {train_size}")
print(f"Val size:   {val_size}")
print(f"Test size:  {len(test_set)}")


# =============== 定义 ResNet-50 ==================
def get_resnet50(num_classes=2):
    # 新版本 torchvision 写法（如果报错可以改成 pretrained=True）
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=True)

    # 替换最后一层全连接为 2 类输出
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =============== 训练函数 ==================
def train_model(model, train_loader, val_loader, num_epochs, name="ResNet50"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        total, correct = 0, 0
        train_loss_sum = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = train_loss_sum / total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_total, val_correct = 0, 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"{name} | Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

    return model, history


# =============== 测试评估 + 混淆矩阵画图 ==================
def plot_training_curves(history, fig_path, title_prefix="ResNet-50"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 训练曲线图已保存为: {fig_path}")


def plot_cm_with_metrics(cm, acc, recall, f1, fig_path, title="Confusion Matrix (Test)"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=16)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    classes = ["benign", "malignant"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label", fontsize=13)
    plt.xlabel("Predicted label", fontsize=13)

    text_str = f"Accuracy: {acc:.3f}\nRecall:   {recall:.3f}\nF1 score: {f1:.3f}"
    plt.gcf().text(
        0.98,
        0.02,
        text_str,
        fontsize=12,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 测试集混淆矩阵图已保存为: {fig_path}")


def evaluate_on_test(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[TEST] 混淆矩阵：")
    print(cm)
    print(f"[TEST] Accuracy: {acc:.4f}")
    print(f"[TEST] Recall (malignant=1): {recall:.4f}")
    print(f"[TEST] F1 score: {f1:.4f}")

    return acc, recall, f1, cm


# =================== 主流程 ===================
def main():
    # 1. 建立 ResNet-50
    resnet50 = get_resnet50(num_classes=2)
    print("\nResNet-50 模型构建完成。")

    # 2. 训练
    resnet50, history = train_model(
        resnet50, train_loader, val_loader, num_epochs, name="ResNet-50"
    )

    # 3. 训练曲线可视化
    plot_training_curves(history, curves_fig_path, title_prefix="ResNet-50")

    # 4. 测试集评估 + 混淆矩阵可视化
    test_acc, test_recall, test_f1, cm_test = evaluate_on_test(resnet50, test_loader)
    plot_cm_with_metrics(
        cm_test,
        acc=test_acc,
        recall=test_recall,
        f1=test_f1,
        fig_path=cm_fig_path,
        title="ResNet-50 Confusion Matrix (Test)",
    )

    # 5. 保存模型参数和结果
    torch.save(resnet50.state_dict(), output_state_dict)
    print(f"\n✅ ResNet-50 参数已保存到: {output_state_dict}")

    results = {
        "model_state_dict": resnet50.state_dict(),
        "history": history,
        "test_metrics": {
            "accuracy": test_acc,
            "recall": test_recall,
            "f1": test_f1,
            "confusion_matrix": cm_test,
        },
        "config": {
            "data_root": data_root,
            "img_size": img_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "val_ratio": val_ratio,
            "device": str(device),
        },
    }

    with open(output_pickle, "wb") as f:
        pickle.dump(results, f)

    print(f"✅ 训练结果和配置已保存到: {output_pickle}")
    print("\n🎉 ResNet-50 训练 & 可视化 全部完成。")


if __name__ == "__main__":
    main()
