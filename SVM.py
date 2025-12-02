import os
import re
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    hinge_loss,
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
)
from sklearn.utils import shuffle

# ============== 配置区域 ==================
data_root = r"./data_split"  # <<< 改成你的路径

IMG_SIZE = (128, 128)
TRAIN_SPLIT_NAME = "train"
TEST_SPLIT_NAME = "test"

val_ratio = 0.2
random_state = 42

n_epochs = 20
batch_size = 64

output_pickle = "svm_pixel_model.pkl"
cm_fig_path = "svm_confusion_matrix_test.png"
# =======================================


def load_images_from_split(split_name, img_size=(128, 128)):
    X = []
    y = []

    for cls_name, label in [("benign", 0), ("malignant", 1)]:
        folder = os.path.join(data_root, split_name, cls_name)
        if not os.path.isdir(folder):
            print(f"[警告] 找不到目录: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                fpath = os.path.join(root, fname)
                img = Image.open(fpath).convert("RGB")
                img = img.resize(img_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"[{split_name}] 加载完成: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y


def plot_cm_with_metrics(cm, acc, recall, f1, fig_path, title="Confusion Matrix"):
    """画混淆矩阵 + 在图里标注 Accuracy / Recall / F1，并保存。"""
    import matplotlib.pyplot as plt
    import numpy as np

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
                fontsize=14,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label", fontsize=13)
    plt.xlabel("Predicted label", fontsize=13)

    # 在图里加一个文本框写指标（测试集）
    text_str = f"Accuracy: {acc:.3f}\nRecall:   {recall:.3f}\nF1 score: {f1:.3f}"
    plt.gcf().text(
        0.98,
        0.02,
        text_str,
        fontsize=12,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 测试集混淆矩阵图已保存为: {fig_path}")


# 1. 加载数据
X_train_full, y_train_full = load_images_from_split(TRAIN_SPLIT_NAME, IMG_SIZE)
X_test, y_test = load_images_from_split(TEST_SPLIT_NAME, IMG_SIZE)

# 2. 从 train 中划分 validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=val_ratio,
    stratify=y_train_full,
    random_state=random_state,
)

print(f"真正用于训练的样本数: {X_train.shape[0]}")
print(f"用于验证的样本数:     {X_val.shape[0]}")

# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. 定义线性 SVM
svm = SGDClassifier(
    loss="hinge",
    alpha=1e-4,
    learning_rate="optimal",
    random_state=random_state,
)

classes = np.array([0, 1])

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

n_samples = X_train_scaled.shape[0]

# 5. 训练循环
for epoch in range(1, n_epochs + 1):
    X_epoch, y_epoch = shuffle(
        X_train_scaled, y_train, random_state=(random_state + epoch)
    )

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        X_batch = X_epoch[start:end]
        y_batch = y_epoch[start:end]

        if epoch == 1 and start == 0:
            svm.partial_fit(X_batch, y_batch, classes=classes)
        else:
            svm.partial_fit(X_batch, y_batch)

    # 计算 train / val loss & acc
    train_decision = svm.decision_function(X_train_scaled)
    val_decision = svm.decision_function(X_val_scaled)

    train_loss = hinge_loss(y_train, train_decision)
    val_loss = hinge_loss(y_val, val_decision)

    y_train_pred = svm.predict(X_train_scaled)
    y_val_pred = svm.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{n_epochs} - "
        f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
        f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}"
    )

# 6. 在测试集上评估 + 混淆矩阵和指标
y_test_pred = svm.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred, average="binary", pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, average="binary", pos_label=1)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"\n[TEST] accuracy: {test_acc:.4f}")
print(f"[TEST] recall (malignant as positive): {test_recall:.4f}")
print(f"[TEST] F1 score: {test_f1:.4f}")
print("\n[TEST] 混淆矩阵:")
print(cm_test)

# 7. 保存模型 + scaler + 历史 + 测试指标
results = {
    "model": svm,
    "scaler": scaler,
    "history": history,
    "test_metrics": {
        "accuracy": test_acc,
        "recall": test_recall,
        "f1": test_f1,
        "confusion_matrix": cm_test,
    },
    "config": {
        "img_size": IMG_SIZE,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "val_ratio": val_ratio,
        "random_state": random_state,
    },
}

with open(output_pickle, "wb") as f:
    pickle.dump(results, f)

print(f"\n✅ SVM 模型和训练历史已保存到: {output_pickle}")

# 8. 画并保存测试集混淆矩阵图（带指标）
plot_cm_with_metrics(
    cm_test,
    acc=test_acc,
    recall=test_recall,
    f1=test_f1,
    fig_path=cm_fig_path,
    title="SVM Confusion Matrix (Test)",
)
