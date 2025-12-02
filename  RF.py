import os
import pickle

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    f1_score,
)

# ============== 配置区域 ==================
data_root = r"./data_split"  # <<< 改成你的路径

IMG_SIZE = (128, 128)
TRAIN_SPLIT_NAME = "train"
TEST_SPLIT_NAME = "test"

val_ratio = 0.2
random_state = 42

n_estimators = 200
max_depth = None

output_pickle = "rf_pixel_model.pkl"
acc_fig_path = "rf_accuracy.png"
cm_fig_path = "rf_confusion_matrix_test.png"
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

# 2. 划分 validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=val_ratio,
    stratify=y_train_full,
    random_state=random_state,
)

print(f"真正用于训练的样本数: {X_train.shape[0]}")
print(f"用于验证的样本数:     {X_val.shape[0]}")

# 3. 定义并训练随机森林
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    n_jobs=-1,
    random_state=random_state,
    class_weight="balanced_subsample",
)

print("\n开始训练随机森林...")
rf.fit(X_train, y_train)
print("训练完成。")

# 4. 评估 train / val / test
y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)
y_test_pred = rf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n[TRAIN] accuracy: {train_acc:.4f}")
print(f"[VAL]   accuracy: {val_acc:.4f}")
print(f"[TEST]  accuracy: {test_acc:.4f}")

print("\n[Validation 集分类报告]:")
print(classification_report(y_val, y_val_pred, target_names=["benign", "malignant"]))

# 5. 测试集指标（你特别要的）
test_recall = recall_score(y_test, y_test_pred, average="binary", pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, average="binary", pos_label=1)
cm_test = confusion_matrix(y_test, y_test_pred)
print("\n[TEST] 混淆矩阵:")
print(cm_test)
print(f"[TEST] recall (malignant as positive): {test_recall:.4f}")
print(f"[TEST] F1 score: {test_f1:.4f}")

# 6. 保存模型 + 结果
results = {
    "model": rf,
    "history": {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    },
    "test_metrics": {
        "accuracy": test_acc,
        "recall": test_recall,
        "f1": test_f1,
        "confusion_matrix": cm_test,
    },
    "config": {
        "img_size": IMG_SIZE,
        "val_ratio": val_ratio,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    },
}

with open(output_pickle, "wb") as f:
    pickle.dump(results, f)

print(f"\n✅ 随机森林模型和结果已保存到: {output_pickle}")

# 7. 准确率柱状图（train/val/test）
plt.figure(figsize=(5, 4))
splits = ["train", "val", "test"]
accs = [train_acc, val_acc, test_acc]
plt.bar(splits, accs)
plt.ylim(0.0, 1.0)
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy (pixel features)")
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(acc_fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ 准确率可视化图已保存为: {acc_fig_path}")

# 8. 测试集混淆矩阵图（带 Accuracy/Recall/F1）
plot_cm_with_metrics(
    cm_test,
    acc=test_acc,
    recall=test_recall,
    f1=test_f1,
    fig_path=cm_fig_path,
    title="Random Forest Confusion Matrix (Test)",
)
