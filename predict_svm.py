import pickle
import numpy as np
from PIL import Image
import sys

# ===== 配置 =====
model_pkl_path = "svm_pixel_model.pkl"   # 训练时保存的文件
# =================

# 1. 加载模型 + scaler + 配置
with open(model_pkl_path, "rb") as f:
    data = pickle.load(f)

svm = data["model"]
scaler = data["scaler"]
config = data["config"]
img_size = tuple(config["img_size"])  # 比如 (128, 128)

def preprocess_image(img_path):
    """把单张图片处理成模型需要的一行特征向量"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = arr.flatten().reshape(1, -1)      # shape: (1, n_features)
    x_scaled = scaler.transform(x)        # 一定要用训练时的 scaler
    return x_scaled

def predict_image(img_path):
    x = preprocess_image(img_path)
    y_pred = svm.predict(x)[0]            # 0 或 1
    # SVM 的决策函数值（可以理解为“距离分界面的远近”）
    score = svm.decision_function(x)[0]

    label_str = "benign (良性)" if y_pred == 0 else "malignant (恶性)"
    print(f"图像: {img_path}")
    print(f"预测结果: {label_str}")
    print(f"decision score: {score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict_svm.py path/to/image.png")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)


#python predict_svm.py /path/to/your/tumor_image.png

