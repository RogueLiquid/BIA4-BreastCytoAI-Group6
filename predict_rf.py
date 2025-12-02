import pickle
import numpy as np
from PIL import Image
import sys

# ===== 配置 =====
model_pkl_path = "rf_pixel_model.pkl"   # 训练时保存的文件
# =================

# 1. 加载模型 + 配置
with open(model_pkl_path, "rb") as f:
    data = pickle.load(f)

rf = data["model"]
config = data["config"]
img_size = tuple(config["img_size"])  # 比如 (128, 128)

def preprocess_image(img_path):
    """把单张图片处理成模型需要的一行特征向量（与训练时一致）"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = arr.flatten().reshape(1, -1)
    return x

def predict_image(img_path):
    x = preprocess_image(img_path)
    y_pred = rf.predict(x)[0]        # 0 或 1
    # 随机森林可以输出类别概率
    prob = rf.predict_proba(x)[0]    # [p(benign), p(malignant)]

    label_str = "benign (良性)" if y_pred == 0 else "malignant (恶性)"
    print(f"图像: {img_path}")
    print(f"预测结果: {label_str}")
    print(f"probabilities: benign={prob[0]:.4f}, malignant={prob[1]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict_rf.py path/to/image.png")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)
