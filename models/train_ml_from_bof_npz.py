import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

# =========================
# CONFIG (edit if needed)
# =========================
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_NPZ = REPO_ROOT / "BoF_cache_fcm" / "bof_features_train.npz"
VAL_NPZ   = REPO_ROOT / "BoF_cache_fcm" / "bof_features_validation.npz"
RANDOM_STATE = 0


# =========================
# I/O with strict checks
# =========================
def filter_original_only(X, y, paths):
    """
    Keep ONLY original (non-augmented) samples.
    """
    mask = np.array(
        ["_augment" not in Path(p).stem for p in paths],
        dtype=bool
    )
    return X[mask], y[mask], paths[mask]


def load_bof_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    required = {"X", "y", "paths"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(
            f"{npz_path} missing keys {missing}. Found keys: {data.files}. "
            "Expected keys are exactly: X, y, paths."
        )

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    paths = np.asarray(data["paths"]).astype(str)

    if X.ndim != 2:
        raise ValueError(f"{npz_path}: X must be 2D [N,d], got {X.shape}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"{npz_path}: y must be [N], got {y.shape}, X has N={X.shape[0]}")
    if paths.ndim != 1 or len(paths) != X.shape[0]:
        raise ValueError(f"{npz_path}: paths must be [N], got {paths.shape}, X has N={X.shape[0]}")

    # sanity: expecting binary labels 0/1
    uniq = np.unique(y)
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"{npz_path}: expected binary labels in {{0,1}}, got unique labels {uniq}")

    return X, y, paths


# =========================
# Paper ML models
# =========================
class ClassConditionalGMM:
    """
    GMM classifier consistent with the paper's GMM description:
    model P(x | class) with mixture Gaussian, classify via Bayes rule.
    """
    def __init__(self, n_components=2, covariance_type="full", reg_covar=1e-6, max_iter=200, random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.gmms = {}
        self.priors = {}

    def fit(self, X, y):
        for c in [0, 1]:
            Xc = X[y == c]
            if len(Xc) == 0:
                raise RuntimeError(f"GMM: no samples for class {c}")
            gmm = GaussianMixture(
                n_components=self.n_components,      # NOT specified by paper
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            gmm.fit(Xc)
            self.gmms[c] = gmm
            self.priors[c] = float(np.mean(y == c))
        return self

    def predict_proba(self, X):
        logp = []
        for c in [0, 1]:
            lp = self.gmms[c].score_samples(X) + np.log(self.priors[c] + 1e-12)
            logp.append(lp)
        logp = np.vstack(logp).T
        m = logp.max(axis=1, keepdims=True)
        p = np.exp(logp - m)
        p = p / p.sum(axis=1, keepdims=True)
        return p  # columns [class0, class1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)


def build_models(random_state=0) -> Dict[str, object]:
    """
    5 non-deep-learning classifiers reported in the paper:
    GMM, DT, SDC, SVM-RBF, NBC.
    """
    models = {}

    models["GMM"] = ClassConditionalGMM(
        n_components=2,          # paper does not specify -> must choose
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=200,
        random_state=random_state,
    )

    models["DT"] = DecisionTreeClassifier(
        criterion="entropy",     # aligns with information gain in paper
        random_state=random_state,
    )

    # SDC (softmax discriminant classifier) ≈ logistic regression with softmax
    models["SDC"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    models["SVM-RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    models["NBC"] = GaussianNB()  # paper does not specify NB type

    return models


# =========================
# Metrics (paper-style)
# =========================
def tp_tn_fp_fn(y_true, y_pred) -> Tuple[int, int, int, int]:
    # label 1 = malignant, 0 = benign
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # [[TP, FN],
    #  [FP, TN]]
    tp = int(cm[0, 0]); fn = int(cm[0, 1])
    fp = int(cm[1, 0]); tn = int(cm[1, 1])
    return tp, tn, fp, fn

def mse_prob(y_true, p_malignant) -> float:
    # probability MSE: mean((y - p)^2)
    y_true = y_true.astype(np.float32)
    p_malignant = p_malignant.astype(np.float32)
    return float(np.mean((y_true - p_malignant) ** 2))

def get_proba_class1(model, X):
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model)} has no predict_proba()")
    proba = model.predict_proba(X)
    # sklearn proba columns align with model.classes_
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if classes == [0, 1]:
            return proba[:, 1]
        if classes == [1, 0]:
            return proba[:, 0]
    # custom GMM returns [p0, p1]
    return proba[:, 1]


# =========================
# Run: train -> validate
# =========================
X_train, y_train, p_train = load_bof_npz(TRAIN_NPZ)
X_val, y_val, p_val = load_bof_npz(VAL_NPZ)

# ✅ KEEP ONLY ORIGINAL (NON-AUGMENTED) IMAGES
X_train, y_train, p_train = filter_original_only(X_train, y_train, p_train)
X_val, y_val, p_val = filter_original_only(X_val, y_val, p_val)

print(f"[Train] Original only: {len(y_train)} samples")
print(f"[Val]   Original only: {len(y_val)} samples")
print(f"[Train] Class balance: benign={np.sum(y_train==0)}, malignant={np.sum(y_train==1)}")
print(f"[Val]   Class balance: benign={np.sum(y_val==0)}, malignant={np.sum(y_val==1)}")

models = build_models(random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    tp, tn, fp, fn = tp_tn_fp_fn(y_val, y_pred)

    p1 = get_proba_class1(model, X_val)
    m = mse_prob(y_val, p1)

    print("=" * 72)
    print(f"{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"MSE(prob): {m:.6e}")
