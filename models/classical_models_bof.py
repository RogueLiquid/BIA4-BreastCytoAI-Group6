#!/usr/bin/env python3
"""
Classical models for SIFT+BoF features on BreakHis 400x.

Implements the five non-deep-learning models from:
G. S. Manivannan et al., Int. J. Comput. Intell. Syst. 18:133, 2025
("A Holistic Strategy of Modified Superpixel Segmentation and Randomized Adam
 Hyperparameter Tuning with Deep Learning Approaches for the Classification of
 Breast Cancer from BreakHis Images").  The paper uses SIFT+BoF features
 (with FCM codebook) followed by GMM, DT, SDC, SVM-RBF, and NBC.  The BoF
 features are assumed to have been saved into:

    BoF_cache_fcm/bof_features_train.npz
    BoF_cache_fcm/bof_features_validation.npz

This script loads those and trains/evaluates the classical models only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
)

RANDOM_STATE = 42
BASE_DIR = Path("../BoF_cache_fcm")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_bof_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load BoF features and labels from a .npz file.

    Expected keys (in order of preference):
        X: 2D array (n_samples, n_features)
        y: 1D array (n_samples,)

    If your saved keys are different, modify this function.
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    # Try some common names for X and y
    candidate_X = ["X", "features", "bof_features", "X_bof"]
    candidate_y = ["y", "labels", "y_labels", "targets"]

    X = y = None

    for k in candidate_X:
        if k in data:
            X = data[k]
            break
    for k in candidate_y:
        if k in data:
            y = data[k]
            break

    if X is None or y is None:
        raise KeyError(
            f"Could not find X/y arrays in {path}. "
            f"Available keys: {keys}"
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    return X, y


@dataclass
class GMMClassifier:
    """
    Supervised Gaussian Mixture Model classifier.

    We fit one GMM per class on the SIFT+BoF histograms and use Bayes rule
    with empirical class priors to obtain p(class | x). This follows the
    description of GMM as a mixture model over feature vectors. :contentReference[oaicite:3]{index=3}
    """

    n_components_per_class: int = 1
    covariance_type: str = "full"
    random_state: int = RANDOM_STATE

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMClassifier":
        self.le_ = LabelEncoder().fit(y)
        y_enc = self.le_.transform(y)

        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        self.gmms_: Dict[int, GaussianMixture] = {}

        # empirical class priors
        counts = np.bincount(y_enc)
        self.class_log_prior_ = np.log(counts / counts.sum())

        for c in range(n_classes):
            gm = GaussianMixture(
                n_components=self.n_components_per_class,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gm.fit(X[y_enc == c])
            self.gmms_[c] = gm

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log p(x, class=c) for each class c."""
        log_jll = []
        for c, gm in self.gmms_.items():
            # score_samples returns log p(x | class=c)
            log_px_given_c = gm.score_samples(X)
            log_pc = self.class_log_prior_[c]
            log_jll.append(log_px_given_c + log_pc)
        return np.vstack(log_jll).T  # shape (n_samples, n_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_jll = self._joint_log_likelihood(X)
        # normalize to probabilities
        log_jll -= log_jll.max(axis=1, keepdims=True)
        proba = np.exp(log_jll)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(idx)


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------
def make_models(n_features: int) -> Dict[str, object]:
    """
    Construct the five classical models:

    - GMM
    - Decision Tree (entropy criterion)
    - Softmax Discriminant Classifier (multinomial logistic regression)
    - SVM-RBF
    - Naive Bayes Classifier (MultinomialNB on BoF histograms)
    """

    models: Dict[str, object] = {}

    # 1) Gaussian Mixture Model
    models["GMM"] = GMMClassifier(
        n_components_per_class=1,
        covariance_type="full",
        random_state=RANDOM_STATE,
    )

    # 2) Decision Tree (using information gain / entropy) :contentReference[oaicite:4]{index=4}
    models["DecisionTree"] = DecisionTreeClassifier(
        criterion="entropy",
        class_weight="balanced",  # dataset is imbalanced
        random_state=RANDOM_STATE,
    )

    # 3) Softmax Discriminant Classifier (multinomial logistic regression) :contentReference[oaicite:5]{index=5}
    models["SDC_LogReg"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,          # L2 penalty (recommended way in 1.8)
                    solver="lbfgs",        # supports multinomial automatically
                    max_iter=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 4) SVM with RBF kernel :contentReference[oaicite:6]{index=6}
    models["SVM_RBF"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",  # 1 / (n_features * X.var()) by default
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 5) Naive Bayes Classifier (on histogram features) :contentReference[oaicite:7]{index=7}
    models["NaiveBayes"] = MultinomialNB()

    return models


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """
    Fit each model on (X_train, y_train) and evaluate on the validation set.
    """
    # For MSE, we encode labels as 0,1,... but we still fit models on original labels.
    le = LabelEncoder().fit(y_train)
    y_val_int = le.transform(y_val)

    models = make_models(n_features=X_train.shape[1])

    for name, model in models.items():
        print("=" * 80)
        print(f"Model: {name}")
        print("-" * 80)

        # For MultinomialNB we must ensure inputs are non-negative
        if isinstance(model, MultinomialNB):
            X_train_in = np.clip(X_train, a_min=0.0, a_max=None)
            X_val_in = np.clip(X_val, a_min=0.0, a_max=None)
        else:
            X_train_in, X_val_in = X_train, X_val

        model.fit(X_train_in, y_train)
        y_pred = model.predict(X_val_in)

        # Map predictions to ints for MSE
        y_pred_int = le.transform(y_pred)

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        mse = mean_squared_error(y_val_int, y_pred_int)
        cm = confusion_matrix(y_val, y_pred, labels=le.classes_)

        print(f"Accuracy           : {acc:.4f}")
        print(f"Balanced accuracy  : {bal_acc:.4f}")
        print(f"MSE (on label ints): {mse:.6f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(le.classes_)
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_val, y_pred, digits=4))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    train_path = BASE_DIR / "bof_features_train.npz"
    val_path = BASE_DIR / "bof_features_validation.npz"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected NPZ files under {BASE_DIR}:\n"
            f"  - {train_path.name}\n"
            f"  - {val_path.name}\n"
        )

    X_train, y_train = load_bof_npz(train_path)
    X_val, y_val = load_bof_npz(val_path)

    print(f"Train shape: {X_train.shape},  Val shape: {X_val.shape}")
    print(f"Train class distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Val   class distribution: {np.unique(y_val, return_counts=True)}")

    evaluate_models(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
