import os
import copy
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
TRAIN_DIR = r"C:\Users\86188\Desktop\BIA\ica\data\train"
TEST_DIR = r"C:\Users\86188\Desktop\BIA\ica\data\test"
MODEL_NAME = "deit_small_patch16_224"
OUTPUT_DIR = r"C:\Users\86188\Desktop\BIA\ica\transformermodel\bestmodels\deit_small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {MODEL_NAME} for benign vs malignant classification.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory for saved artifacts.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    batch_size: int = 16, num_workers: int = 4, val_ratio: float = 0.2, seed: int = 42
):
    """
    Returns:
        train_loader, val_loader, test_loader, class_names, class_counts
    """
    train_transform, val_test_transform = create_transforms()
    base_train = datasets.ImageFolder(TRAIN_DIR)
    class_names = base_train.classes
    counts = np.bincount(base_train.targets, minlength=len(class_names))
    class_counts = {class_names[i]: int(counts[i]) for i in range(len(class_names))}

    total_samples = len(base_train)
    if total_samples < 2:
        raise ValueError("Need at least two training images to build a train/val split.")
    val_size = int(total_samples * val_ratio)
    if val_ratio > 0:
        val_size = max(1, min(total_samples - 1, val_size))
    else:
        val_size = 0

    indices = np.arange(total_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_test_transform), val_indices)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader, class_names, class_counts


def compute_class_weights(class_names: list[str], class_counts: dict[str, int]) -> torch.Tensor:
    counts = np.array([class_counts[name] for name in class_names], dtype=np.float32)
    total = counts.sum()
    weights = []
    for count in counts:
        weights.append(0.0 if count == 0 else total / (len(counts) * count))
    return torch.tensor(weights, dtype=torch.float32)


def create_transformer_model(model_name: str, num_classes: int = 2) -> nn.Module:
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    malignant_idx: int = 1,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        avg_loss, accuracy, y_true, y_pred, y_prob (malignant probability)
    """
    model.eval()
    running_loss = 0.0
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            logits_list.append(outputs.cpu())
            labels_list.append(labels.cpu())
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    probs = torch.softmax(logits, dim=1)
    y_true = labels.numpy()
    y_pred = probs.argmax(dim=1).numpy()
    y_prob = probs[:, malignant_idx].numpy()
    accuracy = float((y_pred == y_true).mean())
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, accuracy, y_true, y_pred, y_prob


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, class_names, class_counts = create_dataloaders(
        args.batch_size, args.num_workers
    )
    print("Training samples per class:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    if len(class_names) < 2:
        raise ValueError("Expected exactly two classes (benign, malignant).")
    malignant_idx = 1
    class_weights = compute_class_weights(class_names, class_counts).to(device)
    model = create_transformer_model(MODEL_NAME, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision_mal": [],
        "val_recall_mal": [],
        "val_f1_mal": [],
    }
    best_f1 = -np.inf
    best_state = None
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, y_true_val, y_pred_val, _ = evaluate(model, val_loader, criterion, device, malignant_idx)
        val_precision = precision_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
        val_f1 = f1_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
        scheduler.step(val_loss)
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision_mal"].append(val_precision)
        history["val_recall_mal"].append(val_recall)
        history["val_f1_mal"].append(val_f1)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_precision_mal={val_precision:.4f} "
            f"val_recall_mal={val_recall:.4f} val_f1_mal={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best validation F1 (malignant): {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))
    _, test_acc, y_true_test, y_pred_test, y_prob_test = evaluate(model, test_loader, criterion, device, malignant_idx)
    cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
    precisions = precision_score(y_true_test, y_pred_test, labels=[0, 1], average=None, zero_division=0)
    recalls = recall_score(y_true_test, y_pred_test, labels=[0, 1], average=None, zero_division=0)
    f1s = f1_score(y_true_test, y_pred_test, labels=[0, 1], average=None, zero_division=0)
    y_true_binary = (y_true_test == 1).astype(int)
    try:
        roc_auc = roc_auc_score(y_true_binary, y_prob_test)
    except ValueError:
        roc_auc = float("nan")
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true_binary, y_prob_test)
    try:
        ap_score = average_precision_score(y_true_binary, y_prob_test)
    except ValueError:
        ap_score = float("nan")
    report = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4, zero_division=0)
    print("Test classification report:\n", report)
    test_metrics = {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "accuracy": float(test_acc),
        "precision": {class_names[i]: float(precisions[i]) for i in range(len(class_names))},
        "recall": {class_names[i]: float(recalls[i]) for i in range(len(class_names))},
        "f1": {class_names[i]: float(f1s[i]) for i in range(len(class_names))},
        "roc_auc_mal": float(roc_auc),
        "pr_curve_mal": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": float(ap_score),
        },
        "classification_report": report,
    }
    with open(os.path.join(args.output_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    with open(os.path.join(args.output_dir, "test_metrics.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
