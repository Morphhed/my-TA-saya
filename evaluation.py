import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from PIL import ImageFile

from eval_path import EXPERIMENT_FOLDER, TEST_DATA, TRAIN_DATA, HASIL_DIR
from config import SEED, WORKERS

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_class_index(class_to_idx, target):
    for name, idx in class_to_idx.items():
        if name.lower() == target.lower():
            return idx
    raise RuntimeError(f"Kelas {target} tidak ditemukan: {class_to_idx}")

def find_best_simclr_checkpoint(folder):
    files = glob.glob(os.path.join(folder, "simclr_best_epoch_*.pth"))
    if not files:
        raise FileNotFoundError(f"Checkpoint SimCLR tidak ditemukan di {folder}")
    def epoch_number(path):
        match = re.search(r"simclr_best_epoch_(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1
    return max(files, key=epoch_number)

def load_simclr_backbone(checkpoint_path, device):
    backbone = resnet50(weights=None)
    backbone.fc = nn.Identity()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = backbone.state_dict()
    compatible_state = {}

    for key, value in state_dict.items():
        key = key.replace("module.", "")
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        if key.startswith("projection_head."):
            continue
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value

    result = backbone.load_state_dict(compatible_state, strict=False)

    if result.missing_keys:
        raise RuntimeError(f"Bobot backbone tidak lengkap. Missing keys: {result.missing_keys}")

    print(f"[Checkpoint] {os.path.basename(checkpoint_path)}")
    print(f"[Checkpoint] Tensor backbone dimuat: {len(compatible_state)}")

    backbone = backbone.to(device)
    backbone.eval()

    for param in backbone.parameters():
        param.requires_grad = False

    return backbone

def extract_features(model, dataloader, device):
    features_list, labels_list = [], []

    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs)
            features = F.normalize(features, p=2, dim=1)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.vstack(features_list), np.concatenate(labels_list)

def calculate_sensitivity_specificity(y_true, y_pred, normal_idx, abnormal_idx):
    cm = confusion_matrix(y_true, y_pred, labels=[normal_idx, abnormal_idx])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity, cm

def evaluate_knn(model, train_loader, test_loader, device, class_to_idx, model_name):
    normal_idx = get_class_index(class_to_idx, "Normal")
    abnormal_idx = get_class_index(class_to_idx, "Abnormal")

    print(f"\n[{model_name}] Ekstraksi fitur development set...")
    X_train, y_train = extract_features(model, train_loader, device)

    print(f"[{model_name}] Ekstraksi fitur independent test set...")
    X_test, y_test = extract_features(model, test_loader, device)

    print(f"Feature train: {X_train.shape}")
    print(f"Feature test : {X_test.shape}")
    print(f"Class mapping: {class_to_idx}")

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "cosine"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    print(f"\n[{model_name}] GridSearchCV hanya pada development set...")
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_

    print(f"Best CV Score : {grid_search.best_score_:.4f}")
    print(f"Best Parameter: {grid_search.best_params_}")

    all_preds = best_knn.predict(X_test)
    all_probs = best_knn.predict_proba(X_test)

    abnormal_column = int(np.where(best_knn.classes_ == abnormal_idx)[0][0])
    abnormal_probs = all_probs[:, abnormal_column]
    binary_test = (y_test == abnormal_idx).astype(int)

    acc = accuracy_score(y_test, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, all_preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(binary_test, abnormal_probs)
    except ValueError:
        auc = float("nan")

    sens, spec, cm = calculate_sensitivity_specificity(y_test, all_preds, normal_idx, abnormal_idx)

    report = classification_report(
        y_test,
        all_preds,
        labels=[normal_idx, abnormal_idx],
        target_names=["Normal", "Abnormal"],
        digits=4,
        zero_division=0
    )

    report_path = os.path.join(HASIL_DIR, f"Report_KNN_{model_name}.txt")

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"HASIL EVALUASI REPRESENTASI KNN: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best CV Balanced Accuracy  : {grid_search.best_score_:.4f}\n")
        f.write(f"Best Parameters            : {grid_search.best_params_}\n")
        f.write(f"Accuracy                   : {acc:.4f}\n")
        f.write(f"Precision (Weighted)       : {prec:.4f}\n")
        f.write(f"Recall (Weighted)          : {rec:.4f}\n")
        f.write(f"F1-Score (Weighted)        : {f1:.4f}\n")
        f.write(f"Sensitivity (Abnormal)     : {sens:.4f}\n")
        f.write(f"Specificity (Normal)       : {spec:.4f}\n")
        f.write(f"ROC-AUC                    : {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("\n" + "=" * 60)
    print(f"HASIL EVALUASI REPRESENTASI KNN: {model_name}")
    print("=" * 60)
    print(f"Best CV Balanced Accuracy  : {grid_search.best_score_:.4f}")
    print(f"Best Parameters            : {grid_search.best_params_}")
    print(f"Accuracy                   : {acc:.4f}")
    print(f"Precision (Weighted)       : {prec:.4f}")
    print(f"Recall (Weighted)          : {rec:.4f}")
    print(f"F1-Score (Weighted)        : {f1:.4f}")
    print(f"Sensitivity (Abnormal)     : {sens:.4f}")
    print(f"Specificity (Normal)       : {spec:.4f}")
    print(f"ROC-AUC                    : {auc:.4f}")
    print("=" * 60)
    print(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    plt.title(f"KNN Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(HASIL_DIR, f"CM_KNN_{model_name}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"[Info] Report: {report_path}")
    print(f"[Info] CM    : {cm_path}")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DATA, transform=transform_pipeline)
    test_dataset = datasets.ImageFolder(TEST_DATA, transform=transform_pipeline)

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError(f"Mapping kelas train dan test berbeda: train={train_dataset.class_to_idx}, test={test_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=WORKERS, pin_memory=True)

    checkpoint_path = find_best_simclr_checkpoint(EXPERIMENT_FOLDER)
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    backbone = load_simclr_backbone(checkpoint_path, device)

    evaluate_knn(
        backbone,
        train_loader,
        test_loader,
        device,
        train_dataset.class_to_idx,
        model_name
    )