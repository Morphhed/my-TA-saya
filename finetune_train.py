import os
import glob
import re
import sys
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm

from finetune_model import get_simclr_model_cbam
from config import EXTERNAL_DRIVE, RAW_DATA_DIR, CHECKPOINT_DIR, FINETUNE_BATCH_SIZE, LEARNING_RATE, WEIGHT, WORKERS, FINETUNE_VAL_RATIO, SEED

VALID_STRATEGIES = {
    "frozen": "FT-A Frozen Backbone",
    "layer4": "FT-B Layer4 + CBAM + FC",
    "full": "FT-C Full Fine-Tuning"
}

STRATEGY_NAMES = {
    "frozen": "ResNet50_USULAN_SimCLR_CBAM_FT_A_Frozen",
    "layer4": "ResNet50_USULAN_SimCLR_CBAM_FT_B_Layer4",
    "full": "ResNet50_USULAN_SimCLR_CBAM_FT_C_Full"
}

def get_finetune_strategy():
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("FINE-TUNING STRATEGY BELUM DITENTUKAN")
        print("=" * 70)
        print("Gunakan salah satu perintah berikut:\n")
        print("  python finetune_train.py frozen")
        print("  python finetune_train.py layer4")
        print("  python finetune_train.py full")
        print("\nStrategi:")
        print("  frozen : FT-A - Frozen Backbone")
        print("  layer4 : FT-B - Layer4 + CBAM + FC")
        print("  full   : FT-C - Full Fine-Tuning")
        print("=" * 70)
        sys.exit(1)

    strategy = sys.argv[1].lower()

    if strategy not in VALID_STRATEGIES:
        print("\n" + "=" * 70)
        print(f"[ERROR] Strategi tidak dikenal: {strategy}")
        print("Gunakan salah satu dari: frozen, layer4, full")
        print("=" * 70)
        sys.exit(1)

    return strategy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_best_simclr_checkpoint(folder):
    files = glob.glob(os.path.join(folder, "simclr_best_epoch_*.pth"))
    if not files:
        raise FileNotFoundError(f"Checkpoint SimCLR tidak ditemukan di {folder}")

    def epoch_number(path):
        match = re.search(r"simclr_best_epoch_(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(files, key=epoch_number)

def calculate_sensitivity_specificity(y_true, y_pred, normal_idx, abnormal_idx):
    cm = confusion_matrix(y_true, y_pred, labels=[normal_idx, abnormal_idx])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity, cm

def freeze_backbone_batchnorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and not name.startswith("cbam"):
            module.eval()

def predict(model, loader, device, abnormal_idx):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, abnormal_idx].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def evaluate_validation(model, loader, device, abnormal_idx):
    labels, preds, probs = predict(model, loader, device, abnormal_idx)
    acc = accuracy_score(labels, preds)
    binary_labels = (labels == abnormal_idx).astype(int)

    try:
        auc = roc_auc_score(binary_labels, probs)
    except ValueError:
        auc = float("nan")

    return acc, auc

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, freeze_bn=False, description="Train"):
    model.train()

    if freeze_bn:
        freeze_backbone_batchnorm(model)

    running_loss = 0.0
    pbar = tqdm(loader, desc=description)

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)

def configure_phase2(model, finetune_strategy):
    if finetune_strategy == "frozen":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.cbam.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW([
            {"params": model.cbam.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 3e-4}
        ], weight_decay=WEIGHT)

        return optimizer, True, "FT-A Frozen Backbone"

    if finetune_strategy == "layer4":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.cbam.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW([
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.cbam.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 3e-4}
        ], weight_decay=WEIGHT)

        return optimizer, True, "FT-B Layer4 + CBAM + FC"

    if finetune_strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW([
            {"params": itertools.chain(model.conv1.parameters(), model.bn1.parameters(), model.layer1.parameters(), model.layer2.parameters()), "lr": 1e-6},
            {"params": model.layer3.parameters(), "lr": 5e-6},
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.cbam.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 3e-4}
        ], weight_decay=WEIGHT)

        return optimizer, False, "FT-C Full Fine-Tuning"

    raise ValueError(f"Strategi tidak dikenal: {finetune_strategy}")

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, class_to_idx, device, epochs=50, warmup_epochs=5, finetune_strategy="full"):
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print(f"STRATEGY: {VALID_STRATEGIES[finetune_strategy]}")
    print("=" * 60)

    normal_idx = class_to_idx["Normal"]
    abnormal_idx = class_to_idx["Abnormal"]

    print(f"Class mapping  : {class_to_idx}")
    print(f"Normal index   : {normal_idx}")
    print(f"Abnormal index : {abnormal_idx}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hasil_dir = os.path.join(EXTERNAL_DRIVE, "hasil_finetune_simclr")
    os.makedirs(hasil_dir, exist_ok=True)

    model_save_path = os.path.join(hasil_dir, f"{model_name}_finetuned_best.pth")
    best_val_score = -float("inf")
    best_epoch = 0

    for name, param in model.named_parameters():
        param.requires_grad = ("cbam" in name or "fc" in name)

    warmup_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(warmup_params, lr=LEARNING_RATE, weight_decay=WEIGHT)

    print("\n>>> Fase 1: Warm-up CBAM + FC")

    for epoch in range(warmup_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, freeze_bn=True, description=f"Warm-up [{epoch+1}/{warmup_epochs}]")
        val_acc, val_auc = evaluate_validation(model, val_loader, device, abnormal_idx)
        score = val_auc if not np.isnan(val_auc) else val_acc

        print(f"Warm-up Epoch [{epoch+1}/{warmup_epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"[Checkpoint] Best validation score: {best_val_score:.4f} | Epoch: {best_epoch}")

    optimizer, freeze_bn_phase2, phase2_name = configure_phase2(model, finetune_strategy)

    print(f"\n>>> Fase 2: {phase2_name}")

    for epoch in range(warmup_epochs, epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, freeze_bn=freeze_bn_phase2, description=f"{phase2_name} [{epoch+1}/{epochs}]")
        val_acc, val_auc = evaluate_validation(model, val_loader, device, abnormal_idx)
        score = val_auc if not np.isnan(val_auc) else val_acc

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"[Checkpoint] Best validation score: {best_val_score:.4f} | Epoch: {best_epoch}")

    print("\n" + "=" * 60)
    print("HASIL FINE-TUNING TERBAIK")
    print("=" * 60)
    print(f"Strategy         : {VALID_STRATEGIES[finetune_strategy]}")
    print(f"Best Epoch       : {best_epoch}")
    print(f"Best Val Score   : {best_val_score:.4f}")
    print(f"Best Checkpoint  : {os.path.basename(model_save_path)}")
    print("=" * 60)

    model.load_state_dict(torch.load(model_save_path, map_location=device))

    all_labels, all_preds, all_probs = predict(model, test_loader, device, abnormal_idx)

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    binary_labels = (all_labels == abnormal_idx).astype(int)

    try:
        auc = roc_auc_score(binary_labels, all_probs)
    except ValueError:
        auc = float("nan")

    sens, spec, cm = calculate_sensitivity_specificity(all_labels, all_preds, normal_idx, abnormal_idx)

    report = classification_report(
        all_labels,
        all_preds,
        labels=[normal_idx, abnormal_idx],
        target_names=["Normal", "Abnormal"],
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"HASIL EVALUASI FINAL TEST: {model_name}")
    print("=" * 60)
    print(f"Accuracy                   : {acc:.4f}")
    print(f"Precision (Weighted)       : {prec:.4f}")
    print(f"Recall (Weighted)          : {rec:.4f}")
    print(f"F1-Score (Weighted)        : {f1:.4f}")
    print(f"Sensitivity (Abnormal)     : {sens:.4f}")
    print(f"Specificity (Normal)       : {spec:.4f}")
    print(f"ROC-AUC                    : {auc:.4f}")
    print("=" * 60)
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(hasil_dir, f"Report_{model_name}.txt")

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"HASIL EVALUASI FINAL TEST: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Fine-Tuning Strategy       : {VALID_STRATEGIES[finetune_strategy]}\n")
        f.write(f"Best Epoch                 : {best_epoch}\n")
        f.write(f"Best Validation Score      : {best_val_score:.4f}\n")
        f.write(f"Accuracy                   : {acc:.4f}\n")
        f.write(f"Precision (Weighted)       : {prec:.4f}\n")
        f.write(f"Recall (Weighted)          : {rec:.4f}\n")
        f.write(f"F1-Score (Weighted)        : {f1:.4f}\n")
        f.write(f"Sensitivity (Abnormal)     : {sens:.4f}\n")
        f.write(f"Specificity (Normal)       : {spec:.4f}\n")
        f.write(f"ROC-AUC                    : {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(hasil_dir, f"CM_{model_name}.png"), dpi=300)
    plt.close()

    print(f"\n[Info] Report disimpan: {report_path}")
    print(f"[Info] Model disimpan : {model_save_path}")

if __name__ == "__main__":
    FINETUNE_STRATEGY = get_finetune_strategy()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("FINE-TUNING SIMCLR + CBAM")
    print("=" * 70)
    print(f"Device   : {device}")
    print(f"Strategy : {VALID_STRATEGIES[FINETUNE_STRATEGY]}")
    print("=" * 70)

    train_dir = RAW_DATA_DIR
    test_dir = os.path.join(EXTERNAL_DRIVE, "test")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    split_dataset = datasets.ImageFolder(train_dir)
    indices = np.arange(len(split_dataset))
    targets = np.array(split_dataset.targets)

    train_indices, val_indices = train_test_split(indices, test_size=FINETUNE_VAL_RATIO, random_state=SEED, shuffle=True, stratify=targets)

    train_dataset_full = datasets.ImageFolder(train_dir, train_transform)
    val_dataset_full = datasets.ImageFolder(train_dir, eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, eval_transform)

    if train_dataset_full.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError(f"Mapping class train dan test berbeda: train={train_dataset_full.class_to_idx}, test={test_dataset.class_to_idx}")

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Train images  : {len(train_dataset)}")
    print(f"Val images    : {len(val_dataset)}")
    print(f"Test images   : {len(test_dataset)}")
    print(f"Class mapping : {test_dataset.class_to_idx}")
    print("=" * 70)

    try:
        best_simclr_checkpoint = find_best_simclr_checkpoint(CHECKPOINT_DIR)

        print("\n" + "=" * 70)
        print("SIMCLR INITIALIZATION")
        print("=" * 70)
        print(f"Checkpoint : {os.path.basename(best_simclr_checkpoint)}")
        print(f"Strategy   : {VALID_STRATEGIES[FINETUNE_STRATEGY]}")
        print("=" * 70)

        simclr_model = get_simclr_model_cbam(best_simclr_checkpoint)

        train_and_evaluate(
            simclr_model,
            STRATEGY_NAMES[FINETUNE_STRATEGY],
            train_loader,
            val_loader,
            test_loader,
            test_dataset.class_to_idx,
            device,
            epochs=50,
            warmup_epochs=5,
            finetune_strategy=FINETUNE_STRATEGY
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}")
        sys.exit(1)