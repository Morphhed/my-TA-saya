import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)

# Mengimpor model dari file arsitektur baru Anda
from finetune_model import get_simclr_model_cbam, get_baseline_model_cbam
from config import EXTERNAL_DRIVE, RAW_DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT, WORKERS

# =====================================================================
# FUNGSI PERHITUNGAN MANUAL SENSITIVITAS & SPESIFISITAS
# =====================================================================
def calculate_sensitivity_specificity(cm, num_classes):
    sensitivities, specificities = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    return np.mean(sensitivities), np.mean(specificities)

def train_and_evaluate(model, model_name, train_loader, test_loader, device, epochs=50, warmup_epochs=2):
    print(f"\n{'='*50}")
    print(f"[{model_name}] Memulai Pelatihan Fine-Tuning")
    print(f"{'='*50}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ----------------------------------------------------
    # TAHAP 1: WARM-UP
    # ----------------------------------------------------
    print(f"\n>>> [Fase 1] Warm-up: Membekukan backbone untuk adaptasi CBAM...")
    for name, param in model.named_parameters():
        if "cbam" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT)
    
    for epoch in range(warmup_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Warm-up Epoch [{epoch+1}/{warmup_epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # ----------------------------------------------------
    # TAHAP 2: FULL FINE-TUNING
    # ----------------------------------------------------
    print(f"\n>>> [Fase 2] Full Fine-Tuning: Melepas kunci seluruh jaringan...")
    for param in model.parameters():
        param.requires_grad = True
        
    fine_tune_lr = LEARNING_RATE * 0.1 
    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=WEIGHT)
    
    for epoch in range(warmup_epochs, epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Fine-Tuning Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # ----------------------------------------------------
    # EVALUASI METRIK AKHIR
    # ----------------------------------------------------
    print(f"\n[{model_name}] Mengevaluasi Model pada Data Uji...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan') 
        
    cm = confusion_matrix(all_labels, all_preds)
    sens, spec = calculate_sensitivity_specificity(cm, num_classes=2)
    class_names = ["Normal", "Abnormal"]

    HASIL_DIR = os.path.join(EXTERNAL_DRIVE, "hasil_finetune")
    os.makedirs(HASIL_DIR, exist_ok=True)
    
    # Cetak Terminal & Simpan TXT
    print("\n" + "=" * 55)
    print(f" HASIL EVALUASI: {model_name} ")
    print("=" * 55)
    print(f"Accuracy                   : {acc:.4f}")
    print(f"Precision (Weighted)       : {prec:.4f}")
    print(f"Recall (Weighted)          : {rec:.4f}")
    print(f"F1-Score (Weighted)        : {f1:.4f}")
    print(f"Sensitivity (Macro)        : {sens:.4f}")
    print(f"Specificity (Macro)        : {spec:.4f}")
    print(f"ROC-AUC                    : {auc:.4f}")
    print("=" * 55)
    
    report_output_name = f"Report_{model_name}.txt"
    report_path = os.path.join(HASIL_DIR, report_output_name)
    with open(report_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write(f" HASIL EVALUASI: {model_name} \n")
        f.write("=" * 55 + "\n")
        f.write(f"Accuracy                   : {acc:.4f}\n")
        f.write(f"Precision (Weighted)       : {prec:.4f}\n")
        f.write(f"Recall (Weighted)          : {rec:.4f}\n")
        f.write(f"F1-Score (Weighted)        : {f1:.4f}\n")
        f.write(f"Sensitivity (Macro)        : {sens:.4f}\n")
        f.write(f"Specificity (Macro)        : {spec:.4f}\n")
        f.write(f"ROC-AUC                    : {auc:.4f}\n")
        f.write("=" * 55 + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    
    # Plot CM
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', pad=20, fontsize=14)
    plt.ylabel('Label Sebenarnya (True)', fontsize=12)
    plt.xlabel('Label Prediksi (Predicted)', fontsize=12)
    
    output_img_name = f"CM_{model_name}.png"
    cm_path = os.path.join(HASIL_DIR, output_img_name)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Simpan Bobot Model Fine-Tuning
    model_output_name = f"{model_name}_finetuned_best.pth"
    model_save_path = os.path.join(HASIL_DIR, model_output_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"[Info] Bobot model Fine-Tuning berhasil disimpan ke: {model_save_path}\n")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan Fine-tuning di perangkat: {DEVICE}")
    
    TRAIN_DIR = RAW_DATA_DIR 
    TEST_DIR = os.path.join(EXTERNAL_DRIVE, 'test')
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(datasets.ImageFolder(TEST_DIR, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # =========================================================
    # DETEKSI OTOMATIS FILE PRE-TRAIN (SIMCLR) DI CHECKPOINT_DIR
    # =========================================================
    pth_files = glob.glob(os.path.join(CHECKPOINT_DIR, "simclr_best_epoch_*.pth"))
    
    if pth_files:
        # Mengambil file yang terbaru dibuat/dimodifikasi
        latest_simclr_checkpoint = max(pth_files, key=os.path.getmtime)
        print(f"[Info] Ditemukan bobot SimCLR: {os.path.basename(latest_simclr_checkpoint)}")
        
        try:
            simclr_finetune_model = get_simclr_model_cbam(latest_simclr_checkpoint)
            train_and_evaluate(simclr_finetune_model, "ResNet50_USULAN_Dengan_SimCLR_dan_CBAM", train_loader, test_loader, DEVICE, epochs=50, warmup_epochs=2)
        except Exception as e:
            print(f"[Error] Gagal melatih model usulan: {e}")
    else:
        print("\n[Info] Tidak ada file bobot SimCLR (.pth) ditemukan di folder checkpoint.")
        print("[Info] Skrip akan melompati pengujian model usulan dan langsung melatih Baseline.\n")

    # =========================================================
    # EKSEKUSI MODEL BASELINE (IMAGENET)
    # =========================================================
    print(f"[Info] Memuat model Baseline ImageNet...")
    baseline_model = get_baseline_model_cbam()
    train_and_evaluate(baseline_model, "ResNet50_BASELINE_Tanpa_SimCLR_Dengan_CBAM", train_loader, test_loader, DEVICE, epochs=50, warmup_epochs=2)
    