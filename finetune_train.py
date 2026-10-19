import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Mengimpor model dari file arsitektur baru Anda
from finetune_model import get_simclr_model_cbam, get_baseline_model_cbam
from config import EXTERNAL_DRIVE, RAW_DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT, WORKERS

def train_and_evaluate(model, model_name, train_loader, test_loader, device, epochs=30, warmup_epochs=2):
    print(f"\n{'='*50}")
    print(f"[{model_name}] Memulai Pelatihan Fine-Tuning")
    print(f"{'='*50}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ----------------------------------------------------
    # TAHAP 1: WARM-UP (Membekukan Backbone, Melatih CBAM & FC)
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
    # TAHAP 2: FULL FINE-TUNING (Unfreeze Semua Layer)
    # ----------------------------------------------------
    print(f"\n>>> [Fase 2] Full Fine-Tuning: Melepas kunci seluruh jaringan...")
    for param in model.parameters():
        param.requires_grad = True
        
    # Memperhalus Learning Rate agar bobot berharga SimCLR tidak rusak oleh gradien drastis
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

    # EVALUASI METRIK AKHIR
    print(f"\n[{model_name}] Mengevaluasi Model pada Data Uji...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(f"\n--- Laporan Klasifikasi Akhir: {model_name} ---")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"], digits=4))


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan Fine-tuning di perangkat: {DEVICE}")
    
    TRAIN_DIR = RAW_DATA_DIR 
    TEST_DIR = os.path.join(EXTERNAL_DRIVE, 'test')
    SIMCLR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "simclr_best_epoch_100.pth")
    
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
    
    train_loader = DataLoader(
        datasets.ImageFolder(TRAIN_DIR, train_transform), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        datasets.ImageFolder(TEST_DIR, test_transform), 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True
    )

    try:
        # Menjalankan pengujian untuk Model Usulan (SimCLR 100 Epoch + CBAM)
        simclr_finetune_model = get_simclr_model_cbam(SIMCLR_CHECKPOINT)
        train_and_evaluate(simclr_finetune_model, "ResNet50_USULAN_Dengan_SimCLR_dan_CBAM", train_loader, test_loader, DEVICE, epochs=30, warmup_epochs=2)
    except Exception as e:
        print(f"Gagal memuat model SimCLR: {e}")

    # Menjalankan pengujian untuk Model Pembanding (Baseline ImageNet + CBAM)
    baseline_model = get_baseline_model_cbam()
    train_and_evaluate(baseline_model, "ResNet50_BASELINE_Tanpa_SimCLR_Dengan_CBAM", train_loader, test_loader, DEVICE, epochs=30, warmup_epochs=2)
