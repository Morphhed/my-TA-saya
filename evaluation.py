import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report
)

# Import target folder dari eval_path
from eval_path import TARGET_FOLDER_PATH

# =====================================================================
# 1. BUNGKUSAN MODEL DOWNSTREAM
# =====================================================================
class CervixClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super(CervixClassifier, self).__init__()
        self.backbone = backbone
        
        # ResNet50 sebelum layer FC mengeluarkan dimensi 2048
        num_features = 2048 
        
        # Layer klasifikasi akhir untuk prediksi tingkat keparahan
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# =====================================================================
# 2. PERHITUNGAN MANUAL SENSITIVITAS & SPESIFISITAS
# =====================================================================
def calculate_sensitivity_specificity(cm, num_classes):
    sensitivities = []
    specificities = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        # Hindari pembagian dengan nol
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    return np.mean(sensitivities), np.mean(specificities)

# =====================================================================
# 3. FUNGSI UTAMA EVALUASI & HITUNG LOSS
# =====================================================================
def evaluate_model(model, dataloader, device, num_classes, model_name):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    print(f"\nMemulai proses inferensi & perhitungan loss untuk: {model_name}...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Hitung Loss per batch menggunakan CrossEntropy
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Hitung Rata-rata Loss keseluruhan batch
    avg_loss = total_loss / len(dataloader)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # --- HITUNG METRIK KLASIFIKASI ---
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc = float('nan') 
        
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    cm = confusion_matrix(all_labels, all_preds)
    sens, spec = calculate_sensitivity_specificity(cm, num_classes)

    # --- CETAK LAPORAN ---
    print("\n" + "=" * 55)
    print(f" HASIL EVALUASI & LOSS: {model_name} ")
    print("=" * 55)
    print(f"Average Loss (CrossEntropy): {avg_loss:.4f}")
    print(f"Accuracy                   : {acc:.4f}")
    print(f"Precision (Weighted)       : {prec:.4f}")
    print(f"Recall (Weighted)          : {rec:.4f}")
    print(f"F1-Score (Weighted)        : {f1:.4f}")
    print(f"Sensitivity (Macro)        : {sens:.4f}")
    print(f"Specificity (Macro)        : {spec:.4f}")
    print(f"ROC-AUC (OvR, Weighted)    : {auc:.4f}")
    print(f"Quadratic Weighted Kappa   : {qwk:.4f}")
    print("=" * 55)
    
    print("\nClassification Report Rinci per Kelas:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # --- PLOT & SIMPAN CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}', pad=20, fontsize=14)
    plt.ylabel('Label Sebenarnya (True)', fontsize=12)
    plt.xlabel('Label Prediksi (Predicted)', fontsize=12)
    
    output_img_name = f"CM_{model_name.replace('.pth', '')}.png"
    plt.tight_layout()
    plt.savefig(output_img_name, dpi=300)
    print(f"Visualisasi confusion matrix disimpan sebagai '{output_img_name}'")
    plt.close()

    return {
        'model_name': model_name,
        'loss': avg_loss,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'sensitivity': sens, 'specificity': spec, 'roc_auc': auc, 'qwk': qwk
    }

# =====================================================================
# 4. IMPLEMENTASI PEMANGGILAN MULTI-MODEL
# =====================================================================
if __name__ == "__main__":
    from torchvision.models import resnet50
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3  # Sesuaikan target kelas klasifikasi Anda
    
    print(f"[Info] Menyiapkan modul evaluasi di device: {DEVICE}")
    print(f"[Info] Memindai folder: {TARGET_FOLDER_PATH}")
    
    if not os.path.exists(TARGET_FOLDER_PATH):
        print(f"[Error] Folder tidak ditemukan: {TARGET_FOLDER_PATH}")
    else:
        pth_files = glob.glob(os.path.join(TARGET_FOLDER_PATH, "*.pth"))
        
        if not pth_files:
            print(f"[Peringatan] Tidak ada file .pth ditemukan di dalam {TARGET_FOLDER_PATH}")
        else:
            pth_files.sort(key=os.path.getmtime)
            print(f"[Info] Ditemukan {len(pth_files)} file model. Memulai evaluasi berurutan...\n")
            
            for file_path in pth_files:
                model_filename = os.path.basename(file_path)
                print(f">>> Memuat model: {model_filename} <<<")
                
                model = CervixClassifier(backbone=resnet50(pretrained=False), num_classes=NUM_CLASSES)
                
                checkpoint = torch.load(file_path, map_location=DEVICE)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                model = model.to(DEVICE)
                
                # --- CATATAN UNTUK ANDA ---
                # Uncomment baris di bawah ini dan masukkan dataloader test Anda saat siap mengeksekusi
                # metrics = evaluate_model(model, dataloader_test, DEVICE, NUM_CLASSES, model_name=model_filename)
            
            print("\n[Info] Seluruh evaluasi model di dalam folder telah selesai!")