import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import target folder dari eval_path 
from eval_path import TARGET_FOLDER_PATH, TEST_DATA, TRAIN_DATA

# =====================================================================
# 1. BUNGKUSAN MODEL UNTUK EKSTRAKSI FITUR SIMCLR
# =====================================================================
class SimCLRFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(SimCLRFeatureExtractor, self).__init__()
        self.backbone = backbone
        # Membuang layer FC bawaan agar outputnya murni vektor fitur (2048 dimensi)
        self.backbone.fc = nn.Identity()
        
    def forward(self, x):
        features = self.backbone(x)
        return features

# =====================================================================
# 2. PERHITUNGAN MANUAL SENSITIVITAS & SPESIFISITAS
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

# =====================================================================
# 3. FUNGSI EKSTRAKSI FITUR
# =====================================================================
def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            features = F.normalize(features, p=2, dim=1)    
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)

# =====================================================================
# 4. FUNGSI UTAMA EVALUASI KNN
# =====================================================================
def evaluate_knn(model, train_loader, test_loader, device, num_classes, model_name, class_names):
    print(f"\n[{model_name}] Mengekstrak fitur data latih (sebagai referensi KNN)...")
    X_train, y_train = extract_features(model, train_loader, device)
    
    print(f"[{model_name}] Mengekstrak fitur data uji...")
    X_test, y_test = extract_features(model, test_loader, device)
    
    print(f"[{model_name}] Menjalankan pencarian parameter terbaik (GridSearchCV) untuk KNN...")    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'cosine']
    }
    grid_search = GridSearchCV(
        KNeighborsClassifier(), 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )    
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    print(f"[{model_name}] Parameter KNN terbaik ditemukan: {grid_search.best_params_}")
    
    all_preds = best_knn.predict(X_test)
    all_probs = best_knn.predict_proba(X_test)

    # --- HITUNG METRIK KLASIFIKASI ---
    acc = accuracy_score(y_test, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc = float('nan') 
        
    qwk = cohen_kappa_score(y_test, all_preds, weights='quadratic')
    cm = confusion_matrix(y_test, all_preds)
    sens, spec = calculate_sensitivity_specificity(cm, num_classes)

    # --- CETAK LAPORAN ---
    print("\n" + "=" * 55)
    print(f" HASIL EVALUASI KNN: {model_name} ")
    print("=" * 55)
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
    print(classification_report(y_test, all_preds, target_names=class_names, zero_division=0))

    # --- PLOT & SIMPAN CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title(f'KNN Confusion Matrix - {model_name}', pad=20, fontsize=14)
    plt.ylabel('Label Sebenarnya (True)', fontsize=12)
    plt.xlabel('Label Prediksi (Predicted)', fontsize=12)
    
    output_img_name = f"CM_KNN_{model_name.replace('.pth', '')}.png"
    plt.tight_layout()
    plt.savefig(output_img_name, dpi=300)
    print(f"Visualisasi disimpan sebagai '{output_img_name}'")
    plt.close()

# =====================================================================
# 5. PEMANGGILAN MULTI-MODEL
# =====================================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3  
    
    # >>> PENTING: ISI FOLDER DATA LATIH (TRAIN) ANDA DI SINI <<<
    # KNN butuh data latih (gambar berlabel dari proses pre-training) sebagai patokan
    TRAIN_DATA_DIR = TRAIN_DATA
    
    TEST_DATA_DIR = TEST_DATA
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("[Info] Memuat Dataset Latih dan Uji...")
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transform_pipeline)
    test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=transform_pipeline)
    
    # Batch size diperbesar agar ekstraksi fitur lebih cepat
    dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
    dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    if not os.path.exists(TARGET_FOLDER_PATH):
        print(f"[Error] Folder tidak ditemukan: {TARGET_FOLDER_PATH}")
    else:
        pth_files = glob.glob(os.path.join(TARGET_FOLDER_PATH, "*.pth"))
        pth_files.sort(key=os.path.getmtime)
        
        for file_path in pth_files:
            model_filename = os.path.basename(file_path)
            
            model = SimCLRFeatureExtractor(backbone=resnet50(weights=None))
            checkpoint = torch.load(file_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
            model = model.to(DEVICE)
            
            class_names = train_dataset.classes
            evaluate_knn(model, dataloader_train, dataloader_test, DEVICE, NUM_CLASSES, model_name=model_filename, class_names=class_names)

