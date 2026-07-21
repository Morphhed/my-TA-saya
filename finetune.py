import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50
from sklearn.metrics import classification_report, accuracy_score

# Mengambil path dari config.py Anda yang sudah ada
from config import RAW_DATA_DIR, CHECKPOINT_DIR

# --- HYPERPARAMETER KHUSUS FINE-TUNING ---
FT_BATCH_SIZE = 16
FT_EPOCHS = 15
FT_LR = 1e-4
NUM_CLASSES = 3

# Skenario A/B Testing: 
# True  = Model Usulan (Menggunakan bobot SimCLR dari train.py)
# False = Model Baseline (Supervised murni / ImageNet)
USE_SSL_WEIGHTS = True 

def get_finetune_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_model(device):
    model = resnet50(pretrained=not USE_SSL_WEIGHTS)
    
    if USE_SSL_WEIGHTS:
        # Mengambil nama file yang dihasilkan oleh train.py Anda
        ssl_weight_path = os.path.join(CHECKPOINT_DIR, "simclr_resnet50_final_backbone.pth")
        if os.path.exists(ssl_weight_path):
            model.load_state_dict(torch.load(ssl_weight_path, map_location=device))
            print("Model Usulan: Menggunakan bobot pre-trained SimCLR.")
        else:
            print("PERINGATAN: Bobot SimCLR tidak ditemukan di harddisk! Melatih dari awal.")
    else:
        print("Model Baseline: Menggunakan bobot standar (tanpa SimCLR).")

    # Ganti classification head untuk 3 kelas
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan Fine-tuning di: {device}")
    
    # Memuat dataset utuh dari RAW_DATA_DIR (yang berisi folder Type_1, Type_2, Type_3)
    full_dataset = datasets.ImageFolder(root=RAW_DATA_DIR)

    # 2 instance terpisah untuk train (dengan augmentasi) dan val (tanpa augmentasi)
    train_full = datasets.ImageFolder(root=RAW_DATA_DIR, transform=get_finetune_transforms(is_training=True))
    val_full = datasets.ImageFolder(root=RAW_DATA_DIR, transform=get_finetune_transforms(is_training=False))

    # Pembagian 80% Latih, 20% Validasi
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Kunci seed generator agar indeks pembagian 80/20 selalu konsisten dan dapat direplikasi
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, _ = random_split(train_full, [train_size, val_size], generator=generator)
    _, val_dataset = random_split(val_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=FT_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=FT_BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FT_LR)

    print("\nMemulai Tahap 4: Fine-Tuning...")
    for epoch in range(FT_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{FT_EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")

    print("\nMemulai Tahap 5: Pengujian & Evaluasi Kinerja...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("=== METRIK EVALUASI KESELURUHAN ===")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

    # Simpan model final hasil klasifikasi
    final_classifier_path = os.path.join(CHECKPOINT_DIR, "final_classifier_model.pth")
    torch.save(model.state_dict(), final_classifier_path)
    print(f"Model klasifikasi final tersimpan di: {final_classifier_path}")

if __name__ == "__main__":
    main()