import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, accuracy_score

# Mengambil path dari config.py Anda
from config import RAW_DATA_DIR, CHECKPOINT_DIR

# --- HYPERPARAMETER KHUSUS FINE-TUNING ---
FT_BATCH_SIZE = 64
FT_EPOCHS = 100
FT_LR = 1e-4
NUM_CLASSES = 3  # Type_1, Type_2, Type_3 (Dataset Intel MobileODT)

# Skenario A/B Testing: 
# True  = Model Usulan (Menggunakan bobot SimCLR + Attention)
# False = Model Baseline (Supervised murni / ImageNet + Attention)
USE_SSL_WEIGHTS = True 


# --- 1. MODUL ATTENTION MECHANISM (CBAM: Channel & Spatial Attention) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class ResNet50WithAttention(nn.Module):
    """
    Arsitektur ResNet50 yang dilengkapi Attention Mechanism.
    Didesain agar atribut .layer4 tetap terekspos untuk kompatibilitas Grad-CAM di evaluation.py
    """
    def __init__(self, num_classes=3, pretrained=False):
        super(ResNet50WithAttention, self).__init__()
        if pretrained:
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            base = resnet50(weights=None)
            
        # Mengekstrak komponen ResNet50
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # Diakses oleh evaluation.py: model.layer4[-1]
        
        # Modul Attention (CBAM)
        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Aplikasi Attention Map
        x = x * self.ca(x)
        x = x * self.sa(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# --- 2. TRANSFORMATION PIPELINE ---
def get_finetune_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# --- 3. MODEL INITIALIZATION ---
def get_model(device):
    model = ResNet50WithAttention(num_classes=NUM_CLASSES, pretrained=not USE_SSL_WEIGHTS)
    
    if USE_SSL_WEIGHTS:
        ssl_weight_path = os.path.join(CHECKPOINT_DIR, "simclr_resnet50_final_backbone.pth")
        if os.path.exists(ssl_weight_path):
            state_dict = torch.load(ssl_weight_path, map_location=device)
            # Memuat bobot SSL backbone dengan strict=False untuk mengabaikan layer fc dan attention
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Model Usulan: Berhasil memuat bobot pre-trained SimCLR dari '{ssl_weight_path}'.")
        else:
            print("PERINGATAN: Bobot SimCLR tidak ditemukan di harddisk! Melatih dari awal.")
    else:
        print("Model Baseline: Menggunakan bobot standar (ImageNet).")

    return model.to(device)


# --- 4. MAIN TRAINING & EVALUATION LOOP ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan Fine-tuning di: {device}")
    
    # Memuat dataset utuh dari RAW_DATA_DIR (Type_1, Type_2, Type_3)
    full_dataset = datasets.ImageFolder(root=RAW_DATA_DIR)

    train_full = datasets.ImageFolder(root=RAW_DATA_DIR, transform=get_finetune_transforms(is_training=True))
    val_full = datasets.ImageFolder(root=RAW_DATA_DIR, transform=get_finetune_transforms(is_training=False))

    # Pembagian 80% Latih, 20% Validasi
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Kunci seed generator agar indeks pembagian 80/20 identik dengan evaluation.py
    generator = torch.Generator().manual_seed(67)
    
    train_dataset, _ = random_split(train_full, [train_size, val_size], generator=generator)
    _, val_dataset = random_split(val_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=FT_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=FT_BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_model(device)
    if torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU untuk fine-tuning.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FT_LR)

    print("\nMemulai Tahap Fine-Tuning...")
    best_val_acc = 0.0

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

        # Evaluasi singkat per epoch
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch [{epoch+1}/{FT_EPOCHS}] - Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")

        # Simpan model terbaik selama pelatihan
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_classifier_model.pth")
            torch.save(raw_model.state_dict(), best_model_path)

    print("\nMemulai Pengujian & Evaluasi Kinerja Akhir...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n=== METRIK EVALUASI KESELURUHAN ===")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

    # Simpan model final tanpa pembungkus DataParallel agar aman dibaca oleh evaluation.py
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    final_classifier_path = os.path.join(CHECKPOINT_DIR, "final_classifier_model.pth")
    torch.save(raw_model.state_dict(), final_classifier_path)
    print(f"Model klasifikasi final tersimpan di: {final_classifier_path}")

if __name__ == "__main__":
    main()