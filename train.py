import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import PATCH_DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE
from dataset import SimCLRDataset, get_simclr_transforms
from model import SimCLRModel, NTXentLoss

def main():
    # 1. Setup Device (Otomatis CPU karena CUDA tidak tersedia untuk Radeon 780M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan training di: {device}")
    
    if device.type == "cpu":
        print("PERINGATAN: Training berjalan di CPU. Proses akan memakan waktu lebih lama.")

    # 2. Setup Data
    print(f"Memuat dataset dari: {PATCH_DATA_DIR}")
    dataset = SimCLRDataset(image_dir=PATCH_DATA_DIR, transform=get_simclr_transforms())
    
    if len(dataset) == 0:
        print("Error: Tidak ada gambar patch ditemukan. Jalankan preprocess.py terlebih dahulu!")
        return

    # num_workers=0 sangat direkomendasikan untuk Windows lokal agar menghindari error Dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # 3. Setup Model & Optimizer
    model = SimCLRModel()
    if torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU (Dual RTX 2070)!")
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = NTXentLoss(device=device, temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    # 4. Resume Checkpoint jika ada
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Menemukan checkpoint di Harddisk! Memuat data...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Melanjutkan dari Epoch {start_epoch+1}...")

    # 5. Training Loop
    print("Memulai Pre-training SimCLR...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            z_i = model(view1)
            z_j = model(view2)
            
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"=== Akhir Epoch {epoch+1} | Rata-rata Loss: {avg_loss:.4f} ===")
        
        # Simpan ke Harddisk Eksternal setiap 2 epoch
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"--> Checkpoint aman tersimpan ke: {checkpoint_path}")

    # Simpan bobot final murni Backbone ResNet50
    final_model_path = os.path.join(CHECKPOINT_DIR, "simclr_resnet50_final_backbone.pth")
    torch.save(model.backbone.state_dict(), final_model_path)
    print("Training Selesai! Model siap digunakan untuk klasifikasi.")

if __name__ == "__main__":
    main()