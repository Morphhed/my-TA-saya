import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import PATCH_DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, TEMPERATURE
from dataset import SimCLRDataset, get_simclr_transforms
from model import SimCLRModel, NTXentLoss

def main():
    # 1. Setup Device (CUDA untuk Dual GPU)
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

    # pin_memory=True mempercepat transfer data dari RAM ke VRAM GPU
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True,
        pin_memory=True
    )

    # 3. Setup Model, Optimizer, Scheduler, & AMP Scaler
    model = SimCLRModel()
    if torch.cuda.device_count() > 1:
        print(f"Menggunakan {torch.cuda.device_count()} GPU untuk training.")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = NTXentLoss(device=device, temperature=TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine Annealing Learning Rate Scheduler untuk 100 Epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Automatic Mixed Precision (AMP) Scaler untuk efisiensi VRAM
    scaler = torch.cuda.amp.GradScaler()

    # 4. Resume Checkpoint jika ada
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Menemukan checkpoint di Harddisk! Memuat data...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load bobot model & optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler & scaler jika ada
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Melanjutkan dari Epoch {start_epoch+1}...")

    # 5. Training Loop
    print("Memulai Pre-training SimCLR...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass dengan Mixed Precision (FP16/FP32)
            with torch.cuda.amp.autocast():
                z_i = model(view1)
                z_j = model(view2)
                loss = criterion(z_i, z_j)
            
            # Backward pass menggunakan GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
                
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"=== Akhir Epoch {epoch+1} | Rata-rata Loss: {avg_loss:.4f} | LR: {current_lr:.6f} ===")
        
        # Update Learning Rate di akhir epoch
        scheduler.step()
        
        # Simpan checkpoint ke Harddisk Eksternal setiap 2 epoch
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"--> Checkpoint aman tersimpan ke: {checkpoint_path}")

    # Simpan bobot final murni Backbone ResNet50
    # Ekstrak model asli jika terbungkus DataParallel
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    final_model_path = os.path.join(CHECKPOINT_DIR, "simclr_resnet50_final_backbone.pth")
    torch.save(raw_model.backbone.state_dict(), final_model_path)
    print("Training Selesai! Model siap digunakan untuk klasifikasi.")

if __name__ == "__main__":
    main()