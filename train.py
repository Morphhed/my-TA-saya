import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from config import PATCH_DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, TEMPERATURE, WARMUP, WORKERS, WEIGHT, ACC_STEP
from dataset import SimCLRDataset, get_simclr_transforms
from model import SimCLRModel, NTXentLoss

class EarlyStopping:
    def __init__(self, patience=12):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} dari {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True

def configure_optimizer(model, learning_rate, weight_decay):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer_grouped_parameters = [
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': weight_decay}
    ]
    
    return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan training di: {device}")
    
    # 2. Setup Data
    dataset = SimCLRDataset(image_dir=PATCH_DATA_DIR, transform=get_simclr_transforms())
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=WORKERS,
        persistent_workers=True, 
        drop_last=True,
        pin_memory=True
    )

    # 3. Setup Model, Optimizer, Scheduler, & AMP
    model = SimCLRModel()
    if torch.cuda.device_count() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = NTXentLoss(device=device, temperature=TEMPERATURE)
    
    optimizer = configure_optimizer(model, LEARNING_RATE, weight_decay=WEIGHT)
    
    steps_per_epoch = len(dataloader) // ACC_STEP # Karena ACC_STEP=1, sama dengan len(dataloader)
    warmup_steps = WARMUP * steps_per_epoch
    cosine_steps = (EPOCHS - WARMUP) * steps_per_epoch
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-5)
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    scaler = torch.cuda.amp.GradScaler()

    log_dir = os.path.join(CHECKPOINT_DIR, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)

    # 4. Inisialisasi Early Stopping
    early_stopping = EarlyStopping(patience=12)
    start_epoch = 0
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latestcheck.pth")
    if os.path.exists(checkpoint_path):
        print("Menemukan file latestcheck.pth! Memuat data...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load state early stopping agar tidak reset dari 0 saat dilanjutkan
        if 'best_loss' in checkpoint:
            early_stopping.best_loss = checkpoint['best_loss']
        if 'early_stop_counter' in checkpoint:
            early_stopping.counter = checkpoint['early_stop_counter']
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Melanjutkan dari Epoch {start_epoch+1} | Best Loss sebelumnya: {early_stopping.best_loss:.4f}")

    # 5. Training Loop
    global_step = 0
    ACCUMULATION_STEPS = ACC_STEP

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad() 
        
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                z_i = model(view1)
                z_j = model(view2)
                loss = criterion(z_i, z_j)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()            
            
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
		        scheduler.step() 
            
            total_loss += (loss.item() * ACCUMULATION_STEPS)
            global_step += 1
            
            if batch_idx % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {(loss.item() * ACCUMULATION_STEPS):.4f} | LR: {current_lr:.6f}")
                
                writer.add_scalar("Training/Batch_Loss", (loss.item() * ACCUMULATION_STEPS), global_step)
                writer.add_scalar("Training/Learning_Rate", current_lr, global_step)
                
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"=== Akhir Epoch {epoch+1} | Rata-rata Loss: {avg_loss:.4f} | LR: {current_lr:.6f} ===")
                
        # 6. PENGECEKAN EARLY STOPPING DAN SAVE BEST MODEL
        is_best = early_stopping(avg_loss)
        
        if is_best:
            best_model_name = f"simclr_best_epoch_{epoch+1}.pth"
            best_model_path = os.path.join(CHECKPOINT_DIR, best_model_name)
            
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw_model.backbone.state_dict(), best_model_path)
            print(f"🌟 Model Terbaik Baru (Loss: {early_stopping.best_loss:.4f})! Menyimpan backbone ke: {best_model_name}")

        if (epoch + 1) % 2 == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'best_loss': early_stopping.best_loss, 
                'early_stop_counter': early_stopping.counter # Simpan hitungan kesabaran
            }
            
            epoch_cp_name = f"checkpoint_epoch_{epoch+1}.pth"
            epoch_cp_path = os.path.join(CHECKPOINT_DIR, epoch_cp_name)
            torch.save(checkpoint_state, epoch_cp_path)
            
            torch.save(checkpoint_state, checkpoint_path)
            
            print(f"--> Checkpoint aman tersimpan ke: {epoch_cp_name} dan latestcheck.pth")

        # 7. HENTIKAN TRAINING JIKA PATIENCE HABIS
        if early_stopping.early_stop:
            print(f"🛑 Early stopping dipicu pada epoch {epoch+1}. Training dihentikan karena loss tidak membaik selama {early_stopping.patience} epoch berturut-turut.")
            break

    print("Training Pre-text SimCLR Selesai!")

if __name__ == "__main__":
    main()