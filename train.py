import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from config import PATCH_DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, TEMPERATURE, WARMUP, WORKERS, WEIGHT, ACC_STEP, SSL_VAL_RATIO, SEED
from dataset import SimCLRDataset, get_simclr_transforms, split_simclr_paths
from model import SimCLRModel, NTXentLoss

class EarlyStopping:
    def __init__(self, patience=12):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} dari {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def configure_optimizer(model, learning_rate, weight_decay):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return optim.AdamW([
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': weight_decay}
    ], lr=learning_rate)

def evaluate_simclr(model, loader, criterion, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for view1, view2 in loader:
            if view1.size(0) < 2:
                continue
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                z_i, z_j = model(view1), model(view2)
                loss = criterion(z_i, z_j)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches if total_batches > 0 else float("inf")

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan training di: {device}")

    train_paths, val_paths = split_simclr_paths(PATCH_DATA_DIR, SSL_VAL_RATIO, SEED)
    print(f"SSL Train patches: {len(train_paths)}")
    print(f"SSL Val patches  : {len(val_paths)}")

    transform = get_simclr_transforms()
    train_dataset = SimCLRDataset(image_paths=train_paths, transform=transform)
    val_dataset = SimCLRDataset(image_paths=val_paths, transform=transform)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, persistent_workers=WORKERS > 0, drop_last=True, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, persistent_workers=WORKERS > 0, drop_last=False, pin_memory=True)

    model = SimCLRModel()
    if torch.cuda.device_count() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = NTXentLoss(device=device, temperature=TEMPERATURE)
    optimizer = configure_optimizer(model, LEARNING_RATE, WEIGHT)

    steps_per_epoch = max(1, len(train_loader) // ACC_STEP)
    warmup_steps = max(1, WARMUP * steps_per_epoch)
    cosine_steps = max(1, (EPOCHS - WARMUP) * steps_per_epoch)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(os.path.join(CHECKPOINT_DIR, "tensorboard_logs"))
    early_stopping = EarlyStopping(patience=12)
    start_epoch = 0
    best_epoch = 0

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latestcheck.pth")
    if os.path.exists(checkpoint_path):
        print("Menemukan latestcheck.pth, melanjutkan training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        early_stopping.best_loss = checkpoint.get('best_loss')
        best_epoch = checkpoint.get('best_epoch', 0)
        early_stopping.counter = checkpoint.get('early_stop_counter', 0)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Melanjutkan dari epoch {start_epoch + 1}")

    global_step = 0
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (view1, view2) in enumerate(train_loader):
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                z_i, z_j = model(view1), model(view2)
                loss = criterion(z_i, z_j)
                scaled_loss = loss / ACC_STEP

            scaler.scale(scaled_loss).backward()

            if ((batch_idx + 1) % ACC_STEP == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if batch_idx % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
                writer.add_scalar("Training/Batch_Loss", loss.item(), global_step)
                writer.add_scalar("Training/Learning_Rate", current_lr, global_step)

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_simclr(model, val_loader, criterion, device)

        print(f"=== Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ===")
        writer.add_scalar("Training/Epoch_Loss", train_loss, epoch + 1)
        writer.add_scalar("Validation/NTXent_Loss", val_loss, epoch + 1)

        is_best = early_stopping(val_loss)

        if is_best:
            best_epoch = epoch + 1
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            best_model_name = f"simclr_best_epoch_{best_epoch}.pth"
            torch.save(raw_model.backbone.state_dict(), os.path.join(CHECKPOINT_DIR, best_model_name))
            print(f"BEST SSL MODEL | Epoch: {best_epoch} | Val Loss: {val_loss:.4f} | {best_model_name}")

        if (epoch + 1) % 2 == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': early_stopping.best_loss,
                'best_epoch': best_epoch,
                'early_stop_counter': early_stopping.counter
            }
            torch.save(checkpoint_state, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
            torch.save(checkpoint_state, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early stopping pada epoch {epoch+1}")
            break

    print("\n" + "=" * 60)
    print("HASIL PRE-TRAINING SIMCLR")
    print("=" * 60)
    print(f"Best Epoch      : {best_epoch}")
    print(f"Best Val Loss   : {early_stopping.best_loss:.4f}")
    print(f"Best Checkpoint : simclr_best_epoch_{best_epoch}.pth")
    print("=" * 60)

    writer.close()
    print("Training SimCLR selesai.")

if __name__ == "__main__":
    main()