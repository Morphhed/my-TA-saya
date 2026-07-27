import os

# GANTI DENGAN PATH FOLDER UTAMA DATASET ANDA DI HARDDISK
EXTERNAL_DRIVE = "F:\\BATCH 8\\SSL CERVIX\\intel-mobileodt-cervical-cancer-screening"

# Jika Anda ingin mengambil data dari folder 'train'
RAW_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'train', 'train')

# Folder output (akan otomatis dibuat di dalam folder intel-mobileodt-cervical-cancer-screening)
PATCH_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'dataset_patches')
CHECKPOINT_DIR = os.path.join(EXTERNAL_DRIVE, 'checkpoints')

# Parameter Training (not for finetune.py)
PATCH_SIZE = 256
BATCH_SIZE = 64 
EPOCHS = 100
LEARNING_RATE = 1e-3

# Buat folder output jika belum ada
os.makedirs(PATCH_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)