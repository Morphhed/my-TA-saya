import os

# 1. PATH UTAMA DATASET GABUNGAN
EXTERNAL_DRIVE = r"E:\batch 8\SSL CERVIX V2\malhari + anno"

# 2. PATH FOLDER TRAIN (Berisi subfolder 'Abnormal' dan 'Normal')
RAW_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'train')

# 3. FOLDER OUTPUT UNTUK PATCH & CHECKPOINT
PATCH_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'dataset_patches')
CHECKPOINT_DIR = os.path.join(EXTERNAL_DRIVE, 'checkpoints')

# Parameter Training (Tetap sama)
PATCH_SIZE = 256
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 3e-4      
TEMPERATURE = 0.3        
WEIGHT = 1e-2
WARMUP = 10  
WORKERS = 8
ACC_STEP = 2

# Buat folder output jika belum ada
os.makedirs(PATCH_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)