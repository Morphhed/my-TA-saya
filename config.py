import os

# GANTI DENGAN PATH FOLDER UTAMA DATASET ANDA DI HARDDISK
# Contoh: "D:/datasetTA" atau folder terluar tempat gambar mentah berada
EXTERNAL_DRIVE = "/mnt/d/TA BOS/intel-mobileodt-cervical-cancer-screening"

# Jika Anda ingin mengambil data dari folder 'train'
RAW_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'train', 'train')

# Folder output (akan otomatis dibuat di dalam folder intel-mobileodt-cervical-cancer-screening)
PATCH_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'dataset_patches')
CHECKPOINT_DIR = os.path.join(EXTERNAL_DRIVE, 'checkpoints')

# Parameter Training
PATCH_SIZE = 256
BATCH_SIZE = 16 # Gunakan 16 atau 32 agar RAM/CPU lokal Anda tidak terlalu berat
EPOCHS = 100
LEARNING_RATE = 1e-3

# Buat folder output jika belum ada
os.makedirs(PATCH_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)