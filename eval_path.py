import os

# Base direktori tempat folder-folder hasil training Anda disimpan
BASE_MODEL_DIR = r"E:\batch 8\SSL CERVIX\training ssl"

# Pilih folder eksperimen yang ingin dievaluasi SEMUA file .pth-nya
# (contoh: "main", "tunigan 1", "tunigan 2")
EXPERIMENT_FOLDER = "tunigan 1"

# Gabungan path folder yang akan dibaca otomatis oleh evaluation.py
TARGET_FOLDER_PATH = os.path.join(BASE_MODEL_DIR, EXPERIMENT_FOLDER)