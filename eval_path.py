import os

# Path ke file CSV Kaggle (berisi ground truth / kunci jawaban)
CSV_PATH = r"C:\TA\intel-mobileodt-cervical-cancer-screening\solution_stg1_release.csv"

# Path ke folder test Kaggle (folder asli yang gambarnya masih tergabung)
SOURCE_TEST_DIR = r"C:\TA\intel-mobileodt-cervical-cancer-screening\test\test"

# Untuk dataset test
TEST_DATA = r"C:\TA\dataset_Test"  
TRAIN_DATA = r"C:\TA\intel-mobileodt-cervical-cancer-screening\train\train"  

# Base direktori tempat folder-folder hasil training Anda disimpan
BASE_MODEL_DIR = r"D:\TA BOS\training ssl"

# Pilih folder eksperimen yang ingin dievaluasi SEMUA file .pth-nya
# (contoh: "main", "tunigan 1", "tunigan 2", etc)
EXPERIMENT_FOLDER = "tunigan 1"

# Gabungan path folder yang akan dibaca otomatis oleh evaluation.py
TARGET_FOLDER_PATH = os.path.join(BASE_MODEL_DIR, EXPERIMENT_FOLDER)