import os

# BASE DIREKTORI CHECKPOINT (Sesuai dengan output training di drive E: sebelumnya)
BASE_MODEL_DIR = r"E:\batch 8\SSL CERVIX V2\malhari + anno"

# PATH DATASET BARU
# (Ganti path TEST_DATA ke lokasi folder test 20% milikmu jika letaknya berbeda)
TRAIN_DATA = os.path.join(BASE_MODEL_DIR, "train")  
TEST_DATA = os.path.join(BASE_MODEL_DIR, "test")  

# Pilih folder eksperimen (karena checkpointmu sebelumnya tersimpan di folder 'checkpoints')
EXPERIMENT_FOLDER = "epochs trained"

# Folder untuk menyimpan hasil evaluasi (CM image dan report)
HASIL_DIR = os.path.join(BASE_MODEL_DIR, "hasil")

# Gabungan path folder yang akan dibaca otomatis oleh evaluation.py
TARGET_FOLDER_PATH = os.path.join(BASE_MODEL_DIR, EXPERIMENT_FOLDER)
