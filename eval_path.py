import os

# BASE DIREKTORI CHECKPOINT (Sesuai dengan output training di drive E: sebelumnya)
BASE_MODEL_DIR = r"C:\TA\malhari + anno"

# PATH DATASET BARU
# (Ganti path TEST_DATA ke lokasi folder test 20% milikmu jika letaknya berbeda)
TRAIN_DATA = os.path.join(BASE_MODEL_DIR, "train")  
TEST_DATA = os.path.join(BASE_MODEL_DIR, "test")  

# Pilih folder eksperimen (karena checkpointmu sebelumnya tersimpan di folder 'checkpoints')
EXPERIMENT_FOLDER = r"D:\TA BOS\malhari + anno\epochs trained temp 0.3"

# Folder untuk menyimpan hasil evaluasi (CM image dan report)
HASIL_DIR = os.path.join(BASE_MODEL_DIR, "hasil")