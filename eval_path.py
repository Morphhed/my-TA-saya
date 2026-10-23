import os
from config import EXTERNAL_DRIVE, RAW_DATA_DIR, CHECKPOINT_DIR

TRAIN_DATA = RAW_DATA_DIR
TEST_DATA = os.path.join(EXTERNAL_DRIVE, "test")
EXPERIMENT_FOLDER = CHECKPOINT_DIR
HASIL_DIR = os.path.join(EXTERNAL_DRIVE, "hasil_knn_simclr")

os.makedirs(HASIL_DIR, exist_ok=True)