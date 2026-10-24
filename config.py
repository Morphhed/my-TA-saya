import os

# root directory dari percobaan
EXTERNAL_DRIVE = r"E:\batch 8\SSL CERVIX V2\malhari + anno"

# path untuk menyimpan data patch, checkpoint, dan hasil lainnya
RAW_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'train')
PATCH_DATA_DIR = os.path.join(EXTERNAL_DRIVE, 'dataset_patches_simclr')
CHECKPOINT_DIR = os.path.join(EXTERNAL_DRIVE, 'checkpoints_simclr')

# parameter untuk augmentasi dan training
PATCH_SIZE = 256
INPUT_SIZE = 224

BATCH_SIZE = 128
FINETUNE_BATCH_SIZE = 32

EPOCHS = 100
LEARNING_RATE = 3e-4
TEMPERATURE = 0.3
WEIGHT = 1e-4
WARMUP = 10
WORKERS = 8
ACC_STEP = 1

SSL_VAL_RATIO = 0.10
FINETUNE_VAL_RATIO = 0.20
SEED = 42

os.makedirs(PATCH_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)