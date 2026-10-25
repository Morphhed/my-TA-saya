from pathlib import Path
import sys

MAIN_CODE_ROOT = Path(r"E:\batch 8\SSL CERVIX V2\malhari + anno\kodings\malhari_plus_anno")

if str(MAIN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_CODE_ROOT))

import config as main_config

MAIN_ROOT = Path(main_config.EXTERNAL_DRIVE)
BAB4_ROOT = MAIN_ROOT.parent / f"{MAIN_ROOT.name}_BAB4"
OUTPUT_ROOT = BAB4_ROOT / "output"

TRAIN_DIR = Path(main_config.RAW_DATA_DIR)
TEST_DIR = MAIN_ROOT / "test"
PATCH_DIR = Path(main_config.PATCH_DATA_DIR)
CHECKPOINT_DIR = Path(main_config.CHECKPOINT_DIR)

KNN_DIR = MAIN_ROOT / "hasil_knn_simclr"
FINETUNE_DIR = MAIN_ROOT / "hasil_finetune_simclr"
TENSORBOARD_DIR = CHECKPOINT_DIR / "tensorboard_logs"

TEMPERATURE = main_config.TEMPERATURE
PATCH_SIZE = main_config.PATCH_SIZE
INPUT_SIZE = main_config.INPUT_SIZE
WARMUP = main_config.WARMUP

OUT_41 = OUTPUT_ROOT / "4_1_preprocessing"
OUT_42 = OUTPUT_ROOT / "4_2_ssl"
OUT_43 = OUTPUT_ROOT / "4_3_knn"
OUT_45 = OUTPUT_ROOT / "4_5_comparison"
OUT_46 = OUTPUT_ROOT / "4_6_gradcam"

for folder in [BAB4_ROOT, OUTPUT_ROOT, OUT_41, OUT_42, OUT_43, OUT_45, OUT_46]:
    folder.mkdir(parents=True, exist_ok=True)

FINETUNE_REPORTS = {
    "Baseline": FINETUNE_DIR / "Report_ResNet50_BASELINE_ImageNet_CBAM.txt",
    "FT-A": FINETUNE_DIR / "Report_ResNet50_USULAN_SimCLR_CBAM_FT_A_Frozen.txt",
    "FT-B": FINETUNE_DIR / "Report_ResNet50_USULAN_SimCLR_CBAM_FT_B_Layer4.txt",
    "FT-C": FINETUNE_DIR / "Report_ResNet50_USULAN_SimCLR_CBAM_FT_C_Full.txt"
}

FINETUNE_CHECKPOINTS = {
    "baseline": FINETUNE_DIR / "ResNet50_BASELINE_ImageNet_CBAM_finetuned_best.pth",
    "frozen": FINETUNE_DIR / "ResNet50_USULAN_SimCLR_CBAM_FT_A_Frozen_finetuned_best.pth",
    "layer4": FINETUNE_DIR / "ResNet50_USULAN_SimCLR_CBAM_FT_B_Layer4_finetuned_best.pth",
    "full": FINETUNE_DIR / "ResNet50_USULAN_SimCLR_CBAM_FT_C_Full_finetuned_best.pth"
}