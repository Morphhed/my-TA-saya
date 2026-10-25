from pathlib import Path
import importlib.util

# ============================================================
# PATH SOURCE CODE UTAMA
# ============================================================

MAIN_CODE_ROOT = Path(r"E:\batch 8\SSL CERVIX V2\malhari + anno\kodings\malhari_plus_anno")
MAIN_CONFIG_PATH = MAIN_CODE_ROOT / "config.py"

# ============================================================
# LOAD CONFIG DARI PROJECT UTAMA
# ============================================================

spec = importlib.util.spec_from_file_location("main_project_config", MAIN_CONFIG_PATH)
main_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_config)

# ============================================================
# ROOT PROJECT
# ============================================================

MAIN_ROOT = Path(main_config.EXTERNAL_DRIVE)

BAB4_ROOT = Path(r"E:\batch 8\SSL CERVIX V2\bab 4 malhari + anno")
OUTPUT_ROOT = BAB4_ROOT / "output"

# ============================================================
# DATA UTAMA
# ============================================================

TRAIN_DIR = Path(main_config.RAW_DATA_DIR)
TEST_DIR = MAIN_ROOT / "test"
PATCH_DIR = Path(main_config.PATCH_DATA_DIR)
CHECKPOINT_DIR = Path(main_config.CHECKPOINT_DIR)

# ============================================================
# HASIL EKSPERIMEN UTAMA
# ============================================================

KNN_DIR = MAIN_ROOT / "hasil_knn_simclr"
FINETUNE_DIR = MAIN_ROOT / "hasil_finetune_simclr"
TENSORBOARD_DIR = CHECKPOINT_DIR / "tensorboard_logs"

# ============================================================
# PARAMETER DARI CONFIG UTAMA
# ============================================================

TEMPERATURE = main_config.TEMPERATURE
PATCH_SIZE = main_config.PATCH_SIZE
INPUT_SIZE = main_config.INPUT_SIZE
WARMUP = main_config.WARMUP

# ============================================================
# OUTPUT BAB IV
# ============================================================

OUT_41 = OUTPUT_ROOT / "4_1_preprocessing"
OUT_42 = OUTPUT_ROOT / "4_2_ssl"
OUT_43 = OUTPUT_ROOT / "4_3_knn"
OUT_45 = OUTPUT_ROOT / "4_5_comparison"
OUT_46 = OUTPUT_ROOT / "4_6_gradcam"

for folder in [BAB4_ROOT, OUTPUT_ROOT, OUT_41, OUT_42, OUT_43, OUT_45, OUT_46]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# REPORT FINE-TUNING
# ============================================================

FINETUNE_REPORTS = {
    "Baseline": FINETUNE_DIR / "Report_ResNet50_BASELINE_ImageNet_CBAM.txt",
    "FT-A": FINETUNE_DIR / "Report_ResNet50_USULAN_SimCLR_CBAM_FT_A_Frozen.txt",
    "FT-B": FINETUNE_DIR / "Report_ResNet50_USULAN_SimCLR_CBAM_FT_B_Layer4.txt",
    "FT-C": FINETUNE_DIR / "Report_ResNet50_USULAN_SimCLR_CBAM_FT_C_Full.txt"
}

# ============================================================
# CHECKPOINT FINE-TUNING
# ============================================================

FINETUNE_CHECKPOINTS = {
    "baseline": FINETUNE_DIR / "ResNet50_BASELINE_ImageNet_CBAM_finetuned_best.pth",
    "frozen": FINETUNE_DIR / "ResNet50_USULAN_SimCLR_CBAM_FT_A_Frozen_finetuned_best.pth",
    "layer4": FINETUNE_DIR / "ResNet50_USULAN_SimCLR_CBAM_FT_B_Layer4_finetuned_best.pth",
    "full": FINETUNE_DIR / "ResNet50_USULAN_SimCLR_CBAM_FT_C_Full_finetuned_best.pth"
}