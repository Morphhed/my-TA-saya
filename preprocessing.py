import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from config import TRAIN_DIR, PATCH_DIR, PATCH_SIZE, OUT_41, MAIN_CODE_ROOT

import sys

if str(MAIN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_CODE_ROOT))

from dataset import get_simclr_transforms
from preprocess import VALID_EXTENSIONS

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def tensor_to_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * STD + MEAN
    return np.clip(image, 0, 1)

def get_valid_patch_examples(max_images=3):
    files = sorted([p for p in PATCH_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])
    normal_files = [p for p in files if p.name.lower().startswith("normal_")]

    if len(normal_files) >= max_images:
        return normal_files[:max_images]

    return files[:max_images]

def find_filtered_examples():
    dark_example = None
    bright_example = None

    image_paths = sorted([p for p in TRAIN_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])

    for image_path in image_paths:
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
        except Exception:
            continue

        h, w, _ = image.shape

        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue

        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_mean = float(patch.mean())

                if dark_example is None and patch_mean < 20:
                    dark_example = (patch.copy(), image_path.name, patch_mean)

                if bright_example is None and patch_mean > 240:
                    bright_example = (patch.copy(), image_path.name, patch_mean)

                if dark_example is not None and bright_example is not None:
                    return dark_example, bright_example

    return dark_example, bright_example

def save_valid_patch_examples():
    patch_paths = get_valid_patch_examples(3)

    if not patch_paths:
        raise RuntimeError(f"Tidak ditemukan patch di {PATCH_DIR}")

    fig, axes = plt.subplots(1, len(patch_paths), figsize=(4 * len(patch_paths), 4))

    if len(patch_paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, patch_paths):
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.set_title(path.name[:35])
        ax.axis("off")

    plt.tight_layout()
    output = OUT_41 / "4_1_valid_patch_examples.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {output}")

def save_filtered_examples():
    dark, bright = find_filtered_examples()

    examples = []

    if dark is not None:
        examples.append(("Rejected: Mean < 20", dark))

    if bright is not None:
        examples.append(("Rejected: Mean > 240", bright))

    if not examples:
        print("[Warning] Tidak ditemukan contoh patch <20 atau >240.")
        return

    fig, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 5))

    if len(examples) == 1:
        axes = [axes]

    for ax, (title, (patch, filename, mean_value)) in zip(axes, examples):
        ax.imshow(patch)
        ax.set_title(f"{title}\nMean={mean_value:.2f}\n{filename[:30]}")
        ax.axis("off")

    plt.tight_layout()
    output = OUT_41 / "4_1_rejected_patch_examples.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {output}")

def save_augmentation_examples():
    patch_paths = get_valid_patch_examples(1)

    if not patch_paths:
        raise RuntimeError("Tidak ada patch valid.")

    image = Image.open(patch_paths[0]).convert("RGB")
    transform = get_simclr_transforms()

    view1 = transform(image)
    view2 = transform(image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original Patch")

    axes[1].imshow(tensor_to_image(view1))
    axes[1].set_title("Augmented View 1")

    axes[2].imshow(tensor_to_image(view2))
    axes[2].set_title("Augmented View 2")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output = OUT_41 / "4_1_augmented_views.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {output}")

def save_summary():
    patch_files = [p for p in PATCH_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    raw_files = [p for p in TRAIN_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    output = OUT_41 / "4_1_preprocessing_summary.txt"

    with open(output, "w", encoding="utf-8") as f:
        f.write("HASIL PREPROCESSING\n")
        f.write("=" * 60 + "\n")
        f.write(f"Jumlah citra development : {len(raw_files)}\n")
        f.write(f"Jumlah patch tersimpan   : {len(patch_files)}\n")
        f.write(f"Patch size               : {PATCH_SIZE}x{PATCH_SIZE}\n")
        f.write(f"Stride                   : {PATCH_SIZE}\n")
        f.write("Filter gelap             : mean < 20\n")
        f.write("Filter terang            : mean > 240\n")

    print(f"[Saved] {output}")

if __name__ == "__main__":
    save_valid_patch_examples()
    save_filtered_examples()
    save_augmentation_examples()
    save_summary()

    print("\nVisualisasi Bab 4.1 selesai.")