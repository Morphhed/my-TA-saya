import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from config import TRAIN_DIR, PATCH_DIR, PATCH_SIZE, OUT_41, MAIN_CODE_ROOT

import sys

if str(MAIN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_CODE_ROOT))

from dataset import get_simclr_transforms, VALID_EXTENSIONS


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

VISUAL_MEAN_MIN = 60
VISUAL_MEAN_MAX = 200

LAST_SELECTION_FILE = OUT_41 / ".last_visual_selection.txt"


def tensor_to_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * STD + MEAN
    return np.clip(image, 0, 1)


def get_image_mean(path):
    try:
        image = np.array(Image.open(path).convert("RGB"))
        return float(image.mean())
    except Exception:
        return None


def load_previous_selection():
    if not LAST_SELECTION_FILE.exists():
        return set()

    with open(LAST_SELECTION_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_current_selection(keys):
    with open(LAST_SELECTION_FILE, "w", encoding="utf-8") as f:
        for key in keys:
            f.write(f"{key}\n")


def get_valid_patch_examples(max_images=3, exclude=None):
    exclude = exclude or set()

    files = [p for p in PATCH_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    if not files:
        raise RuntimeError(f"Tidak ditemukan patch di {PATCH_DIR}")

    normal_files = [p for p in files if p.name.lower().startswith("normal_")]
    candidates = normal_files if len(normal_files) >= max_images else files

    random.shuffle(candidates)

    representative = []

    for path in candidates:
        key = str(path)

        if key in exclude:
            continue

        mean_value = get_image_mean(path)

        if mean_value is None:
            continue

        if VISUAL_MEAN_MIN <= mean_value <= VISUAL_MEAN_MAX:
            representative.append(path)

        if len(representative) >= max_images:
            break

    if len(representative) < max_images:
        fallback = []

        for path in candidates:
            key = str(path)

            if key in exclude or path in representative:
                continue

            mean_value = get_image_mean(path)

            if mean_value is None:
                continue

            if 20 <= mean_value <= 240:
                fallback.append(path)

            if len(representative) + len(fallback) >= max_images:
                break

        representative.extend(fallback)

    if not representative:
        raise RuntimeError("Tidak ditemukan patch valid yang sesuai untuk visualisasi.")

    return representative[:max_images]


def find_filtered_examples(exclude=None):
    exclude = exclude or set()

    image_paths = [p for p in TRAIN_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]

    random.shuffle(image_paths)

    dark_example = None
    bright_example = None

    for image_path in image_paths:
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
        except Exception:
            continue

        h, w, _ = image.shape

        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue

        coordinates = [
            (y, x)
            for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE)
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE)
        ]

        random.shuffle(coordinates)

        for y, x in coordinates:
            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_mean = float(patch.mean())

            key = f"{image_path}|{y}|{x}"

            if key in exclude:
                continue

            if dark_example is None and patch_mean < 20:
                dark_example = (patch.copy(), image_path.name, patch_mean, key)

            if bright_example is None and patch_mean > 240:
                bright_example = (patch.copy(), image_path.name, patch_mean, key)

            if dark_example is not None and bright_example is not None:
                return dark_example, bright_example

    return dark_example, bright_example


def save_valid_patch_examples(patch_paths):
    fig, axes = plt.subplots(1, len(patch_paths), figsize=(4 * len(patch_paths), 4))

    if len(patch_paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, patch_paths):
        image = Image.open(path).convert("RGB")
        mean_value = get_image_mean(path)

        ax.imshow(image)
        ax.set_title(f"{path.name[:28]}\nMean={mean_value:.2f}")
        ax.axis("off")

    plt.tight_layout()

    output = OUT_41 / "4_1_valid_patch_examples.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {output}")


def save_filtered_examples(dark, bright):
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

    for ax, (title, (patch, filename, mean_value, key)) in zip(axes, examples):
        ax.imshow(patch)
        ax.set_title(f"{title}\nMean={mean_value:.2f}\n{filename[:30]}")
        ax.axis("off")

    plt.tight_layout()

    output = OUT_41 / "4_1_rejected_patch_examples.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {output}")


def save_augmentation_examples(patch_path):
    image = Image.open(patch_path).convert("RGB")
    transform = get_simclr_transforms()

    view1 = transform(image)
    view2 = transform(image)

    original_mean = get_image_mean(patch_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title(f"Original Patch\nMean={original_mean:.2f}")

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
    print(f"[Source Augmentation] {patch_path.name}")


def save_summary(augmentation_patch):
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
        f.write(f"Patch augmentasi         : {augmentation_patch.name}\n")
        f.write(f"Mean patch augmentasi    : {get_image_mean(augmentation_patch):.2f}\n")

    print(f"[Saved] {output}")


if __name__ == "__main__":
    previous_selection = load_previous_selection()
    current_selection = set()

    valid_patches = get_valid_patch_examples(3, exclude=previous_selection)

    for path in valid_patches:
        current_selection.add(str(path))

    augmentation_exclude = previous_selection | current_selection
    augmentation_patch = get_valid_patch_examples(1, exclude=augmentation_exclude)[0]
    current_selection.add(str(augmentation_patch))

    dark_example, bright_example = find_filtered_examples(exclude=previous_selection)

    if dark_example is not None:
        current_selection.add(dark_example[3])

    if bright_example is not None:
        current_selection.add(bright_example[3])

    save_valid_patch_examples(valid_patches)
    save_filtered_examples(dark_example, bright_example)
    save_augmentation_examples(augmentation_patch)
    save_summary(augmentation_patch)

    save_current_selection(current_selection)

    print("\nVisualisasi Bab 4.1 selesai.")
    print("Pemilihan gambar diacak dan tidak menggunakan contoh yang sama dengan run sebelumnya jika kandidat lain tersedia.")