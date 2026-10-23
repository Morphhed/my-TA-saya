from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

class SimCLRDataset(Dataset):
    def __init__(self, image_dir=None, image_paths=None, transform=None):
        if image_paths is not None:
            self.image_paths = list(image_paths)
        elif image_dir is not None:
            root = Path(image_dir)
            self.image_paths = sorted(str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        else:
            raise ValueError("image_dir atau image_paths wajib diberikan")
        if len(self.image_paths) == 0:
            raise RuntimeError("Tidak ada image ditemukan untuk SimCLR")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform is None:
            return image, image
        return self.transform(image), self.transform(image)

def get_simclr_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def patch_source_key(path):
    stem = Path(path).stem
    if "_patch_" in stem:
        return stem.rsplit("_patch_", 1)[0]
    return stem

def split_simclr_paths(image_dir, val_ratio=0.10, seed=42):
    root = Path(image_dir)
    paths = sorted(str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
    if len(paths) == 0:
        raise RuntimeError(f"Tidak ada image di {image_dir}")
    groups = {}
    for path in paths:
        key = patch_source_key(path)
        groups.setdefault(key, []).append(path)
    group_names = list(groups.keys())
    if len(group_names) < 2:
        raise RuntimeError("Minimal perlu 2 source image untuk split SSL train/validation")
    rng = random.Random(seed)
    rng.shuffle(group_names)
    val_count = max(1, int(round(len(group_names) * val_ratio)))
    if val_count >= len(group_names):
        val_count = len(group_names) - 1
    val_groups = set(group_names[:val_count])
    train_paths, val_paths = [], []
    for key, group_paths in groups.items():
        if key in val_groups:
            val_paths.extend(group_paths)
        else:
            train_paths.extend(group_paths)
    return train_paths, val_paths