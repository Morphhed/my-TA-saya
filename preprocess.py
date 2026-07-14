import os
import cv2
from pathlib import Path
from config import RAW_DATA_DIR, PATCH_DATA_DIR, PATCH_SIZE

def extract_patches(input_dir, output_dir, patch_size=256):
    print(f"Mencari gambar di dalam folder dan sub-foldernya: {input_dir}")
    
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'):
        image_paths.extend(Path(input_dir).rglob(ext))
        image_paths.extend(Path(input_dir).rglob(ext.upper())) # Untuk mengatasi ekstensi huruf besar (misal .JPG)
    
    patch_count = 0
    
    if len(image_paths) == 0:
        print("Peringatan: Tidak ada gambar ditemukan! Pastikan path di config.py sudah benar.")
        return

    print(f"Ditemukan {len(image_paths)} gambar mentah. Memulai ekstraksi...")

    for img_path in image_paths:
        parent_folder = img_path.parent.name
        img_name = img_path.stem
        
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Gagal membaca gambar: {img_path}")
            continue
            
        h, w, _ = img.shape
        
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = img[y:y+patch_size, x:x+patch_size]
                
                # Filter: Abaikan area yang mayoritas putih/background kosong (sesuaikan threshold 240 jika perlu)
                if patch.mean() > 240: 
                    continue
                    
                patch_filename = f"{parent_folder}_{img_name}_patch_{y}_{x}.jpg"
                cv2.imwrite(os.path.join(output_dir, patch_filename), patch)
                patch_count += 1
                
    print(f"Selesai! Berhasil mengekstrak {patch_count} patches ke folder {output_dir}")

if __name__ == "__main__":
    extract_patches(RAW_DATA_DIR, PATCH_DATA_DIR, PATCH_SIZE)