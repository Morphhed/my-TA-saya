import os
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  
from config import RAW_DATA_DIR, PATCH_DATA_DIR, PATCH_SIZE
cv2.setNumThreads(0)

# Daftar ekstensi gambar yang diizinkan
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def process_single_image(args):
    """
    Fungsi worker untuk memotong (crop) satu gambar menjadi beberapa patch.
    """
    img_path, output_dir, patch_size = args
    parent_folder = img_path.parent.name
    img_name = img_path.stem
    patch_count = 0

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return 0, str(img_path)  # File korup/tidak terbaca

        h, w, _ = img.shape

        # Jika ukuran gambar lebih kecil dari patch_size, lewati
        if h < patch_size or w < patch_size:
            return 0, None

        # Slicing patch berdasarkan grid
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = img[y:y+patch_size, x:x+patch_size]

                # Filter background:
                # 1. Abaikan patch yang terlalu putih/kosong (mean > 240)
                # 2. Abaikan patch yang terlalu gelap/hitam border (mean < 20)
                patch_mean = patch.mean()
                if patch_mean > 240 or patch_mean < 20:
                    continue

                patch_filename = f"{parent_folder}_{img_name}_patch_{y}_{x}.jpg"
                output_path = os.path.join(output_dir, patch_filename)
                
                # Simpan patch dalam format JPG
                cv2.imwrite(output_path, patch)
                patch_count += 1

        return patch_count, None

    except Exception as e:
        return 0, f"{img_path} (Error: {str(e)})"


def extract_patches(input_dir, output_dir, patch_size=256, max_workers=None):
    """
    Mengekstrak patch secara paralel menggunakan seluruh core CPU.
    """
    print(f"Mencari gambar di dalam folder dan sub-foldernya: {input_dir}")
    
    # 1. Mengambil semua gambar dengan ekstensi yang valid
    input_path = Path(input_dir)
    image_paths = [
        p for p in input_path.rglob('*') 
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        print("Peringatan: Tidak ada gambar ditemukan! Pastikan RAW_DATA_DIR di config.py sudah benar.")
        return

    print(f"Ditemukan {len(image_paths)} gambar mentah. Memulai ekstraksi paralel...")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Persiapan tugas untuk multiprocessing
    tasks = [(p, output_dir, patch_size) for p in image_paths]
    
    total_patches = 0
    corrupted_files = []

    # Hitung chunksize optimal untuk mempercepat pembagian tugas pada CPU
    cpu_count = os.cpu_count() or 4
    chunksize = max(1, len(tasks) // (cpu_count * 4))

    # 3. Eksekusi paralel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_image, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Memproses Patches"
            )
        )

    # 4. Rekapitulasi hasil
    for count, failed_path in results:
        total_patches += count
        if failed_path:
            corrupted_files.append(failed_path)

    print(f"\nSelesai! Berhasil mengekstrak {total_patches} patches ke folder: {output_dir}")
    
    if corrupted_files:
        print(f"\n[Peringatan] Ditemukan {len(corrupted_files)} file rusak/gagal dibaca:")
        for f in corrupted_files[:10]:
            print(f" - {f}")
        if len(corrupted_files) > 10:
            print(f" ...dan {len(corrupted_files) - 10} file melebihinya.")


if __name__ == "__main__":
    extract_patches(RAW_DATA_DIR, PATCH_DATA_DIR, PATCH_SIZE)
