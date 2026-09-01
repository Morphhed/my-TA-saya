import os
import shutil
import pandas as pd

# Mengambil variabel TEST_DATA langsung dari file konfigurasi Anda
from eval_path import TEST_DATA, CSV_PATH, SOURCE_TEST_DIR

# =====================================================================
# 1. KONFIGURASI PATH SUMBER (GANTI BAGIAN INI SESUAI LOKASI KAGGLE)
# =====================================================================
CSV_PATH = CSV_PATH  # Path ke file CSV Kaggle (berisi ground truth / kunci jawaban)
SOURCE_TEST_DIR = SOURCE_TEST_DIR  # Path ke folder test Kaggle (folder asli yang gambarnya masih tergabung)

# =====================================================================
# 2. PROSES PENYUSUNAN
# =====================================================================
def susun_dataset():
    # Menggunakan TEST_DATA dari eval_path_2.py sebagai direktori tujuan
    if not os.path.exists(TEST_DATA):
        os.makedirs(TEST_DATA)

    # Buat sub-folder untuk masing-masing kelas di dalam TEST_DATA
    classes = ['Type_1', 'Type_2', 'Type_3']
    for cls in classes:
        cls_dir = os.path.join(TEST_DATA, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

    print(f"Membaca file CSV dari: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    berhasil = 0
    gagal = 0

    print(f"Memulai proses copy gambar menuju: {TEST_DATA} ...")
    
    for index, row in df.iterrows():
        img_name = str(row.iloc[0]) 
        
        # Mengecek label One-Hot Encoding dari Kaggle
        if row['Type_1'] == 1:
            label = 'Type_1'
        elif row['Type_2'] == 1:
            label = 'Type_2'
        elif row['Type_3'] == 1:
            label = 'Type_3'
        else:
            continue

        src_path = os.path.join(SOURCE_TEST_DIR, img_name)
        dest_path = os.path.join(TEST_DATA, label, img_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            berhasil += 1
        else:
            print(f"[Peringatan] Gambar tidak ditemukan: {src_path}")
            gagal += 1

    print("\n" + "="*50)
    print("PROSES SELESAI!")
    print(f"Gambar berhasil disalin : {berhasil}")
    print(f"Gambar gagal/hilang     : {gagal}")
    print(f"Folder uji yang baru    : {TEST_DATA}")
    print("="*50)

if __name__ == "__main__":
    susun_dataset()