# Proyek TA: Peningkatan Kinerja Klasifikasi Tingkat Keparahan Lesi Pra Kanker Cervix dengan Teknik Self Supervised

## Mulai Menjalankan Proyek

Proyek ini menggunakan pendekatan *Self-Supervised Learning* (SimCLR) dengan arsitektur ResNet50 untuk mengekstrak fitur dari gambar medis sebelum dilakukan *fine-tuning* untuk klasifikasi. Dikembangkan dan dioptimalkan untuk berjalan di lingkungan WSL2 (Windows Subsystem for Linux).

---
## Tuneable Hyperparameter

[config.py]
- BATCH_SIZE : Ukuran batch data yang diproses.
- EPOCHS : Berapa kali model melihat keseluruhan dataset.
- LEARNING_RATE : Kecepatan model dalam belajar/mengubah bobot.
- PATCH_SIZE : Ukuran potongan gambar asli.

[model.py]
- projection_dim : Dimensi output dari projection head (standar: 128).
- temperature : Parameter tingkat penolakan gambar negatif pada NTXentLoss (standar: 0.5).

[train.py]
- weight_decay : Regularisasi untuk mencegah overfitting pada optim.Adam (standar: 1e-6).

[dataset.py]
- Augmentasi data : Kekuatan dan probabilitas augmentasi (contoh: brightness=0.4, p=0.8 pada ColorJitter, atau ukuran blur pada GaussianBlur).

---

## Tahap 1: Persiapan Dataset (Manual)
Sebelum menjalankan kode apa pun, pastikan dataset dari kompetisi Kaggle sudah dibersihkan:
* Pindahkan isi folder `additional_Type_1_v2`, `additional_Type_2_v2`, dan `additional_Type_3_v2` ke dalam folder `train/train/Type_1`, `Type_2`, dan `Type_3`. Jika ada peringatan nama file ganda, pilih **"Keep Both"**.
* Berdasarkan pembaruan Kaggle, pindahkan file `80.jpg` ke folder `Type_3`.
* Pastikan file `968.jpg` dan `1120.jpg` berada di folder `Type_1`.

## Tahap 2: Konfigurasi Path
Pastikan file `config.py` sudah menggunakan path format Linux (WSL2), bukan partisi Windows biasa.
* `EXTERNAL_DRIVE` = `/mnt/d/TA BOS/intel-mobileodt-cervical-cancer-screening`
* `RAW_DATA_DIR` = `os.path.join(EXTERNAL_DRIVE, 'train', 'train')`

## Tahap 3: Menjalankan Proyek

Buka terminal di dalam VS Code (pastikan berada di sistem operasi Ubuntu/WSL2) dan aktifkan *environment*:

```bash
source .venv/bin/activate
```
---

# Urutan Eksekusi Skrip (Dijalankan di Terminal)

## Langkah 1: Ekstraksi Data (Preprocessing)
*   **File yang dieksekusi:** `preprocess.py`
*   **Fungsi:** Mengambil gambar medis mentah dari folder `train/train/`, memotongnya menjadi *patch* berukuran 256x256, menyaring dan membuang area yang tidak relevan (background putih/kosong), lalu menyimpannya ke folder `dataset_patches`[cite: 35].
*   **Perintah Terminal:**
    ```bash
    python preprocess.py
    ```
*(Tunggu hingga proses ekstraksi selesai dan muncul keterangan jumlah patch yang berhasil diekstrak)*.

## Langkah 2: Memulai Pre-Training (Self-Supervised Learning)
*   **File yang dieksekusi:** `train.py`
*   **Fungsi:** Menyatukan logika dari `config.py`, `dataset.py`, dan `model.py`[cite: 37]. Skrip ini melatih model ResNet50 untuk mengenali fitur dari *patch* gambar yang sudah dibuat pada Langkah 1, mengevaluasi *loss*, dan menyimpan bobot kemajuan model (*checkpoints*) ke harddisk secara otomatis[cite: 37].
*   **Perintah Terminal:**
    ```bash
    python train.py
    ```
