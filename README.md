# Proyek TA: Peningkatan Kinerja Klasifikasi Tingkat Keparahan Lesi Pra Kanker Cervix dengan Teknik Self Supervised

## Mulai Menjalankan Proyek

Proyek ini menggunakan pendekatan *Self-Supervised Learning* (SimCLR) dengan arsitektur ResNet50 untuk mengekstrak fitur dari gambar medis sebelum dilakukan *fine-tuning* untuk klasifikasi. Dikembangkan dan dioptimalkan untuk berjalan di lingkungan WSL2 (Windows Subsystem for Linux).

---
## Tuneable Hyperparameter

[config.py]
- BATCH_SIZE : Ukuran batch data yang diproses pada tahap pre-training SimCLR.
- EPOCHS : Berapa kali model melihat keseluruhan dataset pada tahap pre-training.
- LEARNING_RATE : Kecepatan model dalam belajar/mengubah bobot pada tahap pre-training.
- PATCH_SIZE : Ukuran potongan gambar asli.

[model.py]
- projection_dim : Dimensi output dari projection head (standar: 128).
- temperature : Parameter tingkat penolakan gambar negatif pada NTXentLoss (standar: 0.5).

[train.py]
- weight_decay : Regularisasi untuk mencegah overfitting pada optim.Adam (standar: 1e-6).

[dataset.py]
- Augmentasi data : Kekuatan dan probabilitas augmentasi (contoh: brightness=0.4, p=0.8 pada ColorJitter, atau ukuran blur pada GaussianBlur).

[finetune.py]
- FT_BATCH_SIZE : Ukuran batch data khusus untuk tahap fine-tuning/klasifikasi.
- FT_EPOCHS : Jumlah epoch untuk tahap fine-tuning.
- FT_LR : Learning rate khusus untuk tahap fine-tuning.
- USE_SSL_WEIGHTS : Pengaturan A/B Testing (True = Menggunakan bobot pre-trained SimCLR, False = Baseline ImageNet).

[preprocess.py]
- threshold background : Nilai batas (contoh: 240) untuk membuang patch gambar yang dominan putih atau kosong.

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

## Langkah 3: Fine-Tuning Model Klasifikasi
*   **File yang dieksekusi:** `finetune.py`
*   **Fungsi:** Melakukan *fine-tuning* pada model klasifikasi untuk 3 kelas tingkat keparahan kanker serviks[cite: 4]. Skrip ini secara otomatis memuat bobot *pre-trained* SimCLR yang telah dilatih pada Langkah 2 (model usulan), lalu melatih ulang *classification head* menggunakan dataset utuh yang dibagi menjadi 80% data latih dan 20% data validasi[cite: 4]. Setelah proses selesai dan metrik evaluasi ditampilkan, model klasifikasi final akan disimpan ke harddisk dengan nama `final_classifier_model.pth`[cite: 4].
*   **Perintah Terminal:**
    ```bash
    python finetune.py
    ```

## Langkah 4: Evaluasi dan Visualisasi Kinerja Model
*   **File yang dieksekusi:** `evaluation.py`
*   **Fungsi:** Memvisualisasikan hasil dari model klasifikasi final yang telah dilatih[cite: 3]. Skrip ini akan membaca data validasi untuk membangun dan menampilkan *Confusion Matrix* guna melihat letak keakuratan dan kesalahan prediksi model pada setiap kelas[cite: 3]. Selain itu, skrip ini menggunakan teknik *Grad-CAM* untuk menghasilkan *heatmap*, yang memungkinkan kita melihat area spesifik mana pada gambar serviks yang menjadi fokus utama model dalam mengambil keputusan[cite: 3].
*   **Perintah Terminal:**
    ```bash
    python evaluation.py
    ```