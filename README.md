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

## Persyaratan Sistem dan Library

Proyek ini membutuhkan beberapa pustaka (library) eksternal berbasis Python. Berikut adalah daftar library yang digunakan beserta fungsinya:

### 1. Deep Learning & Machine Learning
* **PyTorch** (`torch`): Framework utama untuk membangun model, mendefinisikan *loss function*, dan menjalankan proses pelatihan.
* **Torchvision** (`torchvision`): Digunakan untuk memuat arsitektur ResNet-50, memproses transformasi gambar (augmentasi), dan memuat dataset.
* **Scikit-Learn** (`scikit-learn`): Digunakan untuk menghitung metrik evaluasi kinerja model (akurasi, *Classification Report*, dan *Confusion Matrix*).

### 2. Pengolahan Citra (Image Processing)
* **OpenCV** (`opencv-python`): Digunakan pada tahap *preprocessing* untuk membaca gambar mentah dan memotongnya menjadi *patches*.
* **Pillow** (`Pillow`): Digunakan untuk membuka, mengubah format warna menjadi RGB, dan memanipulasi gambar saat memuat dataset dan menjalankan Grad-CAM.

### 3. Analisis Data & Visualisasi
* **NumPy** (`numpy`): Digunakan untuk komputasi numerik dan manipulasi matriks/array gambar.
* **Matplotlib** (`matplotlib`): Digunakan untuk menggambar grafik visual seperti *Confusion Matrix* dan hasil *Grad-CAM*.
* **Seaborn** (`seaborn`): Digunakan untuk mempercantik tampilan *Confusion Matrix* menjadi *heatmap* berwarna yang mudah dibaca.
* **PyTorch Grad-CAM** (`grad-cam`): Library khusus *(Explainable AI)* untuk menghasilkan *heatmap* visual guna menganalisis area fokus model saat memprediksi kelas gambar.

> **Catatan:** Proyek ini juga menggunakan pustaka bawaan Python seperti `os`, `pathlib`, dan `glob` yang otomatis tersedia dan tidak perlu diinstal secara terpisah.

---

## Cara Instalasi

Untuk mempermudah persiapan *environment* Anda, Anda dapat menginstal semua pustaka eksternal yang dibutuhkan secara bersamaan melalui terminal. 

Pastikan Anda sudah mengaktifkan *virtual environment* (misal: `source .venv/bin/activate`), lalu jalankan perintah berikut:

```bash
pip install torch torchvision scikit-learn opencv-python Pillow numpy matplotlib seaborn grad-cam

---

# Urutan Eksekusi Skrip (Dijalankan di Terminal)

## Langkah 1: Ekstraksi Data (Preprocessing)
*   **File yang dieksekusi:** `preprocess.py`
*   **Metode & Fungsi:** Menggunakan metode **Patch Extraction** untuk memotong gambar medis mentah dari folder `train/train/` menjadi potongan (*patch*) berukuran 256x256 piksel. Proses ini menerapkan **Thresholding Filtering** yang menghitung rata-rata nilai piksel pada setiap potongan; jika area tersebut terlalu terang atau dominan putih (nilai > 240), maka potongan tersebut dianggap sebagai *background* kosong dan akan dibuang. Hasil potongan gambar yang relevan kemudian disimpan ke folder `dataset_patches`.
*   **Perintah Terminal:**
    ```bash
    python preprocess.py
    ```
*(Tunggu hingga proses ekstraksi selesai dan muncul keterangan jumlah patch yang berhasil diekstrak)*.

## Langkah 2: Memulai Pre-Training (Self-Supervised Learning)
*   **File yang dieksekusi:** `train.py`
*   **Metode & Fungsi:** Tahap ini menerapkan algoritma **SimCLR** untuk melatih model memahami struktur visual sel tanpa memerlukan label kelas. Setiap *patch* gambar akan diberikan augmentasi ekstrem untuk menghasilkan dua versi yang berbeda secara acak (`view1` dan `view2`). Arsitektur **ResNet-50** kemudian dilatih menggunakan metode *Contrastive Learning* dengan fungsi **NT-Xent Loss** untuk menarik fitur gambar yang sama dan menjauhkan fitur gambar yang berbeda di dalam sebuah *batch*. Setelah proses selesai, skrip ini membuang *projection head* dan hanya menyimpan bobot kemajuan murni dari *backbone* ResNet-50 (*checkpoints*) ke harddisk secara otomatis.
*   **Perintah Terminal:**
    ```bash
    python train.py
    ```

## Langkah 3: Fine-Tuning Model Klasifikasi
*   **File yang dieksekusi:** `finetune.py`
*   **Metode & Fungsi:** Melakukan **Transfer Learning** dan **Supervised Learning** untuk tugas klasifikasi multikelas. Skrip ini secara otomatis memuat bobot *pre-trained* SimCLR yang telah dilatih pada Langkah 2 (sebagai model usulan), sehingga model tidak mulai belajar dari nol. Selanjutnya, *classification head* disesuaikan untuk memprediksi 3 kelas tingkat keparahan kanker serviks. Model dilatih menggunakan fungsi **Cross-Entropy Loss** dengan dataset gambar utuh yang dibagi menjadi 80% data latih dan 20% data validasi. Kinerja model dievaluasi menggunakan *Classification Report*, dan model klasifikasi final disimpan ke harddisk dengan nama `final_classifier_model.pth`.
*   **Perintah Terminal:**
    ```bash
    python finetune.py
    ```

## Langkah 4: Evaluasi dan Visualisasi Kinerja Model
*   **File yang dieksekusi:** `evaluation.py`
*   **Metode & Fungsi:** Mengevaluasi dan memvisualisasikan kinerja model klasifikasi final. Skrip ini membaca data validasi untuk membangun dan menampilkan **Confusion Matrix**, guna melihat secara rinci sebaran letak keakuratan dan kesalahan prediksi model pada setiap kelas. Selain itu, skrip ini mengimplementasikan metode **Explainable AI (XAI)** menggunakan algoritma **Grad-CAM**. Metode ini menghasilkan *heatmap* visual yang ditumpuk pada gambar asli, memungkinkan kita untuk melihat area spesifik mana pada gambar serviks yang menjadi fokus utama model dalam mengambil keputusan.
*   **Perintah Terminal:**
    ```bash
    python evaluation.py
    ```