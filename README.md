# Proyek TA: Peningkatan Kinerja Klasifikasi Tingkat Keparahan Lesi Pra Kanker Cervix dengan Teknik Self Supervised

## Mulai Menjalankan Proyek

Proyek ini menggunakan pendekatan *Self-Supervised Learning* (SimCLR) dengan arsitektur ResNet50 untuk mengekstrak fitur dari gambar medis sebelum dilakukan *fine-tuning* untuk klasifikasi. Dikembangkan dan dioptimalkan untuk berjalan di built-in Terminal vscode.

---
## Tuneable Hyperparameter

[config.py]
- BATCH_SIZE : Ukuran batch data pada pre-training SimCLR (default: 64).
- EPOCHS : Jumlah epoch pada pre-training SimCLR (default: 100).
- LEARNING_RATE : Kecepatan belajar pada pre-training SimCLR (default: 1e-3).
- PATCH_SIZE : Ukuran potongan gambar asli (default: 256).

[model.py]
- projection_dim : Dimensi output dari projection head (default: 128).
- temperature : Parameter tingkat penolakan gambar negatif pada NTXentLoss (default: 0.5).

[train.py]
- weight_decay : Regularisasi untuk mencegah overfitting pada optim.Adam (default: 1e-6).

[dataset.py]
- Augmentasi data : Kekuatan dan probabilitas augmentasi SimCLR (contoh: brightness=0.4, p=0.8 pada ColorJitter, kernel_size=9 pada GaussianBlur, serta proporsi crop 0.2-1.0).

[finetune.py]
- FT_BATCH_SIZE : Ukuran batch data khusus untuk tahap fine-tuning/klasifikasi (default: 64).
- FT_EPOCHS : Jumlah epoch untuk tahap fine-tuning (default: 100).
- FT_LR : Learning rate khusus untuk tahap fine-tuning (default: 1e-4).
- USE_SSL_WEIGHTS : Pengaturan A/B Testing (True = Menggunakan bobot pre-trained SimCLR, False = Baseline ImageNet).

[preprocess.py]
- threshold background : Nilai batas (default: 240) untuk membuang patch gambar yang dominan putih atau kosong.

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
```
---

# Urutan Eksekusi Skrip (Dijalankan di Terminal)

## Langkah 1: Ekstraksi Data (Preprocessing)
*   **File yang dieksekusi:** `preprocess.py`
*   **Metode & Fungsi:** Mengekstrak potongan gambar medis (*patches*) berukuran 256x256 piksel dari dataset mentah (`RAW_DATA_DIR`) secara efisien menggunakan teknik komputasi paralel:
    *   **Multiprocessing Paralel:** Memanfaatkan `ThreadPoolExecutor` untuk membagi beban ekstraksi gambar ke seluruh *core* CPU secara otomatis dengan pemantauan *progress bar* (`tqdm`).
    *   **Pembersihan Citra Dua Arah (Mean Filtering):** Menghitung nilai rata-rata piksel (*mean*) pada setiap potongan. Potongan gambar akan dibuang jika terlalu terang/putih kosong (`mean > 240`) ATAU terlalu gelap/latar hitam border (`mean < 20`) untuk memastikan hanya area jaringan serviks yang relevan yang disimpan.
    *   **Penyimpanan Terstruktur:** Hasil ekstraksi disimpan ke folder `dataset_patches` dengan format penamaan sistematis `{parent_folder}_{img_name}_patch_{y}_{x}.jpg` untuk mempertahankan jejak asal gambar mentahnya.
*   **Perintah Terminal:**
    ```bash
    python preprocess.py
    ```

## Langkah 2: Memulai Pre-Training (Self-Supervised Learning)
*   **File yang dieksekusi:** `train.py`
*   **Metode & Fungsi:** Tahap ini menerapkan algoritma **SimCLR** berbasis *Self-Supervised Learning* untuk melatih model memahami fitur visual jaringan serviks tanpa bergantung pada label. Setiap *patch* gambar di-augmentasi acak menjadi dua sudut pandang (`view1` dan `view2`). Arsitektur **ResNet-50** dilatih menggunakan **NT-Xent Loss** dengan beberapa optimisasi tingkat tinggi:
    *   **Multi-GPU & Pipeline Data:** Mendukung *DataParallel* otomatis untuk akselerasi Dual-GPU serta alokasi `pin_memory` & `non_blocking` untuk mempercepat pemuatan data dari RAM ke VRAM.
    *   **Automatic Mixed Precision (AMP):** Menggunakan `torch.cuda.amp` (FP16/FP32) guna menghemat konsumsi VRAM hingga 50% dan mempercepat proses iterasi *batch*.
    *   **Cosine Annealing LR Scheduler:** Menyesuaikan nilai *learning rate* secara bertahap dari `1e-3` hingga mendekati `0` sepanjang 100 epoch agar konvergensi *loss* lebih optimal.
    *   **Checkpoint & Auto-Resume:** Menyimpan status pelatihan (model, *optimizer*, *scheduler*, dan *scaler*) setiap 2 epoch sekali. Jika proses terhenti, skrip akan otomatis melanjutkan dari epoch terakhir.
    *   **Ekstraksi Model Final:** Di akhir epoch, *projection head* dibuang dan hanya bobot murni *backbone* ResNet-50 (`simclr_resnet50_final_backbone.pth`) yang disimpan ke harddisk untuk siap digunakan di tahap *fine-tuning*.
*   **Perintah Terminal:**
    ```bash
    python train.py
    ```

## Langkah 3: Fine-Tuning Model Klasifikasi
*   **File yang dieksekusi:** `finetune.py`
*   **Metode & Fungsi:** Melakukan **Transfer Learning** dan **Supervised Learning** untuk tugas klasifikasi 3 tingkat keparahan kanker serviks (Type_1, Type_2, Type_3):
    *   **Konsistensi Pembagian Data (80/20):** Dataset utuh dibagi menjadi 80% data latih dan 20% data validasi. Proses *split* menggunakan generator dengan *seed* terkunci (`manual_seed(67)`) untuk menjamin pembagian data selalu konsisten dan dapat direplikasi.
    *   **Fitur A/B Testing (`USE_SSL_WEIGHTS`):** Mendukung pengujian komparatif antara **Model Usulan** (memuat bobot *pre-trained* SimCLR `simclr_resnet50_final_backbone.pth` hasil Langkah 2) dan **Model Baseline** (menggunakan bobot standar ImageNet).
    *   **Penyesuaian Arsitektur & Multi-GPU:** Mengganti lapisan *Fully Connected* (`fc`) ResNet-50 menjadi 3 luaran kelas (`NUM_CLASSES = 3`) serta mendukung akselerasi multi-GPU menggunakan `nn.DataParallel`.
    *   **Evaluasi Kinerja Komprehensif:** Dilatih menggunakan **Cross-Entropy Loss** dan optimizer **Adam** (`lr=1e-4`, 100 epoch). Di akhir pelatihan, skrip mengevaluasi data validasi dan menampilkan `classification_report` (Presisi, Recall, F1-Score) serta menyimpan model klasifikasi final ke `final_classifier_model.pth`.
*   **Perintah Terminal:**
    ```bash
    python finetune.py
    ```

## Langkah 4: Evaluasi dan Visualisasi Kinerja Model
*   **File yang dieksekusi:** `evaluation.py`
*   **Metode & Fungsi:** Mengevaluasi dan memvisualisasikan model klasifikasi final (`final_classifier_model.pth`) melalui dua pendekatan analisis:
    *   **Confusion Matrix Heatmap (`plot_confusion_matrix`):** Memuat kembali data validasi 20% murni menggunakan *seed* yang identik (`manual_seed(67)`). Skrip ini mengalkulasi sebaran prediksi aktual vs prediksi model, lalu menggambarkannya dalam bentuk *heatmap* visual interaktif berbasis `seaborn` dan `matplotlib` untuk mendeteksi tingkat misklasifikasi antar-kelas.
    *   **Explainable AI / Grad-CAM (`generate_gradcam`):** Menerapkan algoritma **Grad-CAM** yang menyasar lapisan konvolusi terakhir (`model.layer4[-1]`). Metode ini menghasilkan peta visualisasi *heatmap* berwarna yang ditumpuk di atas gambar serviks asli, memungkinkan peneliti memverifikasi area spesifik organ serviks yang menjadi fokus perhatian utama model dalam mengambil keputusan.
*   **Perintah Terminal:**
    ```bash
    python evaluation.py
    ```