# Proyek TA: Peningkatan Kinerja Klasifikasi Tingkat Keparahan Lesi Pra Kanker Cervix dengan Teknik Self Supervised

## Mulai Menjalankan Proyek

Proyek ini menggunakan pendekatan *Self-Supervised Learning* (SimCLR) berbasis arsitektur ResNet-50 untuk mengekstrak fitur jaringan medis tanpa label, dilanjutkan dengan tahap *fine-tuning* terawasi (*supervised*) yang dilengkapi **CBAM (Convolutional Block Attention Module)** untuk klasifikasi 3 tingkat keparahan lesi pra-kanker serviks (`Type_1`, `Type_2`, `Type_3`). Dikembangkan dan dioptimalkan untuk berjalan di VS Code Terminal.

---

## Tuneable Hyperparameter

### `config.py`
- **`BATCH_SIZE`**: Ukuran batch data pada pre-training SimCLR (default: `64`).
- **`EPOCHS`**: Jumlah epoch pada pre-training SimCLR (default: `100`).
- **`LEARNING_RATE`**: Kecepatan belajar pada pre-training SimCLR (default: `1e-3`).
- **`PATCH_SIZE`**: Ukuran potongan citra medis (default: `256`).

### `model.py`
- **`projection_dim`**: Dimensi output dari *projection head* MLP (default: `128`).
- **`temperature`**: Parameter skala penolakan sampel negatif pada `NTXentLoss` (default: `0.5`).

### `train.py`
- **`weight_decay`**: Regularisasi L2 untuk mencegah *overfitting* pada Adam optimizer (default: `1e-6`).

### `dataset.py`
- **Augmentasi Data**: Parameter kekuatan transformasi citra SimCLR (*ColorJitter*: `brightness=0.4`, `contrast=0.4`, `saturation=0.4`, `p=0.8`; *GaussianBlur*: `kernel_size=9`, `sigma=(0.1, 2.0)`; *RandomResizedCrop*: `scale=(0.2, 1.0)`).

### `finetune.py`
- **`FT_BATCH_SIZE`**: Ukuran batch data khusus tahap *fine-tuning* (default: `64`).
- **`FT_EPOCHS`**: Jumlah epoch untuk tahap *fine-tuning* (default: `100`).
- **`FT_LR`**: Learning rate khusus untuk *fine-tuning* (default: `1e-4`).
- **`NUM_CLASSES`**: Jumlah kelas klasifikasi target (default: `3`).
- **`USE_SSL_WEIGHTS`**: Pengaturan A/B Testing (`True` = Menggunakan bobot *pre-trained* SimCLR + CBAM, `False` = Baseline ImageNet + CBAM).

### `preprocess.py`
- **Threshold Background**: Nilai batas kecerahan piksel rata-rata (`mean > 240` untuk area putih kosong atau `mean < 20` untuk *border* hitam) untuk membuang *patch* yang tidak informatif.

---

## Tahap 1: Persiapan Dataset (Manual)
Sebelum menjalankan kode apa pun, pastikan dataset dari kompetisi Kaggle (*Intel & MobileODT Cervical Cancer Screening*) sudah dibersihkan:
* Pindahkan isi folder `additional_Type_1_v2`, `additional_Type_2_v2`, dan `additional_Type_3_v2` ke dalam folder `train/train/Type_1`, `Type_2`, dan `Type_3`. Jika ada peringatan nama file ganda, pilih **"Keep Both"**.
* Berdasarkan pembaruan dataset Kaggle, pindahkan file `80.jpg` ke folder `Type_3`.
* Pastikan file `968.jpg` dan `1120.jpg` berada di folder `Type_1`.

---

## Tahap 2: Konfigurasi Path
Pastikan file `config.py` telah dikonfigurasi menggunakan struktur path lingkungan kerja Anda (misal: format Linux/WSL2 ataupun Terminal):
* `EXTERNAL_DRIVE = "F:\\BATCH 8\\SSL CERVIX\\intel-mobileodt-cervical-cancer-screening"`
* `RAW_DATA_DIR` = `os.path.join(EXTERNAL_DRIVE, 'train', 'train')`

---

## Tahap 3: Menjalankan Proyek (jika dijalankan di WSL2)

Buka terminal di dalam VS Code (pastikan berada di sistem operasi Ubuntu/WSL2) dan aktifkan *virtual environment*:

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
pip install torch torchvision scikit-learn opencv-python Pillow numpy matplotlib seaborn grad-cam tqdm
```
---

# Urutan Eksekusi Skrip (Dijalankan di Terminal)

## Langkah 1: Ekstraksi Data (Preprocessing)
* **File yang dieksekusi:** `preprocess.py`
* **Metode & Fungsi:** Mengekstrak potongan gambar medis (*patches*) berukuran 256x256 piksel dari dataset mentah (`RAW_DATA_DIR`) secara efisien menggunakan teknik komputasi paralel:
  * **Multiprocessing Paralel:** Memanfaatkan `ThreadPoolExecutor` untuk membagi beban ekstraksi gambar ke seluruh *core* CPU secara otomatis dengan pemantauan *progress bar* (`tqdm`).
  * **Pembersihan Citra Dua Arah (Mean Filtering):** Menghitung nilai rata-rata piksel (*mean*) pada setiap potongan. Potongan gambar akan dibuang jika terlalu terang/putih kosong (`mean > 240`) ATAU terlalu gelap/latar hitam border (`mean < 20`) untuk memastikan hanya area jaringan serviks yang relevan yang disimpan.
  * **Penyimpanan Terstruktur:** Hasil ekstraksi disimpan ke folder `dataset_patches` dengan format penamaan sistematis `{parent_folder}_{img_name}_patch_{y}_{x}.jpg` untuk mempertahankan jejak asal gambar mentahnya.
  * **Perintah Terminal:**
    ```bash
    python preprocess.py
    ```

## Langkah 2: Memulai Pre-Training (Self-Supervised Learning)
* **File yang dieksekusi:** `train.py`
* **Metode & Fungsi:** Menerapkan algoritma **SimCLR** berbasis *Self-Supervised Learning* untuk melatih *backbone* ResNet-50 memahami fitur visual jaringan serviks tanpa bergantung pada label:
  * **Pasangan Augmentasi SimCLR:** Setiap *patch* di-augmentasi acak menjadi dua sudut pandang (`view1` dan `view2`) melalui `SimCLRDataset` dan diproyeksikan menggunakan *Projection Head* MLP ($2048 \rightarrow 512 \rightarrow 128$).
  * **Optimasi & Loss Function:** Melatih model menggunakan **NT-Xent Loss**, Adam optimizer, dan `CosineAnnealingLR` scheduler.
  * **Akselerasi Dual-GPU & AMP:** Mendukung `nn.DataParallel` serta *Automatic Mixed Precision* (`torch.cuda.amp`) untuk menghemat konsumsi VRAM dan mempercepat proses iterasi *batch*.
  * **Auto-Resume Checkpoint:** Menyimpan status pelatihan lengkap setiap 2 epoch sekali (`latest_checkpoint.pth`).
  * **Ekstraksi Backbone Final:** Di akhir epoch 100, *projection head* dilepas dan hanya bobot murni *backbone* ResNet-50 (`simclr_resnet50_final_backbone.pth`) yang disimpan ke harddisk untuk siap digunakan di tahap *fine-tuning*.
  * **Perintah Terminal:**
    ```bash
    python train.py
    ```

## Langkah 3: Fine-Tuning Model Klasifikasi
* **File yang dieksekusi:** `finetune.py`
* **Metode & Fungsi:** Melakukan **Transfer Learning** dan **Supervised Learning** untuk tugas klasifikasi 3 tingkat keparahan kanker serviks (`Type_1`, `Type_2`, `Type_3`):
  * **Integrasi CBAM Attention Mechanism:** Membangun arsitektur kustom `ResNet50WithAttention` yang menggabungkan *backbone* ResNet-50 dengan modul *Channel Attention* dan *Spatial Attention* (CBAM) tepat setelah `layer4`. Hal ini memaksa model memfokuskan ekstraksi fitur pada area lesi jaringan pra-kanker dan mengabaikan distractor visual.
  * **Fitur A/B Testing (`USE_SSL_WEIGHTS`):** Saat `USE_SSL_WEIGHTS = True`, model memuat bobot *pre-trained* SimCLR (`simclr_resnet50_final_backbone.pth`) menggunakan `strict=False` untuk mentransfer pemahaman fitur SSL. Sementara saat `False`, model bertindak sebagai *baseline* standar ImageNet.
  * **Konsistensi Pembagian Data (80/20):** Dataset utuh dibagi menjadi 80% data latih dan 20% data validasi. Proses *split* dikunci menggunakan generator ber-*seed* khusus (`manual_seed(67)`) untuk menjamin subset validasi 100% identik dengan yang diuji pada `evaluation.py`.
  * **Penyimpanan Bebas Wrapper Multi-GPU:** Sebelum disimpan ke `best_classifier_model.pth` dan `final_classifier_model.pth`, pembungkus `nn.DataParallel` dibongkar secara otomatis (`model.module`) agar aman dimuat ulang tanpa terjadi *key error*.
  * **Perintah Terminal:**
    ```bash
    python finetune.py
    ```

## Langkah 4: Evaluasi dan Visualisasi Kinerja Model
* **File yang dieksekusi:** `evaluation.py`
* **Metode & Fungsi:** Mengevaluasi dan memvisualisasikan model klasifikasi final (`final_classifier_model.pth`) melalui dua pendekatan analisis:
  * **Confusion Matrix Heatmap (`plot_confusion_matrix`):** Memuat kembali subset 20% data validasi murni menggunakan *seed* yang identik (`manual_seed(67)`). Skrip mengalkulasi matriks prediksi aktual vs prediksi model dan memvisualisasikannya dalam bentuk *heatmap* interaktif berbasis `seaborn` untuk menganalisis misklasifikasi antar-kelas.
  * **Explainable AI / Grad-CAM (`generate_gradcam`):** Menerapkan algoritma **Grad-CAM** yang menargetkan lapisan konvolusi terakhir (`model.layer4[-1]`) pada arsitektur `ResNet50WithAttention`. Metode ini menghasilkan *heatmap* transparan berwarna yang ditumpuk di atas citra serviks asli untuk memverifikasi secara visual bahwa keputusan prediksi model didasarkan pada area jaringan organ yang tepat.
  * **Perintah Terminal:**
    ```bash
    python evaluation.py
    ```