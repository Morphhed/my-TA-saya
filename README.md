# Proyek TA: Peningkatan Kinerja Klasifikasi Tingkat Keparahan Lesi Pra Kanker Cervix dengan Teknik Self Supervised

## Mulai Menjalankan Proyek

Proyek ini menggunakan pendekatan *Self-Supervised Learning* (SimCLR) berbasis arsitektur ResNet-50 untuk mengekstrak fitur jaringan medis tanpa label, dilanjutkan dengan tahap *fine-tuning* terawasi (*supervised*) yang dilengkapi **CBAM (Convolutional Block Attention Module)** untuk klasifikasi 3 tingkat keparahan lesi pra-kanker serviks (`Type_1`, `Type_2`, `Type_3`). Dikembangkan dan dioptimalkan untuk berjalan di VS Code Terminal.

**Sumber Dataset**  
Data citra medis yang digunakan dalam proyek ini bersumber dari kompetisi Kaggle berikut:  
🔗 [Intel & MobileODT Cervical Cancer Screening Dataset](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)

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
* `EXTERNAL_DRIVE = "E:\\batch 8\\SSL CERVIX\\intel-mobileodt-cervical-cancer-screening"`
* `RAW_DATA_DIR` = `os.path.join(EXTERNAL_DRIVE, 'train', 'train')`

---

## Tahap 3: Membuat dan Mengaktifkan Virtual Environment (Windows PowerShell)

Buka terminal di dalam VS Code (pastikan menggunakan terminal PowerShell). Berikut adalah langkah-langkah untuk membuat *environment* Python, mengatur izin eksekusi skrip (jika diperlukan), dan mengaktifkannya:

**1. Membuat Virtual Environment**
Jalankan perintah berikut untuk membuat *environment* baru bernama `venv`:
```powershell
py -m venv venv
```

**2. Mengatur Permission (Bypass Error)**
Jalankan perintah ini hanya jika Anda mengalami error (seperti "running scripts is disabled") saat mencoba mengaktifkan environment :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**3. Mengaktifkan Virtual Environment**
Setelah environment berhasil dibuat dan izin diberikan, aktifkan dengan perintah berikut :
```powershell
.\venv\Scripts\Activate.ps1
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

Pastikan Anda sudah mengaktifkan *virtual environment* (misal: `source .venv/bin/activate`atau `py -m venv venv`), lalu jalankan perintah berikut:

```bash
pip install torch torchvision scikit-learn opencv-python Pillow numpy matplotlib seaborn grad-cam tqdm
```

### 1. Instalasi Versi CPU (Tanpa GPU)
Gunakan opsi ini jika tidak memiliki GPU NVIDIA

Jalankan perintah berikut:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install scikit-learn opencv-python Pillow numpy matplotlib seaborn grad-cam tqdm
```

### 2. Instalasi Versi CUDA
Gunakan opsi ini jika memiliki GPU NVIDIA 

Jalankan perintah berikut:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install scikit-learn opencv-python Pillow numpy matplotlib seaborn grad-cam tqdm
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
