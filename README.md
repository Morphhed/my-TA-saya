# Ringkasan Perubahan / Tuning SimCLR untuk Citra Serviks

## Pertama
Berikut adalah ringkasan perubahan teknis dan peningkatan awal yang dilakukan pada skrip:

*   **Peningkatan Kapasitas Batch Size (`config.py`):**
    *   Mengubah nilai `BATCH_SIZE` dari `64` menjadi `128`. Peningkatan ini memanfaatkan kapasitas VRAM secara optimal untuk memperbanyak jumlah sampel negatif dalam satu iterasi, sehingga memperkuat representasi kontrastif model.
*   **Peningkatan Num_Workers (`config.py`):**
    *   Mengubah nilai `num_worker` dari `4` menjadi `8`.
*   **Penyesuaian Parameter Suhu / Temperature (`config.py` & `train.py`):**
    *   Menambahkan dan menurunkan parameter `TEMPERATURE` dari `0.5` menjadi `0.1`. Suhu yang lebih rendah memberikan penalti (*harsher penalty*) yang lebih ketat terhadap sampel negatif, memaksa model agar jauh lebih sensitif terhadap perbedaan tekstur dan detail halus pada jaringan serviks.
*   **Migrasi dari Multithreading ke Multiprocessing (`preprocess.py`):**
    *   Mengganti `ThreadPoolExecutor` dengan `ProcessPoolExecutor`. Pemotongan citra beresolusi tinggi dan kalkulasi matriks citra (seperti `patch.mean()`) adalah tugas *CPU-bound*. Multiprocessing menembus batasan GIL (*Global Interpreter Lock*) pada Python.
*   **Penyesuaian Augmentasi Khusus Medis (`dataset.py`):**
    *   **Penghapusan Grayscale:** Menghapus `transforms.RandomGrayscale` secara penuh karena warna dan kemerahan (*biomarker*) merupakan indikator esensial dalam mendeteksi lesi serviks.
    *   **Penambahan Rotasi Bebas:** Menyisipkan `transforms.RandomRotation(degrees=360)` untuk mengajari model bahwa jaringan biologis tidak memiliki orientasi mutlak atas-bawah.
    *   **Penambahan Deformasi Elastis:** Menyisipkan `transforms.ElasticTransform(alpha=50.0, sigma=5.0)` untuk memetakan sifat fleksibel/elastis dari jaringan organ seluler.
    *   **Penyetelan Jitter & Crop:** Membatasi kekuatan *ColorJitter* agar tidak merusak rona warna asli jaringan, serta menaikkan batas bawah *RandomResizedCrop* dari `0.2` ke `0.4` agar model tidak terlalu fokus pada area kosong/mikro.
*   **Penggantian Optimizer ke AdamW (`train.py`):**
    *   Mengganti `optim.Adam` dengan `optim.AdamW` disertai peningkatan nilai *weight decay* menjadi `1e-4`. Algoritma *decoupled weight decay* pada AdamW memberikan stabilitas dan regularisasi bobot yang jauh lebih baik untuk proses *Self-Supervised Learning* jangka panjang.

---

## Kedua 
Optimasi tingkat *engineering* untuk mencegah *representation collapse*, menstabilkan gradien, dan memantau performa model secara presisi:

*   **Penggantian Mesin Pembaca (OpenCV ke PIL):**
    *   Mengganti penggunaan `cv2.imread` dengan antarmuka pembacaan dari pustaka **Pillow (PIL)**. Sebelumnya, gambar dengan struktur JPEG yang cacat ekstrem (*premature end of JPEG*) membuat *decoder* C++ internal OpenCV terjebak dalam *infinite loop* (hang secara diam-diam tanpa memicu *error* di Python). Hal ini mengunci memori *worker* dan memicu *deadlock*.
*   **Toleransi Gambar Terpotong (*Truncated Images*):**
    *   Mengaktifkan parameter `ImageFile.LOAD_TRUNCATED_IMAGES = True`. Konfigurasi ini memaksa skrip untuk tetap memuat blok piksel yang masih selamat dari gambar yang terpotong, alih-alih langsung menggagalkannya, sehingga meminimalisir kehilangan data pelatihan.
*   **Peningkatan Kapasitas Projection Head (`model.py`):**
    *   Memperdalam arsitektur *projection head* dari yang awalnya standar menjadi **3-Layer MLP** (dengan ukuran 512 dimensi pada *hidden layer*). Ini terbukti secara signifikan mencegah nilai loss "menipu" (turun ke 0 karena model sekadar menghafal trik) dan memperkuat kualitas ekstraksi fitur dari *backbone* ResNet50.
*   **Pencegahan *Shortcut Learning* Antar GPU (`train.py`):**
    *   Mengonversi model menggunakan `nn.SyncBatchNorm.convert_sync_batchnorm` sebelum dibungkus dengan `DataParallel`. Hal ini krusial pada setup *Multi-GPU* agar model tidak menggunakan statistik *Batch Normalization* lokal sebagai contekan untuk mencocokkan gambar.
*   **Penerapan *Linear Warmup* & *Cosine Annealing* (`config.py` & `train.py`):**
    *   Menambahkan parameter `WARMUP = 10` dan menggunakan kombinasi `SequentialLR` untuk menggabungkan `LinearLR` (pemanasan bertahap di 10 epoch awal) dengan `CosineAnnealingLR`. Teknik ini mencegah *exploding gradients* saat bobot ResNet50 masih acak di awal fase *training*.
*   **Isolasi Parameter Bias & BatchNorm dari Weight Decay (`train.py`):**
    *   Mengimplementasikan fungsi khusus `configure_optimizer` yang memisahkan parameter *bias* dan layer 1D (BatchNorm). Parameter ini diberi *weight decay* `0.0` untuk mencegah model mengalami *underfitting*. Regularisasi hanya difokuskan pada matriks bobot (*weights*).
*   **Perlindungan *Gradient Clipping* (`train.py`):**
    *   Menambahkan perintah `torch.nn.utils.clip_grad_norm_` (max_norm=1.0) tepat setelah *loss backward* dan sebelum langkah *optimizer*. Ini membatasi lonjakan nilai gradien ekstrem yang bisa merusak bobot jaringan.
*   **Integrasi TensorBoard & Patch Scheduler (`train.py`):**
    *   Memasukkan modul `SummaryWriter` untuk memantau metrik pergerakan kurva *Loss* dan *Learning Rate* per batch secara visual.
    *   Menyisipkan *workaround* `scheduler.last_epoch += 1` sebagai penambal *bug* internal PyTorch pada `SequentialLR` agar kurva LR tidak macet (*stuck*) di tengah jalan.

## Donlod Versi Tuning (Git Clone)

```bash
git clone -b tuning-pertama https://github.com/Morphhed/my-TA-saya.git tuning-pertama
```

## Updated Library (Instalasi TensorBoard)
Pastikan pustaka TensorBoard telah terpasang di lingkungan Python / conda Anda untuk memantau grafik metrik *training*:

```bash
python -m pip install tensorboard
```