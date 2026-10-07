# Dokumentasi Dataset: Malhari & AnnoCerv (Base dari "Tuning-Kedua")

Proyek klasifikasi kanker serviks ini (menggunakan ResNet-50 + CBAM + SimCLR) menggunakan gabungan dataset klinis/kolposkopi dari dua sumber publik untuk menghasilkan klasifikasi biner (**Normal** vs **Abnormal**).

## Sumber Dataset
* **Malhari Dataset:** Dataset citra klinis kanker serviks ([Mendeley Data](https://data.mendeley.com/datasets/m5kxdj7m36/1)).
* **AnnoCerv Dataset:** Dataset citra kolposkopi beserta anotasi/masking ([GitHub](https://github.com/iclx/AnnoCerv)).

## Dasar Klinis Pemetaan Kelas (Normal vs Abnormal)

**AnnoCerv (Berdasarkan Swede Score)**
* **Normal:** Swede Score < 5
* **Abnormal:** Swede Score ≥ 5
> **Justifikasi:** Sesuai standar klinis (Strander et al., 2005; Bowring et al., 2010), Swede Score < 5 mengindikasikan lesi *low-risk* yang tidak memerlukan biopsi rutin. Skor ≥ 5 adalah ambang batas klinis (*cutoff*) pendeteksian lesi pra-kanker derajat tinggi (CIN2+) yang wajib dibiopsi.

**Malhari (Berdasarkan Keparahan CIN)**
* **Normal:** CIN 1
* **Abnormal:** CIN 2 dan CIN 3
> **Justifikasi:** Merujuk pada panduan ASCCP dan standar *benchmark ML*, klasifikasi ini menggunakan ambang batas CIN2+. CIN 1 dikategorikan *low-risk* karena dapat mengalami regresi spontan, sedangkan CIN 2+ adalah lesi *high-risk* yang memerlukan tindakan medis.

## Penggabungan dan Split Dataset
File citra `.jpg` dari masing-masing dataset disatukan berdasarkan kelasnya (Normal dengan Normal, Abnormal dengan Abnormal). Kumpulan data gabungan ini kemudian dibagi secara acak dengan rasio:
* **Train (80%):** Untuk *pre-training* SSL dan *fine-tuning* model.
* **Test (20%):** Disisihkan khusus untuk evaluasi metrik akhir.

## Penyesuaian Preprocessing & Augmentasi untuk SimCLR
Untuk memastikan dataset kompatibel dengan *pipeline* PyTorch dan menjaga integritas fitur medis pada metode *Self-Supervised Learning* (SimCLR), dilakukan penyesuaian berikut:

### 1. Preprocessing Struktur File & Direktori
* **Perataan Direktori (Flat Directory):** Mengeluarkan gambar dari dalam folder masing-masing pasien sehingga struktur akhirnya langsung merujuk pada kelas (contoh: `train/Normal/image.jpg`).
* **Hanya Menggunakan Citra `.jpg`:** File anotasi `.png` (masking tepi) tidak diikutsertakan ke dalam *DataLoader*. Jika file `.png` ikut masuk ke dalam pipeline augmentasi SSL, fungsi *contrastive loss* (NT-Xent) akan mempelajari fitur garis buatan yang salah, bukan tekstur biologis asli lesi serviks. *(Catatan: File `.png` ini dapat dimanfaatkan nanti jika diperlukan preprocessing ekstraksi Region of Interest menggunakan OpenCV).*

### 2. Domain Adaptation pada Augmentasi Visual
Augmentasi standar SimCLR dirancang untuk citra objek umum (ImageNet) dan terbukti terlalu agresif untuk citra medis. Oleh karena itu, dilakukan modifikasi:
* **Penghapusan *Random Grayscale*:** Augmentasi *grayscale* sepenuhnya dihilangkan. Merujuk pada penelitian Hu et al. (2019) dan Azizi et al. (2021), fitur diagnostik utama lesi pra-kanker (seperti reaksi *acetowhite* dan vaskularisasi) sangat bergantung pada spektrum warna. Pengubahan ke hitam-putih akan menghapus sinyal biologis krusial ini dan menyatukan kontras antara jaringan sehat dengan lesi.
* **Pembatasan Ekstrim pada *Color Jitter (Hue)*:** Nilai rotasi warna (*hue*) ditekan ke batas minimum (`0.02`). Sesuai temuan Tellez et al. (2019), pergeseran spektrum warna yang tinggi akan merepresentasikan jaringan biologis yang mustahil ada secara alamiah (misalnya serviks berwarna hijau/biru neon). Hal ini akan menyebabkan model SimCLR mempelajari distribusi fitur *noise* yang salah.
* **Modifikasi Crop Scale untuk *Preservasi Inti Sel*:** Rentang pemotongan RandomResizedCrop dipersempit dari (0.4, 1.0) menjadi (0.7, 1.0). Pada sitologi/kolposkopi, rasio inti sel terhadap sitoplasma adalah diagnostik utama. Membatasi pemotongan di minimal 70% mencegah model salah menginterpretasikan dua area gambar yang kehilangan konteks inti selnya (mencegah false positive pairs).

## Mekanisme Pelatihan & Checkpointing (SimCLR)
Pada fase *pre-training* SimCLR yang bersifat *unsupervised*, model menggunakan mekanisme pengawasan metrik evaluasi kustom untuk memastikan model belajar dengan optimal:
* **Pembaruan Skala Learning Rate & Scheduler:** Base learning rate ditetapkan pada angka moderat 3e-4 untuk mencegah overshooting. Selain itu, jadwal penurunan Cosine Annealing dimodifikasi dari perhitungan per-epoch menjadi per-batch (step-based scheduling) dengan batas bawah learning rate (eta_min) sebesar 1e-5.
* **Penyesuaian Gradient Accumulation:** Nilai *accumulation steps* diturunkan menjadi 2 dari 4 karena ukuran dataset yang jauh lebih kecil.
* **Early Stopping & Pemantauan NT-Xent Loss:** Pelatihan memantau rata-rata *loss* setiap *epoch*. Jika *loss* tidak mengalami penurunan selama 12 *epoch* berturut-turut (*patience* = 12), *training* akan dihentikan otomatis untuk mencegah *overfitting* dan membuang waktu komputasi.
* **Stop at Best:** Sistem secara otomatis menyimpan bobot *backbone* ResNet-50 terbaik setiap kali rekor *loss* terendah tercapai (disimpan sebagai `simclr_best_epoch_X.pth`).
* **Seamless Resume:** *State* dari pelatihan disimpan secara berkala ke dalam file `latestcheck.pth`. File ini tidak hanya mengamankan bobot model dan *optimizer*, tetapi juga menyimpan nilai `best_loss` dan hitungan *early stop counter*. Hal ini memastikan bahwa jika proses *training* terputus, pelatihan dapat dilanjutkan persis dari titik terakhirnya tanpa mereset memori *Early Stopping*.