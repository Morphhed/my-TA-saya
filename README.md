# Dokumentasi Dataset: Malhari & AnnoCerv

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

## Penyesuaian Preprocessing untuk SimCLR
Untuk memastikan dataset kompatibel dengan *pipeline* PyTorch dan *Self-Supervised Learning* (SimCLR), dilakukan dua penyesuaian:
* **Perataan Direktori (Flat Directory):** Mengeluarkan gambar dari dalam folder masing-masing pasien sehingga struktur akhirnya langsung merujuk pada kelas (contoh: `train/Normal/image.jpg`).
* **Hanya Menggunakan Citra `.jpg`:** File anotasi `.png` (masking tepi) tidak diikutsertakan ke dalam *DataLoader*. Jika file `.png` ikut masuk ke dalam pipeline augmentasi SSL, fungsi *contrastive loss* (NT-Xent) akan mempelajari fitur garis buatan yang salah, bukan tekstur biologis asli lesi serviks. *(Catatan: File `.png` ini dapat dimanfaatkan nanti jika diperlukan preprocessing ekstraksi Region of Interest menggunakan OpenCV).*