# Dokumentasi Dataset: Malhari & AnnoCerv (Base dari "Tuning-Kedua")

Proyek klasifikasi kanker serviks ini (menggunakan ResNet-50 + CBAM + SimCLR) menggunakan gabungan dataset klinis/kolposkopi dari dua sumber publik untuk menghasilkan klasifikasi biner (**Normal** vs **Abnormal**). yang dari sebelumnya menggunakan dataset Intel

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

## Preprocessing & Augmentasi SimCLR

### 1. Struktur Data & Format
- **Flat Directory:** setelah pemetaan (normal vs abnormal) File gambar disusun langsung berdasarkan folder kelas (misal: `train/Normal/image.jpg`).
- **Filter `.jpg` Murni (AnnoCerv):** Mengabaikan file anotasi `.png` agar fungsi *NT-Xent Loss* tidak mempelajari fitur garis *masking* buatan.

### 2. Augmentasi Khusus Citra Medis
- **Tanpa Grayscale:** Mempertahankan warna asli untuk preservasi fitur vaskularisasi dan reaksi *acetowhite*.
- **Hue Jitter Dibatasi (0.02):** Mencegah pergeseran warna yang tidak alamiah pada jaringan biologis.
- **Crop Scale (0.7 - 1.0):** Menjaga integritas rasio inti sel terhadap sitoplasma agar tidak kehilangan konteks diagnostik.

## Mekanisme Pelatihan & Checkpointing

- **Hyperparameter:** Learning rate `3e-4`, *step-based* Cosine Annealing (`eta_min = 1e-5`), dan *gradient accumulation steps* = 2.
- **Early Stopping:** Pelatihan dihentikan otomatis jika *loss* tidak turun selama 12 *epoch* berturut-turut (*patience* = 12).
- **Mekanisme Checkpoint:**
  - `simclr_best_epoch_X.pth`: Menyimpan bobot *backbone* ResNet-50 dengan *loss* terendah.
  - `latestcheck.pth`: Menyimpan *state* lengkap (model, *optimizer*, `best_loss`, & *early stop counter*) untuk fitur *seamless resume*.


# Finetune and Comparisons
Pada tahap ini, *Projection Head* dari SimCLR dibuang. Lapisan atensi **CBAM** disuntikkan tepat setelah *layer* konvolusi terakhir (`layer4`, 2048 *channels*) pada ResNet-50.

**Eksperimen Komparasi dilakukan pada dua model:**
1. `ResNet50_USULAN_Dengan_SimCLR_dan_CBAM`: Menggunakan inisialisasi bobot *pre-trained* hasil SimCLR.
2. `ResNet50_BASELINE_Tanpa_SimCLR_Dengan_CBAM`: Menggunakan inisialisasi bobot standar (ImageNet).

### Strategi Pelatihan (2-Phase Fine-Tuning)
Guna mencegah lonjakan gradien merusak bobot representasi fitur SimCLR yang sudah matang, pelatihan dilakukan dalam dua tahap adaptasi:
* **Fase 1 (Warm-Up):** Seluruh *backbone* ResNet-50 dikunci (*frozen*). Model hanya melatih modul CBAM dan *Fully Connected Layer* yang bobot awalnya masih acak selama 2 *epoch*.
* **Fase 2 (Full Fine-Tuning):** Seluruh jaringan dilepas kuncinya (*unfrozen*). Model dilatih secara *end-to-end* dengan *Learning Rate* yang diturunkan secara drastis (hanya 10% dari *base learning rate*) agar dapat beradaptasi halus pada fitur klasifikasi serviks medis.


## Instalasi & Repository

Gunakan perintah berikut untuk melakukan *clone* langsung pada *branch* `Malhari-Anno`:

```bash
git clone -b Malhari-Anno https://github.com/Morphhed/my-TA-saya.git Malhari-Anno
```