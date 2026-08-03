# Ringkasan Perubahan / Tuning Pertama

Berikut adalah ringkasan perubahan teknis dan peningkatan yang dilakukan pada skrip :

*   **Peningkatan Kapasitas Batch Size (`config.py`):**
    *   Mengubah nilai `BATCH_SIZE` dari `64` menjadi `128`. Peningkatan ini memanfaatkan kapasitas VRAM secara optimal untuk memperbanyak jumlah sampel negatif dalam satu iterasi, sehingga memperkuat representasi kontrastif model.
*   **Penyesuaian Parameter Suhu / Temperature (`config.py` & `train.py`):**
    *   Menambahkan dan menurunkan parameter `TEMPERATURE` dari `0.5` menjadi `0.1`. Suhu yang lebih rendah memberikan penalti (*harsher penalty*) yang lebih ketat terhadap sampel negatif, memaksa model agar jauh lebih sensitif terhadap perbedaan tekstur dan detail halus pada jaringan serviks.
*   **Penerapan Overlapping Patches / Stride (`preprocess.py`):**
    *   Memodifikasi fungsi ekstraksi *patch* dari yang sebelumnya menggunakan pemotongan grid ketat berbasis `patch_size` menjadi penerapan *stride* selektif (`patch_size // 2` atau tumpang tindih 50%). Perubahan ini melipatgandakan volume dataset *patch* secara instan dan memperkaya konteks spasial antar-wilayah citra.
*   **Penyesuaian Augmentasi Khusus Medis (`dataset.py`):**
    *   **Penghapusan Grayscale:** Menghapus `transforms.RandomGrayscale` secara penuh karena warna dan kemerahan (*biomarker*) merupakan indikator esensial dalam mendeteksi lesi serviks.
    *   **Penambahan Rotasi Bebas:** Menyisipkan `transforms.RandomRotation(degrees=360)` untuk mengajari model bahwa jaringan biologis tidak memiliki orientasi mutlak atas-bawah.
    *   **Penambahan Deformasi Elastis:** Menyisipkan `transforms.ElasticTransform(alpha=50.0, sigma=5.0)` untuk memetakan sifat fleksibel/elastis dari jaringan organ seluler.
    *   **Penyetelan Jitter & Crop:** Membatasi kekuatan *ColorJitter* agar tidak merusak rona warna asli jaringan, serta menaikkan batas bawah *RandomResizedCrop* dari `0.2` ke `0.4` agar model tidak terlalu fokus pada area kosong/mikro.
*   **Penggantian Optimizer ke AdamW (`train.py`):**
    *   Mengganti `optim.Adam` dengan `optim.AdamW` disertai peningkatan nilai *weight decay* menjadi `1e-4`. Algoritma *decoupled weight decay* pada AdamW memberikan stabilitas dan regularisasi bobot yang jauh lebih baik untuk proses *Self-Supervised Learning* jangka panjang.

## Donlod Versi Tuning (Git Clone)

```bash
git clone -b tuning-pertama [https://github.com/Morphhed/my-TA-saya.git](https://github.com/Morphhed/my-TA-saya.git) tuning-pertama
```

