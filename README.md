# Ringkasan Perubahan (Mengikuti Base dari 'Tuning-Pertama')

## 1. `train.py` (Pelatihan & Optimasi)
*   **Persistensi Pekerja Dataloader (*Persistent Workers*) (`dataset.py` / `train.py`):**
    *   Menambahkan parameter `persistent_workers=True` pada konfigurasi `DataLoader` PyTorch. Optimasi ini krusial untuk mencegah terjadinya *MemoryError* (lonjakan memori/RAM yang ekstrem) di sistem operasi Windows saat pergantian *epoch*. Daripada menghancurkan dan menciptakan ulang (*spawn*) pekerja yang memicu penyalinan ulang jutaan *path* file ke memori secara serentak, parameter ini menahan proses *worker* agar tetap hidup. Hasilnya, konsumsi RAM menjadi jauh lebih stabil (*anti-spike*) dan jeda waktu transisi antar *epoch* menjadi instan tanpa memengaruhi logika pengacakan augmentasi data.
*   **Pengetatan Regularisasi (*Weight Decay*):**
    *   Menaikkan parameter `weight_decay` pada optimizer AdamW dari `1e-4` menjadi `1e-3`. Hal ini bertujuan untuk memberikan hukuman (*penalty*) yang lebih berat pada bobot model agar tidak mudah beradaptasi pada *noise* (*overfitting*) dan mencegah *shortcut learning* saat memproses volume *patch* yang ekstrem.
*   **Implementasi *Gradient Accumulation*:**
    *   Menambahkan mekanisme `ACCUMULATION_STEPS = 4` untuk menstabilkan arah pembaruan bobot (*gradient direction*). Model menumpuk gradien selama 4 iterasi *batch* (128 gambar) sebelum mengeksekusi langkah *optimizer*. Teknik ini menyimulasikan efek kestabilan *batch size* raksasa tanpa membuat VRAM GPU *Out of Memory* (OOM).

## 2. `config.py` (HyperParameter)
*   **Pelunakan Suhu (*Temperature*):**
    *   Mengembalikan nilai `TEMPERATURE` dari `0.1` menjadi `0.5`. Suhu `0.1` terbukti terlalu tajam untuk dataset berskala jutaan *patch*, menyebabkan *loss* terjun bebas terlalu cepat menuju *Representation Collapse*. Suhu `0.5` membuat penalti lebih lunak, sehingga kurva penurunan *loss* bergerak lebih bertahap dan model mempelajari fitur sel secara matang.
*   **Pengendalian Laju Pembelajaran (*Learning Rate*):**
    *   Menurunkan `LEARNING_RATE` dari `1e-3` menjadi `5e-4`. Penyesuaian ini mencegah *optimizer* mengambil langkah tebakan yang terlalu lebar, menjaga stabilitas dinamika pelatihan pada iterasi yang sangat panjang.

## 3. `dataset.py` (Augmentasi Data)
*   **Pencegahan *Shortcut* Warna (*Grayscale*):**
    *   Mengembalikan augmentasi `transforms.RandomGrayscale` dengan probabilitas rendah (`p=0.2`). Penghapusan total gambar hitam-putih sebelumnya berisiko membuat model berbuat curang dengan sekadar mencocokkan intensitas warna pewarna klinis/kamera antar *patch*. Probabilitas 20% ini memaksa model untuk sesekali murni menganalisis bentuk (*shape*) dan tekstur jaringan sel serviks tanpa bergantung pada rona warna.

---
## Donlod Versi Tuning (Git Clone)

```bash
git clone -b tuning-kedua https://github.com/Morphhed/my-TA-saya.git tuning-kedua
```

## Updated Library (Instalasi TensorBoard)
Pastikan pustaka TensorBoard telah terpasang di lingkungan Python / conda Anda untuk memantau grafik metrik *training*:

```bash
python -m pip install tensorboard
```