# Ringkasan Perubahan (Mengikuti Base dari 'Tuning-Pertama')

## 1. `train.py` (Pelatihan & Optimasi)
*   **Persistensi Pekerja Dataloader (*Persistent Workers*) (`dataset.py` / `train.py`):**
    *   Menambahkan parameter `persistent_workers=True` pada konfigurasi `DataLoader` PyTorch. Optimasi ini krusial untuk mencegah terjadinya *MemoryError* (lonjakan memori/RAM yang ekstrem) di sistem operasi Windows saat pergantian *epoch*. Daripada menghancurkan dan menciptakan ulang (*spawn*) pekerja yang memicu penyalinan ulang jutaan *path* file ke memori secara serentak, parameter ini menahan proses *worker* agar tetap hidup. Hasilnya, konsumsi RAM menjadi jauh lebih stabil (*anti-spike*) dan jeda waktu transisi antar *epoch* menjadi instan tanpa memengaruhi logika pengacakan augmentasi data.
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