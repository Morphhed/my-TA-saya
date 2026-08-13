# Modul Evaluasi Model Klasifikasi Lesi Serviks

Repositori ini berisi skrip Python untuk melakukan evaluasi otomatis terhadap sekumpulan file bobot model checkpoint (`.pth`) hasil pelatihan (*training*) jaringan saraf tiruan berarsitektur ResNet50.

## Struktur File
* **`eval_path.py`**: Berfungsi sebagai berkas konfigurasi fleksibel untuk menentukan direktori folder eksperimen pelatihan yang ingin dipindai.
* **`evaluation.py`**: Skrip utama untuk memuat model secara berurutan, menghitung nilai *Loss* (CrossEntropy), dan mengevaluasi beragam metrik performa klasifikasi medis.

---

## Metrik Evaluasi yang Digunakan
Skrip `evaluation.py` secara otomatis menghitung metrik-metrik standar berikut pada setiap file model yang diuji:
1. **Average Loss**: Rata-rata nilai galat (*loss*) menggunakan fungsi *CrossEntropyLoss*.
2. **Accuracy**: Persentase total tebakan prediksi yang benar secara keseluruhan.
3. **Precision, Recall, & F1-Score (Weighted)**: Mengukur ketepatan dan sensitivitas model, sangat krusial untuk menangani distribusi data medis yang tidak seimbang (*imbalanced data*).
4. **Sensitivity & Specificity (Macro)**: Standar emas diagnosis medis untuk mengukur kemampuan mendeteksi lesi secara tepat serta mengenali jaringan normal/sehat.
5. **ROC-AUC (OvR, Weighted)**: Mengukur kemampuan model dalam membedakan antar kelas tingkat keparahan.
6. **Quadratic Weighted Kappa (QWK)**: Menilai tingkat kesepakatan klasifikasi ordinal (tingkat keparahan lesi) dengan memberikan penalti bobot jika kesalahan prediksi melompat terlalu jauh.
7. **Confusion Matrix**: Visualisasi matriks kebingungan yang disimpan otomatis dalam format gambar `.png` untuk setiap model.

---

## Cara Penggunaan

1. Atur target folder eksperimen dalam **`eval_path.py`**:
   ```python
   EXPERIMENT_FOLDER = "nama_folder_eksperimen_anda"
   ```

## Download Evaluasi (Git Clone)

```bash
git clone -b evaluasi https://github.com/Morphhed/my-TA-saya.git Evaluasi
```