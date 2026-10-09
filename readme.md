# Modul Evaluasi Feature Representation (SimCLR ResNet50) - Klasifikasi Lesi Pra-Kanker Serviks

Repositori ini berisi skrip Python untuk melakukan evaluasi kualitas representasi fitur (*feature representation*) secara otomatis terhadap sekumpulan file bobot checkpoint (`.pth`) hasil pelatihan *Self-Supervised Learning* (SimCLR) berarsitektur ResNet50. Evaluasi dilakukan secara murni tanpa *fine-tuning* bobot (*non-parametric linear evaluation*) menggunakan algoritma **K-Nearest Neighbors (KNN)** yang dioptimalkan secara otomatis menggunakan **GridSearchCV**.

---

## Struktur Berkas & Modul

* **`eval_path.py`**: Berkas konfigurasi terpusat untuk menentukan direktori dataset latih (`TRAIN_DATA`), dataset uji (`TEST_DATA`), berkas CSV ground truth Kaggle (`CSV_PATH`), serta folder eksperimen checkpoint model (`TARGET_FOLDER_PATH`).
* **`evaluation.py`**: Skrip utama evaluasi KNN. Bertugas mengekstrak fitur 2048-dimensi dari *backbone* ResNet50, menerapkan normalisasi L2, menjalankan pencocokan KNN (k=5), dan menghitung seluruh metrik performa medis.
* **`test_split.py`**: Skrip utilitas untuk menyusun dan mengelompokkan gambar uji Kaggle secara otomatis ke dalam struktur sub-folder kelas (`Type_1`, `Type_2`, `Type_3`) berdasarkan berkas ground truth CSV.

## Metrik Evaluasi Medis

1. **Accuracy**: Persentase total tebakan prediksi kelas lesi yang benar secara keseluruhan.
2. **Precision, Recall, & F1-Score (Weighted)**: Mengukur ketepatan dan sensitivitas diagnosis klasifikasi biner (Normal vs Abnormal) yang disesuaikan dengan proporsi distribusi data (*imbalanced dataset*).
3. **Sensitivity & Specificity (Macro)**: Standar emas diagnosis medis untuk menilai kemampuan mendeteksi lesi abnormal secara tepat (sensitivitas) dan mengenali jaringan serviks normal (spesifisitas).
4. **ROC-AUC**: Area under the ROC curve untuk mengevaluasi seberapa baik representasi fitur *backbone* dalam memisahkan probabilitas kelas Normal dan Abnormal.
5. **Confusion Matrix Heatmap**: Visualisasi matriks prediksi vs ground truth (Normal & Abnormal) yang disajikan dan disimpan otomatis sebagai berkas `.png` resolusi tinggi (300 DPI) untuk setiap *checkpoint*.

---

## Cara Penggunaan

### Menjalankan Evaluasi KNN Berantai
Jalankan skrip evaluasi utama untuk memindai dan mengevaluasi seluruh berkas `.pth` di folder target:
```bash
python evaluation.py
```

---

## Git Clone (Branch Evaluasi)

```bash
git clone -b evaluasi https://github.com/Morphhed/my-TA-saya.git Evaluasi
```