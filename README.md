# Klasifikasi Lesi Prakanker Cervix Menggunakan ResNet50 + SimCLR + CBAM (COMPLETE OVERHAUL)

Proyek ini mengembangkan klasifikasi biner citra *cervix* menggunakan **Self-Supervised Learning (SimCLR)** dengan backbone **ResNet50**, kemudian dilanjutkan dengan *fine-tuning* menggunakan **CBAM (Convolutional Block Attention Module)**.

Dataset merupakan gabungan **Malhari** dan **AnnoCerv** yang dipetakan menjadi dua kelas:

- **Normal**
- **Abnormal**

Total dataset yang digunakan adalah **666 citra**, dengan pembagian utama **80% data pengembangan** dan **20% independent test set**.

## Sumber Dataset

- **Malhari Dataset:** [Mendeley Data](https://data.mendeley.com/datasets/m5kxdj7m36/1)
- **AnnoCerv Dataset:** [GitHub](https://github.com/iclx/AnnoCerv)

## Pemetaan Kelas

### AnnoCerv

Pemetaan dilakukan berdasarkan **Swede Score**:

- **Normal:** Swede Score < 5
- **Abnormal:** Swede Score ≥ 5

Ambang Swede Score digunakan untuk memisahkan kelompok risiko rendah dan kelompok dengan indikasi lesi derajat lebih tinggi.

### Malhari

Pemetaan berdasarkan derajat CIN:

- **Normal:** CIN 1
- **Abnormal:** CIN 2 dan CIN 3

Klasifikasi menggunakan batas **CIN2+** sebagai kelompok abnormal.

> Catatan: terminologi kelas `Normal` digunakan sebagai label eksperimen biner. Dalam interpretasi klinis, CIN1 lebih tepat dipandang sebagai kelompok *low-grade/low-risk* dibanding jaringan normal sepenuhnya.

## Pembagian Dataset

Dataset utama dibagi menjadi:

```text
666 citra
│
├── 80% Development Set
│     ├── Training
│     └── Validation
│
└── 20% Independent Test Set
```

**Test set tidak digunakan untuk memilih checkpoint selama training.** Test hanya digunakan setelah model terbaik ditentukan berdasarkan validation set.

Pada tahap *fine-tuning*, development set kembali dibagi secara **stratified** menjadi training dan validation.

# Struktur/Strategi Trainingnya

## 1. Preprocessing dan Ekstraksi Patch

Pre-training SimCLR hanya menggunakan citra berformat **`.jpg`** dari development set. Setiap citra dipotong menjadi patch **256×256 piksel** menggunakan `stride=256`, sehingga patch tidak saling overlap. Patch dengan rata-rata intensitas **>240** atau **<20** dibuang untuk mengurangi area yang terlalu terang atau terlalu gelap.

Nama patch mempertahankan identitas gambar sumber dan koordinat pemotongan. Informasi ini digunakan ketika membagi patch menjadi SSL training dan validation agar patch dari gambar sumber yang sama tidak masuk ke dua subset berbeda.

## 2. Augmentasi SimCLR

Setiap patch menghasilkan dua *augmented views* secara independen. Augmentasi menggunakan `RandomResizedCrop` menjadi **224×224** dengan `scale=(0.80, 1.00)` dan `ratio=(0.90, 1.10)`, `RandomHorizontalFlip(p=0.5)`, serta `RandomRotation(±20°)`.

Variasi fotometrik menggunakan `ColorJitter` dengan **brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01** dan probabilitas **0.5**. `GaussianBlur` menggunakan **kernel=5**, `sigma=(0.1, 1.0)`, dan probabilitas **0.2**. Vertical flip, elastic transformation, rotasi 360°, dan random grayscale tidak digunakan.

Citra kemudian dinormalisasi menggunakan:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 3. Arsitektur dan Pre-Training SimCLR

Backbone menggunakan **ResNet50 tanpa bobot supervised ImageNet**. Fully Connected Layer bawaan diganti dengan `Identity`, sehingga backbone menghasilkan representasi fitur **2048 dimensi**. Projection Head menggunakan konfigurasi:

```text
2048 → 512 → 512 → 128
```

Dua layer pertama menggunakan `Linear → BatchNorm → ReLU`, sedangkan layer terakhir menghasilkan vektor proyeksi 128 dimensi. Projection Head hanya digunakan untuk menghitung contrastive loss dan dilepas setelah pre-training.

Pre-training menggunakan **NT-Xent Loss** dengan `temperature=0.3 dan 0.2`. Optimizer menggunakan **AdamW** dengan `learning_rate=3e-4` dan `weight_decay=1e-4`. Parameter bias dan parameter berdimensi satu tidak diberikan weight decay. Batch size adalah **128**, accumulation step **1**, dan training maksimum **100 epoch**.

Learning rate menggunakan **Linear Warm-up selama 10 epoch** dengan `start_factor=0.01`, kemudian dilanjutkan dengan **Cosine Annealing** hingga `eta_min=1e-5`. Training menggunakan **Automatic Mixed Precision (AMP)** dan gradient clipping `max_norm=1.0`.

## 4. SSL Validation dan Checkpoint

Patch development dibagi menjadi **90% SSL training dan 10% SSL validation berdasarkan gambar sumber**. Checkpoint terbaik ditentukan berdasarkan **validation NT-Xent loss**, bukan training loss, hasil KNN, maupun test set.

Early stopping menggunakan `patience=12`. Setiap kali validation loss mencapai nilai terbaik baru, backbone disimpan menggunakan format:

```text
simclr_best_epoch_X.pth
```

Nomor `X` menunjukkan epoch ketika validation loss terbaik tersebut diperoleh. Karena checkpoint hanya dibuat ketika validation loss membaik, checkpoint dengan nomor epoch terbesar merupakan **best checkpoint terakhir**.

Pada akhir pre-training sistem menampilkan:

```text
Best Epoch
Best Validation Loss
Best Checkpoint
```

File `latestcheck.pth` digunakan untuk menyimpan state lengkap model, optimizer, scheduler, AMP scaler, dan status early stopping agar training dapat dilanjutkan apabila terhenti.

## 5. Evaluasi Representasi SimCLR dengan KNN

KNN digunakan **hanya untuk mengevaluasi kualitas representasi hasil pre-training SimCLR** dan tidak digunakan untuk menentukan checkpoint SimCLR. Backbone checkpoint terbaik dibekukan dan setiap citra diekstraksi menjadi fitur **2048 dimensi**, kemudian dinormalisasi menggunakan **L2 normalization**.

Hyperparameter KNN dicari menggunakan **GridSearchCV dengan Stratified 5-Fold Cross-Validation** hanya pada development set. Parameter yang diuji adalah:

```text
n_neighbors : 3, 5, 7, 9, 11, 15, 21
weights     : uniform, distance
metric      : euclidean, manhattan, cosine
scoring     : balanced_accuracy
```

Setelah parameter terbaik ditemukan, KNN dilatih menggunakan seluruh fitur development set dan dievaluasi satu kali menggunakan independent test set.


## 6. Arsitektur dan Skenario Fine-Tuning

Setelah pre-training selesai, Projection Head dilepaskan dan backbone ResNet50 hasil SimCLR digunakan sebagai inisialisasi model klasifikasi. **CBAM** ditempatkan setelah `layer4` ResNet50 yang menghasilkan **2048 channels**, kemudian diikuti Global Average Pooling dan Fully Connected Layer untuk menghasilkan dua kelas, yaitu Normal dan Abnormal.

Untuk mengetahui seberapa besar perubahan bobot backbone memengaruhi representasi hasil SimCLR, dilakukan tiga skenario fine-tuning. **FT-A** mempertahankan seluruh backbone ResNet50 dalam kondisi frozen dan hanya melatih CBAM serta Fully Connected Layer. **FT-B** mempertahankan `conv1` sampai `layer3` dalam kondisi frozen dan hanya membuka `layer4`, CBAM, serta Fully Connected Layer. **FT-C** merupakan skenario full fine-tuning, yaitu seluruh backbone dibuka dengan discriminative learning rate.

Sebagai pembanding, digunakan **ResNet50_BASELINE_ImageNet_Dengan_CBAM** yang diinisialisasi menggunakan bobot supervised ImageNet. Baseline menggunakan arsitektur CBAM dan classifier yang sama, serta tetap menjalankan full fine-tuning.

## 7. Strategi Fine-Tuning

Development set dibagi kembali secara *stratified* menjadi **80% training dan 20% validation** menggunakan `seed=42`. Seluruh skenario menggunakan **batch size 32**, maksimum **50 epoch**, `CrossEntropyLoss`, optimizer **AdamW**, `weight_decay=1e-4`, serta Automatic Mixed Precision (AMP).

Pada **fase warm-up selama 5 epoch**, seluruh backbone ResNet50 dibekukan. Hanya CBAM dan Fully Connected Layer yang dilatih menggunakan `learning_rate=3e-4`. Setelah warm-up, strategi fine-tuning ditentukan melalui argumen command line saat menjalankan `finetune_train.py`. Program tidak akan memulai training apabila strategi tidak diberikan. Strategi yang tersedia adalah `frozen` untuk FT-A, `layer4` untuk FT-B, dan `full` untuk FT-C.

### FT-A — Frozen Backbone

Pada FT-A, seluruh backbone ResNet50 tetap dibekukan sampai akhir training. Hanya CBAM dan Fully Connected Layer yang diperbarui.

```text
ResNet50 Backbone = Frozen
CBAM              = Train
FC                = Train
```

Learning rate:

```text
CBAM : 1e-4
FC   : 3e-4
```

Menjalankan FT-A pada Bash:

```bash
python finetune_train.py frozen
```

### FT-B — Partial Fine-Tuning Layer4

Pada FT-B, `conv1`, `bn1`, `layer1`, `layer2`, dan `layer3` tetap dibekukan. Hanya `layer4`, CBAM, dan Fully Connected Layer yang diperbarui.

```text
conv1-layer3 = Frozen
layer4       = Train
CBAM         = Train
FC           = Train
```

Learning rate:

```text
layer4 : 1e-5
CBAM   : 1e-4
FC     : 3e-4
```

Menjalankan FT-B pada Bash:

```bash
python finetune_train.py layer4
```

### FT-C — Full Fine-Tuning

Pada FT-C, seluruh backbone ResNet50 dibuka dan dilatih menggunakan *discriminative learning rate*. Strategi ini digunakan untuk membandingkan dampak full fine-tuning terhadap representasi yang telah dipelajari SimCLR.

```text
conv1 + bn1 + layer1 + layer2 : 1e-6
layer3                        : 5e-6
layer4                        : 1e-5
CBAM                          : 1e-4
FC                            : 3e-4
```

Menjalankan FT-C pada Bash:

```bash
python finetune_train.py full
```

Secara ringkas, skenario yang digunakan adalah:

```text
FT-A : SimCLR backbone frozen + CBAM + FC
FT-B : SimCLR layer4 + CBAM + FC
FT-C : SimCLR full fine-tuning
```

Checkpoint terbaik pada setiap skenario dipilih berdasarkan **validation ROC-AUC**. Jika ROC-AUC tidak dapat dihitung, validation accuracy digunakan sebagai fallback. Independent test set hanya digunakan setelah checkpoint terbaik selesai ditentukan.

## 8. Evaluasi Model

Evaluasi akhir menggunakan **Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, Sensitivity, Specificity, ROC-AUC, Classification Report**, dan **Confusion Matrix**.

Kelas **Abnormal** didefinisikan sebagai positive class dan **Normal** sebagai negative class:

```text
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

Mapping kelas mengikuti `ImageFolder.class_to_idx` dan tidak diasumsikan secara manual sehingga probabilitas ROC-AUC selalu diambil dari indeks kelas Abnormal yang sebenarnya.

## 9. Grad-CAM

Grad-CAM digunakan setelah fine-tuning untuk menganalisis area citra yang berkontribusi terhadap keputusan model. Grad-CAM tidak diterapkan pada KNN maupun backbone SimCLR mentah karena keduanya tidak memiliki class score dari classifier neural network.

Target Grad-CAM adalah **blok terakhir `layer4` ResNet50**. Gradien kelas prediksi digunakan untuk memberikan bobot pada feature map, kemudian hasilnya diterapkan ReLU, dinormalisasi ke rentang 0–1, diperbesar menjadi ukuran input, dan ditumpangkan pada citra asli.

Secara default dipilih **5 citra Normal dan 5 citra Abnormal** dari independent test set. Setiap hasil menampilkan citra asli, heatmap Grad-CAM, overlay, true label, predicted label, dan confidence score. Grad-CAM digunakan sebagai analisis kualitatif, bukan sebagai metrik performa klasifikasi.

## 10. Urutan Eksperimen

Eksperimen dijalankan secara berurutan sebagai berikut:

```bash
python preprocess.py
python train.py
python evaluation.py

```
"untuk fine tuning sendiri ada 3 strategi, refer back ke bagian 7. sebelum running gradcam"

```bash
python gradcam.py
```

`preprocess.py` menghasilkan patch untuk pre-training SimCLR. `train.py` melakukan pre-training dan menentukan checkpoint terbaik berdasarkan validation NT-Xent loss. `evaluation.py` mengevaluasi kualitas frozen representation menggunakan KNN dan GridSearchCV. `finetune_train.py` kemudian menjalankan skenario **FT-A**, **FT-B**, dan baseline ImageNet, sedangkan hasil **FT-C** digunakan sebagai pembanding full fine-tuning SimCLR. Setiap skenario menggunakan validation ROC-AUC untuk menentukan checkpoint klasifikasi terbaik. Setelah evaluasi akhir pada independent test set, `gradcam.py` digunakan untuk menghasilkan visualisasi interpretabilitas dari model fine-tuning yang dipilih.

## Instalasi & Repository

Clone repository:

```bash
git clone -b Malhari-Anno https://github.com/Morphhed/my-TA-saya.git Malhari-Anno
```