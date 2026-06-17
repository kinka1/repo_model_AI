# BAB 4 EKSPERIMEN DAN ANALISIS

Bab ini menjelaskan pelaksanaan eksperimen dan analisis hasil pengujian sistem deteksi dan klasifikasi bakteri Gram. Pembahasan meliputi parameter eksperimen, karakteristik data, waktu uji coba, spesifikasi peralatan, hasil setiap skenario pelatihan, serta analisis performa model berdasarkan metrik evaluasi. Struktur bab ini mengikuti pola penyajian pada dokumen referensi `docs/main.pdf`, yaitu memisahkan bagian parameter, data, hasil eksperimen, dan analisis secara sistematis.

## 4.1 Parameter Eksperimen

Eksperimen dilakukan untuk mengevaluasi kemampuan model dalam mengklasifikasikan citra bakteri hasil pewarnaan Gram ke dalam dua kelas, yaitu Gram-positif dan Gram-negatif. Parameter eksperimen disusun agar setiap skenario dapat dibandingkan secara adil berdasarkan akurasi, precision, recall, F1-score, confusion matrix, ROC-AUC jika tersedia, serta pola training loss dan validation loss.

### 4.1.1 Konfigurasi Dataset

Dataset yang digunakan adalah dataset citra mikroskopis bakteri hasil pewarnaan Gram. Citra dibagi ke dalam dua kelas utama, yaitu Gram-positif dan Gram-negatif. Pada eksperimen awal, dataset memiliki ketidakseimbangan kelas dengan jumlah Gram-positif lebih besar daripada Gram-negatif. Kondisi ini perlu diperhatikan karena dapat menyebabkan model lebih condong memprediksi kelas mayoritas.

Tabel 4.1 menunjukkan konfigurasi umum dataset yang digunakan pada eksperimen.

| Parameter | Nilai |
|---|---|
| Jenis data | Citra mikroskopis hasil pewarnaan Gram |
| Tugas | Klasifikasi biner |
| Kelas | Gram-positif dan Gram-negatif |
| Ukuran input CNN | 224 x 224 piksel |
| Format citra | RGB |
| Pembagian data | Train, validation, dan test set |
| Metode pembagian | Stratified split |

### 4.1.2 Konfigurasi Model

Model yang diuji terdiri atas Simple CNN, ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121. Simple CNN digunakan sebagai baseline, sedangkan arsitektur lain digunakan untuk mengevaluasi pengaruh arsitektur CNN yang lebih dalam serta transfer learning dari bobot ImageNet.

Tabel 4.2 menunjukkan konfigurasi umum model.

| Komponen | Konfigurasi |
|---|---|
| Framework training | PyTorch |
| Model deteksi | YOLO11 |
| Model klasifikasi | CNN classifier |
| Pretrained weights | ImageNet untuk skenario transfer learning |
| Loss function | CrossEntropyLoss atau weighted CrossEntropyLoss |
| Optimizer | AdamW, Adam, atau SGD sesuai arsitektur |
| Scheduler | ReduceLROnPlateau jika tersedia pada script pelatihan |
| Checkpoint | Model terbaik berdasarkan performa validation set |
| Output | Label kelas dan confidence score |

### 4.1.3 Penanganan Data Imbalance

Ketidakseimbangan data ditangani menggunakan dua pendekatan utama. Pendekatan pertama adalah penggunaan class weights pada loss function agar kesalahan pada kelas minoritas mendapatkan penalti lebih besar. Pendekatan kedua adalah balancing data training melalui augmentasi atau pemilihan sampel tertentu, sehingga jumlah citra Gram-positif dan Gram-negatif pada data latih menjadi lebih seimbang.

Pada eksperimen ResNet yang menggunakan script TA, data training aktual setelah balancing berisi 6.952 citra Gram-negatif dan 6.952 citra Gram-positif. Namun, class weight dihitung berdasarkan distribusi data asli sebelum balancing, yaitu 6.952 Gram-negatif dan 3.748 Gram-positif. Dengan konfigurasi tersebut, class weight yang digunakan adalah 0,7696 untuk Gram-negatif dan 1,4274 untuk Gram-positif.

## 4.2 Karakteristik Data

Data yang digunakan pada penelitian ini berupa citra mikroskopis hasil pewarnaan Gram. Citra Gram-positif umumnya berwarna ungu atau biru karena mempertahankan kristal violet, sedangkan citra Gram-negatif cenderung berwarna merah muda karena menyerap safranin setelah proses dekolorisasi.

Karakteristik data yang memengaruhi proses pelatihan model antara lain variasi pencahayaan, tingkat fokus, ketajaman citra, intensitas pewarnaan, bentuk bakteri, ukuran objek, dan keberadaan artefak mikroskopis. Variasi tersebut menyebabkan model perlu mempelajari fitur yang tidak hanya bergantung pada warna, tetapi juga tekstur, bentuk, dan konteks visual di sekitar objek.

Tabel 4.3 menunjukkan karakteristik utama data penelitian.

| Karakteristik | Keterangan |
|---|---|
| Sumber data | Citra mikroskopis hasil pewarnaan Gram |
| Kelas | Gram-positif dan Gram-negatif |
| Morfologi | Kokus, basil, dan variasi bentuk lain sesuai data |
| Warna dominan | Ungu/biru untuk Gram-positif, merah muda untuk Gram-negatif |
| Tantangan | Variasi fokus, pencahayaan, stain, artefak, dan latar belakang |
| Preprocessing | Resize, normalisasi, crop area bakteri, dan konversi tensor |

## 4.3 Waktu Uji Coba

Uji coba dilakukan secara bertahap mulai dari persiapan data, pelatihan model baseline, eksperimen augmentasi, eksperimen transfer learning, fine-tuning, hingga evaluasi model terbaik. Eksperimen dilakukan secara lokal dan/atau menggunakan lingkungan komputasi GPU sesuai kebutuhan arsitektur model.

Tabel 4.4 menunjukkan pembagian waktu uji coba secara umum.

| Tahap | Kegiatan |
|---|---|
| Tahap 1 | Persiapan dataset, preprocessing, dan pembagian data |
| Tahap 2 | Pelatihan Simple CNN sebagai baseline |
| Tahap 3 | Pelatihan Simple CNN dengan augmentasi |
| Tahap 4 | Pelatihan arsitektur CNN from scratch |
| Tahap 5 | Transfer learning dan fine-tuning |
| Tahap 6 | Evaluasi model, pembuatan confusion matrix, dan analisis hasil |

## 4.4 Spesifikasi Peralatan Uji Coba

Eksperimen membutuhkan perangkat keras dan perangkat lunak yang mendukung pelatihan model deep learning, penyimpanan data, dan integrasi sistem aplikasi.

### 4.4.1 Perangkat Keras

Tabel 4.5 menunjukkan spesifikasi perangkat keras yang digunakan pada pengembangan dan uji coba.

| Komponen | Spesifikasi |
|---|---|
| Laptop | Lenovo IdeaPad Gaming 3 15ACH6 |
| CPU | AMD Ryzen 7 5800H |
| RAM | 24 GB |
| Penyimpanan | SSD 512 GB |
| GPU | NVIDIA GeForce RTX 3050 |
| Sistem operasi | Windows 11 Home |

### 4.4.2 Perangkat Lunak

Tabel 4.6 menunjukkan perangkat lunak yang digunakan.

| Komponen | Perangkat Lunak |
|---|---|
| Bahasa pemrograman | Python |
| Framework model | PyTorch |
| Deteksi objek | YOLO11 |
| Backend | FastAPI |
| Database | PostgreSQL |
| Frontend | HTML/CSS/JS statis |
| Editor | Visual Studio Code |
| Evaluasi | scikit-learn, matplotlib, dan library pendukung lain |

## 4.5 Hasil Eksperimen

Bagian ini menyajikan hasil pelatihan dan evaluasi model berdasarkan skenario eksperimen yang telah dirancang pada BAB 3. Eksperimen disusun dari model baseline sederhana hingga model transfer learning dan fine-tuning.

### 4.5.1 Pembagian Data

Dataset dibagi menjadi train set, validation set, dan test set. Train set digunakan untuk melatih model, validation set digunakan untuk memilih checkpoint terbaik, dan test set digunakan untuk evaluasi akhir. Pada eksperimen ResNet TA, data training aktual setelah balancing terdiri atas 13.904 citra, yaitu 6.952 citra Gram-negatif dan 6.952 citra Gram-positif.

Tabel 4.7 menunjukkan distribusi data pada eksperimen ResNet TA.

| Kelas | Training Aktual (setelah balancing) | Training Asli (sebelum balancing) | Validasi |
|---|---:|---:|---:|
| Gram-negatif | 6.952 | 6.952 | 1.739 |
| Gram-positif | 6.952 | 3.748 | 1.739 |
| Total | 13.904 | 10.700 | 3.478 |

### 4.5.2 Preprocessing Data

Preprocessing dilakukan untuk menyamakan format input citra sebelum masuk ke model. Tahapan preprocessing meliputi resize ke 224 x 224 piksel, normalisasi menggunakan mean dan standard deviation ImageNet, serta konversi citra menjadi tensor PyTorch.

Jika sistem menggunakan mode deteksi otomatis, YOLO11 digunakan terlebih dahulu untuk menemukan area bakteri. Area tersebut kemudian dipotong dan digunakan sebagai input CNN classifier. Jika mode crop otomatis tidak digunakan, citra yang telah diproses langsung masuk ke model CNN.

### 4.5.3 Augmentasi Data

Augmentasi data digunakan untuk meningkatkan variasi citra latih dan membantu model belajar fitur yang lebih stabil. Transformasi yang digunakan meliputi rotasi, flipping, perubahan brightness, contrast, saturation, serta resize atau crop sesuai kebutuhan script pelatihan.

Tabel 4.8 menunjukkan teknik augmentasi yang digunakan.

| Teknik Augmentasi | Tujuan |
|---|---|
| Random rotation | Membuat model tahan terhadap perubahan orientasi bakteri |
| Horizontal flip | Menambah variasi posisi objek |
| Color jitter | Meniru variasi pencahayaan dan intensitas pewarnaan |
| Resize/crop | Menyesuaikan ukuran citra dengan input model |

### 4.5.4 Skenario Eksperimen

Eksperimen dibagi menjadi lima skenario utama. Setiap skenario dirancang untuk melihat pengaruh komponen tertentu terhadap performa model, seperti augmentasi data, arsitektur CNN, transfer learning, dan fine-tuning.

Tabel 4.9 menunjukkan daftar skenario eksperimen.

| Skenario | Arsitektur | Augmentasi | Transfer Learning | Fine-tuning | Epoch Maksimum |
|---|---|---|---|---|---:|
| 1 | Simple CNN | Tidak | Tidak | Tidak | 40 |
| 2 | Simple CNN | Ya | Tidak | Tidak | 40 |
| 3 | ResNet, EfficientNet, VGG, DenseNet | Tidak | Tidak | Tidak | 50 |
| 4 | ResNet dan EfficientNet | Tidak | Ya | Ya | 80 |
| 5 | ResNet, EfficientNet, VGG, DenseNet | Ya | Ya | Ya | 40 sampai 80 |

### 4.5.5 Skenario 1: Baseline Simple CNN

Skenario pertama menggunakan Simple CNN yang dibangun dari awal tanpa bobot pretrained. Model ini digunakan sebagai baseline untuk melihat kemampuan awal sistem dalam mengklasifikasikan citra bakteri Gram.

Tabel 4.10 menunjukkan hasil uji Simple CNN pada 10 citra uji.

| No. | Data Uji | Akurasi |
|---:|---|---:|
| 1 | Image 1 | 0,633 |
| 2 | Image 2 | 0,712 |
| 3 | Image 3 | 0,658 |
| 4 | Image 4 | 0,701 |
| 5 | Image 5 | 0,786 |
| 6 | Image 6 | 0,754 |
| 7 | Image 7 | 0,679 |
| 8 | Image 8 | 0,685 |
| 9 | Image 9 | 0,697 |
| 10 | Image 10 | 0,716 |

Hasil tersebut menunjukkan bahwa Simple CNN masih memiliki performa yang bervariasi antar citra uji. Akurasi tertinggi diperoleh pada Image 5 sebesar 0,786, sedangkan akurasi terendah diperoleh pada Image 1 sebesar 0,633. Rentang akurasi yang cukup besar menunjukkan bahwa model baseline masih sensitif terhadap variasi kualitas citra dan belum memiliki representasi fitur yang cukup kuat.

### 4.5.6 Skenario 2: CNN dengan Data Augmentation

Skenario kedua menggunakan arsitektur Simple CNN yang sama, tetapi dengan penambahan augmentasi data. Tujuannya adalah meningkatkan kemampuan generalisasi model terhadap variasi orientasi, pencahayaan, dan posisi objek.

Tabel 4.11 menunjukkan perbandingan hasil Simple CNN tanpa augmentasi dan dengan augmentasi.

| No. | Data Uji | Akurasi dengan Augmentasi | Akurasi tanpa Augmentasi | Perubahan |
|---:|---|---:|---:|---:|
| 1 | Image 1 | 0,759 | 0,633 | +12,6% |
| 2 | Image 2 | 0,813 | 0,712 | +10,1% |
| 3 | Image 3 | 0,782 | 0,658 | +12,4% |
| 4 | Image 4 | 0,752 | 0,701 | +5,1% |
| 5 | Image 5 | 0,803 | 0,786 | +1,7% |
| 6 | Image 6 | 0,741 | 0,754 | -1,3% |
| 7 | Image 7 | 0,713 | 0,679 | +3,4% |
| 8 | Image 8 | 0,726 | 0,685 | +4,1% |
| 9 | Image 9 | 0,743 | 0,697 | +4,6% |
| 10 | Image 10 | 0,764 | 0,716 | +4,8% |

Augmentasi data meningkatkan performa pada sebagian besar citra uji. Peningkatan terbesar terjadi pada Image 1 dan Image 3. Hal ini menunjukkan bahwa augmentasi membantu model mempelajari fitur yang lebih tahan terhadap variasi visual.

### 4.5.7 Skenario 3: Training Arsitektur CNN from Scratch

Skenario ketiga melatih beberapa arsitektur CNN dari awal tanpa bobot pretrained. Tujuan skenario ini adalah membandingkan kemampuan arsitektur model ketika seluruh representasi fitur dipelajari langsung dari dataset penelitian.

Tabel 4.12 menunjukkan hasil evaluasi model CNN yang dilatih dari awal.

| No. | Arsitektur Model | Akurasi |
|---:|---|---:|
| 1 | ResNet50 | 0,921273 |
| 2 | ResNet101 | 0,926173 |
| 3 | EfficientNet-B0 | 0,838358 |
| 4 | EfficientNet-B3 | 0,839127 |
| 5 | VGG-16 | 0,808688 |
| 6 | VGG-19 | 0,825571 |
| 7 | DenseNet121 | 0,782712 |

ResNet101 memperoleh akurasi tertinggi pada skenario from scratch, yaitu 0,926173. Hasil ini menunjukkan bahwa residual connection pada ResNet membantu proses optimasi ketika model dilatih tanpa bobot pretrained.

### 4.5.8 Skenario 4: Transfer Learning dan Fine-Tuning

Skenario keempat menggunakan transfer learning dan fine-tuning pada arsitektur ResNet dan EfficientNet. Model menggunakan bobot pretrained ImageNet, kemudian classifier head diganti dan beberapa layer akhir dibuka pada fase fine-tuning.

Tabel 4.13 menunjukkan hasil evaluasi Skenario 4.

| No. | Arsitektur Model | Akurasi |
|---:|---|---:|
| 1 | ResNet50 | 0,941567 |
| 2 | ResNet101 | 0,918219 |
| 3 | EfficientNet-B0 | 0,868261 |
| 4 | EfficientNet-B3 | 0,876251 |
| 5 | VGG-16 | 0,789251 |
| 6 | VGG-19 | 0,852618 |
| 7 | DenseNet121 | 0,797219 |

ResNet50 menjadi model terbaik pada skenario ini dengan akurasi 0,941567. Hal ini menunjukkan bahwa transfer learning dan fine-tuning membantu model memanfaatkan representasi visual pretrained secara lebih efektif dibanding pelatihan dari awal.

### 4.5.9 Skenario 5: Transfer Learning dan Fine-Tuning Arsitektur CNN

Skenario kelima merupakan eksperimen utama yang membandingkan beberapa arsitektur CNN menggunakan transfer learning, augmentasi data, dan fine-tuning. Arsitektur yang diuji meliputi ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121.

#### 4.5.9.1 Konfigurasi Umum Pelatihan Skenario 5

Konfigurasi umum Skenario 5 ditunjukkan pada Tabel 4.14.

| Komponen | Konfigurasi Umum |
|---|---|
| Framework | PyTorch |
| Input citra | RGB 224 x 224 piksel |
| Pretrained weights | ImageNet |
| Jumlah kelas output | 2 kelas: Gram-negatif dan Gram-positif |
| Loss function | Weighted CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau jika tersedia |
| Checkpoint | Model terbaik berdasarkan validation set |
| Metrik evaluasi | Accuracy, precision, recall, F1-score, ROC-AUC, dan confusion matrix |

#### 4.5.9.2 Perbedaan Hyperparameter per Arsitektur

Setiap arsitektur memiliki kebutuhan komputasi dan strategi fine-tuning yang berbeda. Perbedaan hyperparameter ditunjukkan pada Tabel 4.15.

| Arsitektur | Optimizer | Batch Size | Epoch Maksimum | Learning Rate | Strategi Fine-tuning |
|---|---|---:|---:|---|---|
| ResNet50 | AdamW | 32 | 39 (best: 31) | 0,001 lalu 0,0001 | Melatih classifier head 10 epoch, lalu membuka `layer4` dan classifier (fine-tuning) |
| ResNet101 | AdamW | 32 | 40 (best: 37) | 0,001 lalu 0,0001 | Melatih classifier head 10 epoch, lalu membuka blok residual terakhir dan classifier (fine-tuning) |
| EfficientNet-B0 | AdamW | 32 | 60 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka blok akhir EfficientNet dan classifier |
| EfficientNet-B3 | AdamW | 16 | 60 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka beberapa blok akhir EfficientNet dan classifier |
| VGG-16 | SGD momentum 0,9 | 8 | 80 | 0,00005 | Fine-tuning full model dengan classifier custom |
| VGG-19 | SGD momentum 0,9 | 8 | 80 | 0,00005 | Fine-tuning full model dengan classifier custom |
| DenseNet121 | AdamW | 16 | 48 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka dense block terakhir dan classifier |

#### 4.5.9.3 Hasil Pelatihan ResNet50

Model ResNet50 menggunakan data riil dari eksperimen `ta_resnet50_20260615_021715`. Strategi pelatihan yang digunakan adalah ImageNet pretrained backbone dengan pelatihan classifier head selama 10 epoch, dilanjutkan fine-tuning dengan membuka `layer4` dan classifier. Pada epoch terbaik ke-31 (fase fine-tuning), ResNet50 memperoleh accuracy 0,951696 dan macro F1-score 0,951696.

Tabel 4.16 menunjukkan metrik detail ResNet50 pada epoch terbaik.

| Metrik | Nilai |
|---|---:|
| Best epoch | 31 |
| Phase | finetune |
| Accuracy | 0,951696 |
| Macro F1-score | 0,951696 |
| F1 Gram-positif | 0,951862 |
| F1 Gram-negatif | 0,951529 |
| Precision Gram-positif | 0,948601 |
| Recall Gram-positif | 0,955147 |
| Precision Gram-negatif | 0,954835 |
| Recall Gram-negatif | 0,948246 |
| Train loss | 0,105381 |
| Validation loss | 0,146586 |
| ROC-AUC | 0,988272 |
| TN | 1.649 |
| FP | 90 |
| FN | 78 |
| TP | 1.661 |
| Ukuran model | 93,99 MB |
| Total parameter | 24.558.146 |
| Waktu inferensi rata-rata | 2,94 ms/gambar |

Hasil tersebut menunjukkan bahwa ResNet50 memiliki performa sangat baik dan relatif seimbang pada kedua kelas. Nilai recall Gram-positif (0,955147) lebih tinggi dari precision Gram-positif (0,948601), yang berarti model lebih sensitif dalam mendeteksi sampel Gram-positif namun sedikit lebih banyak menghasilkan FP. Nilai FP (90) lebih tinggi dari FN (78), yang berarti model lebih sering salah mengklasifikasikan Gram-negatif sebagai Gram-positif dibanding sebaliknya. ROC-AUC sebesar 0,988272 menunjukkan kemampuan pemisahan kelas yang sangat baik.

#### 4.5.9.4 Hasil Pelatihan ResNet101

Model ResNet101 menggunakan data riil dari eksperimen `ta_resnet101_20260615_044118`. Strategi pelatihan yang digunakan adalah ImageNet pretrained backbone dengan pelatihan classifier head selama 10 epoch, dilanjutkan fine-tuning dengan membuka blok residual terakhir dan classifier. Pada epoch terbaik ke-37 (fase fine-tuning), ResNet101 memperoleh accuracy 0,944508 dan macro F1-score 0,944502.

Tabel 4.17 menunjukkan metrik detail ResNet101 pada epoch terbaik.

| Metrik | Nilai |
|---|---:|
| Best epoch | 37 |
| Phase | finetune |
| Accuracy | 0,944508 |
| Macro F1-score | 0,944502 |
| F1 Gram-positif | 0,945092 |
| F1 Gram-negatif | 0,943912 |
| Precision Gram-positif | 0,935248 |
| Recall Gram-positif | 0,955147 |
| Precision Gram-negatif | 0,954172 |
| Recall Gram-negatif | 0,933870 |
| Train loss | 0,083178 |
| Validation loss | 0,163975 |
| ROC-AUC | 0,988207 |
| TN | 1.624 |
| FP | 115 |
| FN | 78 |
| TP | 1.661 |
| Ukuran model | 166,74 MB |
| Total parameter | 43.550.274 |
| Waktu inferensi rata-rata | 4,88 ms/gambar |

ResNet101 memiliki performa yang sedikit di bawah ResNet50 meskipun arsitekturnya lebih dalam. Walaupun jaringan ResNet101 memiliki jumlah parameter hampir dua kali lipat ResNet50 (43,5 juta vs 24,5 juta), peningkatan kedalaman tidak menghasilkan peningkatan akurasi pada dataset ini. Nilai FP (115) yang lebih tinggi dari FN (78) menunjukkan pola serupa dengan ResNet50, yaitu model lebih sering salah mengklasifikasikan Gram-negatif sebagai Gram-positif. Train loss ResNet101 (0,083178) lebih rendah dibanding ResNet50 (0,105381), tetapi validation loss-nya lebih tinggi (0,163975 vs 0,146586), mengindikasikan kecenderungan overfitting yang lebih besar pada ResNet101.

#### 4.5.9.5 Hasil Pelatihan EfficientNet-B0

Berdasarkan hasil eksperimen `experiments/efficientnet_b0_finetune_20260413_145240`, EfficientNet-B0 memperoleh accuracy 0,910868 dan macro F1-score 0,910782 pada epoch terbaik ke-34.

| Metrik | Nilai |
|---|---:|
| Best epoch | 34 |
| Accuracy | 0,910868 |
| Macro F1-score | 0,910782 |
| Train loss | 0,1713 |
| Validation loss | 0,2389 |
| TN | 1.638 |
| FP | 101 |
| FN | 209 |
| TP | 1.530 |

EfficientNet-B0 memiliki jumlah parameter yang lebih kecil dibanding ResNet, tetapi performanya masih berada di bawah ResNet50 dan ResNet101. Nilai FN yang cukup tinggi menunjukkan bahwa model lebih sering gagal mengenali Gram-positif.

#### 4.5.9.6 Hasil Pelatihan EfficientNet-B3

Berdasarkan `experiments/scenario_3d_efficientnet_b3/metrics_summary.json`, EfficientNet-B3 memperoleh accuracy 0,866967, precision 0,888376, recall 0,866967, F1-score 0,872866, dan ROC-AUC 0,944491.

| Metrik | Nilai |
|---|---:|
| Accuracy | 0,866967 |
| Precision | 0,888376 |
| Recall | 0,866967 |
| F1-score | 0,872866 |
| ROC-AUC | 0,944491 |
| TN | 1.195 |
| FP | 181 |
| FN | 55 |
| TP | 343 |

EfficientNet-B3 memiliki kemampuan pemisahan kelas yang cukup baik berdasarkan ROC-AUC, tetapi nilai FP yang tinggi menunjukkan bahwa model masih cukup sering memprediksi Gram-negatif sebagai Gram-positif.

#### 4.5.9.7 Hasil Pelatihan VGG-16

Berdasarkan `experiments/scenario_5a_vgg16/metrics_summary.json`, VGG-16 memperoleh accuracy 0,953113, precision 0,954171, recall 0,953113, F1-score 0,953399, kappa 0,892819, dan ROC-AUC 0,988942.

| Metrik | Nilai |
|---|---:|
| Accuracy | 0,953113 |
| Precision | 0,954171 |
| Recall | 0,953113 |
| F1-score | 0,953399 |
| Kappa | 0,892819 |
| ROC-AUC | 0,988942 |
| Total parameters | 119.587.138 |
| Trainable parameters | 119.587.138 |
| TN | 2.509 |
| FP | 119 |
| FN | 61 |
| TP | 1.150 |

VGG-16 menunjukkan performa yang sangat kompetitif meskipun memiliki jumlah parameter besar. Nilai ROC-AUC yang tinggi menunjukkan kemampuan pemisahan kelas yang baik.

#### 4.5.9.8 Hasil Pelatihan VGG-19

Berdasarkan `experiments/scenario_5b_vgg19/metrics_summary.json`, VGG-19 memperoleh accuracy 0,956239, precision 0,956847, recall 0,956239, F1-score 0,956426, kappa 0,899567, dan ROC-AUC 0,990255.

| Metrik | Nilai |
|---|---:|
| Accuracy | 0,956239 |
| Precision | 0,956847 |
| Recall | 0,956239 |
| F1-score | 0,956426 |
| Kappa | 0,899567 |
| ROC-AUC | 0,990255 |
| Total parameters | 124.896.834 |
| Trainable parameters | 124.896.834 |
| TN | 2.524 |
| FP | 104 |
| FN | 64 |
| TP | 1.147 |

VGG-19 menjadi salah satu model dengan performa tertinggi pada data eksperimen yang tersedia. Namun, jumlah parameter yang besar perlu menjadi pertimbangan ketika model akan digunakan pada sistem produksi.

#### 4.5.9.9 Hasil Pelatihan DenseNet121

Berdasarkan `experiments/retrain_densenet121_20260414_171014/summary.json`, DenseNet121 memperoleh accuracy 0,868678 dan macro F1-score 0,868514.

| Metrik | Nilai |
|---|---:|
| Accuracy | 0,868678 |
| Macro F1-score | 0,868514 |
| Train loss | 0,4002 |
| Validation loss | 0,3044 |
| TN | 1.573 |
| FP | 166 |
| FN | 291 |
| TP | 1.450 |

DenseNet121 memiliki jumlah parameter yang lebih efisien dibanding VGG, tetapi pada eksperimen ini performanya belum melampaui ResNet maupun VGG. Hasil DenseNet121 masih bersifat sementara karena hasil training Colab terbaru belum tersedia di workspace lokal.

## 4.6 Analisis Eksperimen

Bagian ini membahas perbandingan hasil dari seluruh skenario, analisis performa arsitektur, confusion matrix, metrik evaluasi, dan loss function.

### 4.6.1 Analisis Perbandingan Skenario

Berdasarkan lima skenario yang diuji, performa model meningkat ketika model menggunakan arsitektur yang lebih kuat, transfer learning, dan fine-tuning. Simple CNN tanpa augmentasi memiliki performa paling rendah dan paling tidak stabil. Penambahan augmentasi meningkatkan performa Simple CNN karena model memperoleh variasi visual yang lebih banyak.

Pada skenario from scratch, ResNet101 memperoleh performa terbaik. Hal ini menunjukkan bahwa residual connection membantu proses pelatihan jaringan dalam dari bobot acak. Namun, pada skenario transfer learning dan fine-tuning, performa terbaik tidak selalu bergantung pada kedalaman arsitektur, tetapi juga pada strategi fine-tuning, regularisasi, dan kecocokan arsitektur terhadap data.

### 4.6.2 Analisis Performa Arsitektur CNN

Tabel 4.18 menunjukkan ringkasan performa utama Skenario 5.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Keterangan |
|---|---:|---:|---:|---:|---:|---|
| ResNet50 | 0,951696 | 0,951718 | 0,951696 | 0,951696 | 0,988272 | Data riil `ta_resnet50_20260615_021715` |
| ResNet101 | 0,944508 | 0,944710 | 0,944508 | 0,944502 | 0,988207 | Data riil `ta_resnet101_20260615_044118` |
| EfficientNet-B0 | 0,910868 | Tidak tersedia | Tidak tersedia | 0,910782 | Tidak tersedia | `summary.json` |
| EfficientNet-B3 | 0,866967 | 0,888376 | 0,866967 | 0,872866 | 0,944491 | `metrics_summary.json` |
| VGG-16 | 0,953113 | 0,954171 | 0,953113 | 0,953399 | 0,988942 | `metrics_summary.json` |
| VGG-19 | 0,956239 | 0,956847 | 0,956239 | 0,956426 | 0,990255 | `metrics_summary.json` |
| DenseNet121 | 0,868678 | Tidak tersedia | Tidak tersedia | 0,868514 | Tidak tersedia | `summary.json` sementara |

VGG-19 memperoleh accuracy dan F1-score tertinggi pada ringkasan data yang tersedia. VGG-16, ResNet50, dan ResNet101 memiliki performa yang sangat berdekatan. Walaupun VGG-19 unggul secara metrik, ResNet50 tetap menjadi kandidat kuat karena memiliki performa tinggi dengan jumlah parameter yang lebih efisien dibanding VGG.

### 4.6.3 Analisis Confusion Matrix

Tabel 4.19 menunjukkan confusion matrix untuk model pada Skenario 5.

| Model | TN | FP | FN | TP | Pola Kesalahan Dominan |
|---|---:|---:|---:|---:|---|
| ResNet50 | 1.649 | 90 | 78 | 1.661 | FP lebih tinggi dari FN |
| ResNet101 | 1.624 | 115 | 78 | 1.661 | FP lebih tinggi dari FN |
| EfficientNet-B0 | 1.638 | 101 | 209 | 1.530 | FN jauh lebih tinggi |
| EfficientNet-B3 | 1.195 | 181 | 55 | 343 | FP jauh lebih tinggi |
| VGG-16 | 2.509 | 119 | 61 | 1.150 | FP lebih tinggi dari FN |
| VGG-19 | 2.524 | 104 | 64 | 1.147 | FP lebih tinggi dari FN |
| DenseNet121 | 1.573 | 166 | 291 | 1.450 | FN jauh lebih tinggi |

ResNet50 dan ResNet101 menunjukkan pola kesalahan dengan FP lebih tinggi dari FN, artinya model lebih sering salah mengklasifikasikan Gram-negatif sebagai Gram-positif. Pola ini serupa dengan VGG-16 dan VGG-19. EfficientNet-B0 dan DenseNet121 memiliki FN yang cukup tinggi, menunjukkan bahwa model masih sering gagal mengenali Gram-positif. Sedangkan EfficientNet-B3 memiliki FP jauh lebih tinggi dari FN, yang berarti model sangat condong memprediksi kelas Gram-positif.

### 4.6.4 Analisis Metrik Evaluasi

Accuracy memberikan gambaran performa keseluruhan, tetapi tidak cukup untuk melihat bias antar kelas. Oleh karena itu, precision, recall, F1-score, dan confusion matrix perlu dianalisis bersama. Pada data dengan potensi ketidakseimbangan kelas, F1-score menjadi metrik penting karena mempertimbangkan precision dan recall secara bersamaan.

ResNet50 memiliki precision Gram-positif sebesar 0,948601 dan recall Gram-positif sebesar 0,955147. Recall yang lebih tinggi dari precision menunjukkan bahwa model lebih sensitif dalam mendeteksi sampel Gram-positif, namun sedikit meningkatkan prediksi FP. ResNet101 memiliki precision Gram-positif 0,935248 dan recall Gram-positif 0,955147. Recall ResNet101 identik dengan ResNet50, namun precision-nya lebih rendah, sehingga ResNet101 menghasilkan lebih banyak FP (115 vs 90). Kedua model memiliki ROC-AUC yang hampir identik (0,9883 vs 0,9882), menunjukkan kemampuan pemisahan kelas yang sangat baik dan setara.

VGG-19 memiliki nilai F1-score tertinggi pada data yang tersedia, yaitu 0,956426. Namun, model ini memiliki jumlah parameter 124.896.834, sehingga membutuhkan memori dan komputasi lebih besar dibanding ResNet. Oleh karena itu, pemilihan model terbaik perlu mempertimbangkan tidak hanya akurasi, tetapi juga efisiensi inference dan kebutuhan deployment.

### 4.6.5 Analisis Loss Function

Loss function digunakan untuk memantau proses pembelajaran model selama training. Train loss yang terus menurun menunjukkan bahwa model semakin mampu menyesuaikan diri terhadap data latih. Validation loss digunakan untuk menilai kemampuan generalisasi model terhadap data yang tidak digunakan dalam training.

Pada ResNet50, train loss pada epoch terbaik (epoch 31) adalah 0,105381 dan validation loss 0,146586. Pada ResNet101, train loss pada epoch terbaik (epoch 37) adalah 0,083178 dan validation loss 0,163975. Walaupun ResNet101 memiliki train loss lebih rendah, validation loss-nya lebih tinggi dibanding ResNet50. Hal ini mengindikasikan bahwa ResNet101 lebih kuat menyesuaikan diri pada data latih, tetapi tidak menghasilkan generalisasi yang lebih baik dibanding ResNet50. Selisih antara train loss dan validation loss pada ResNet101 (0,081) lebih besar dibanding ResNet50 (0,041), yang memperkuat indikasi overfitting yang lebih besar pada arsitektur yang lebih dalam.

EfficientNet-B0 memiliki train loss 0,1713 dan validation loss 0,2389, sedangkan DenseNet121 memiliki train loss 0,4002 dan validation loss 0,3044. Nilai loss tersebut menunjukkan bahwa kedua model belum mencapai performa setinggi ResNet atau VGG pada konfigurasi eksperimen yang tersedia.

Secara keseluruhan, hasil eksperimen menunjukkan bahwa transfer learning dan fine-tuning memberikan kontribusi besar terhadap performa klasifikasi bakteri Gram. VGG-19 menghasilkan metrik tertinggi pada data yang tersedia (accuracy 0,956239, F1-score 0,956426), namun membutuhkan jumlah parameter yang jauh lebih besar (124,9 juta). ResNet50 dengan 24,5 juta parameter mencapai accuracy 0,951696 dan ROC-AUC 0,988272 dengan waktu inferensi rata-rata hanya 2,94 ms per gambar, menjadikannya alternatif paling seimbang antara performa, efisiensi komputasi, dan kecepatan deployment.
