# BAB 3 DESKRIPSI SISTEM

Bab ini menjelaskan rancangan sistem deteksi dan klasifikasi bakteri Gram berdasarkan citra mikroskopis hasil pewarnaan Gram. Perancangan sistem meliputi solusi yang dikembangkan, desain sistem model, desain sistem aplikasi, rancangan basis data, rancangan antarmuka, serta rancangan skenario uji coba. Struktur bab ini disusun agar setiap tahapan pengembangan dapat dipahami secara sistematis, mulai dari pengolahan data, pelatihan model, integrasi model dengan backend, hingga penyajian hasil klasifikasi kepada pengguna.

## 3.1 Deskripsi Solusi

Berdasarkan permasalahan yang telah diidentifikasi, solusi yang dikembangkan pada proyek akhir ini adalah sistem berbasis kecerdasan buatan untuk mendeteksi dan mengklasifikasikan bakteri Gram-positif dan Gram-negatif secara otomatis dari citra mikroskopis. Sistem ini dirancang untuk membantu proses interpretasi citra hasil pewarnaan Gram yang sebelumnya masih banyak dilakukan secara manual oleh tenaga laboratorium.

Sistem yang dibangun terdiri atas beberapa komponen utama, yaitu model deteksi objek menggunakan YOLO11, model klasifikasi citra berbasis Convolutional Neural Network (CNN), backend REST API menggunakan FastAPI, basis data PostgreSQL, serta frontend berbasis HTML/CSS/JS statis. Setiap komponen memiliki peran yang berbeda tetapi saling terhubung dalam satu alur kerja.

Model YOLO11 digunakan untuk mendeteksi lokasi bakteri pada citra mikroskopis. Hasil deteksi tersebut digunakan untuk melakukan pemotongan area objek bakteri secara otomatis. Area hasil crop kemudian diproses oleh model CNN untuk menentukan kelas bakteri, yaitu Gram-positif atau Gram-negatif. Hasil klasifikasi berupa label dan confidence score dikirimkan kembali ke backend untuk disimpan di PostgreSQL dan ditampilkan pada antarmuka pengguna.

Dengan pendekatan tersebut, sistem tidak hanya melakukan klasifikasi terhadap citra secara keseluruhan, tetapi juga memanfaatkan proses deteksi untuk memfokuskan model pada area yang mengandung objek bakteri. Hal ini diharapkan dapat meningkatkan akurasi klasifikasi dan mengurangi pengaruh latar belakang, artefak pewarnaan, atau elemen lain yang tidak relevan pada citra mikroskopis.

## 3.2 Desain Sistem Model

Desain sistem model menjelaskan tahapan pengembangan model kecerdasan buatan yang digunakan pada penelitian ini. Tahapan tersebut mencakup pembuatan dataset, preprocessing data, deteksi objek menggunakan YOLO11, pembentukan model CNN, serta perancangan skenario eksperimen.

### 3.2.1 Pembuatan Dataset

Dataset yang digunakan pada penelitian ini berupa citra mikroskopis hasil pewarnaan Gram. Data terdiri atas dua kelas utama, yaitu Gram-positif dan Gram-negatif. Setiap citra telah melalui proses pelabelan sehingga dapat digunakan untuk melatih model klasifikasi biner.

Data citra dikumpulkan dari sumber publik dan/atau sumber laboratorium yang relevan dengan penelitian klasifikasi bakteri Gram. Citra tersebut memuat variasi warna, bentuk, ukuran, fokus, dan pencahayaan yang umum ditemukan pada hasil pengamatan mikroskopis. Variasi tersebut penting agar model tidak hanya menghafal pola tertentu, tetapi mampu mengenali karakteristik visual bakteri pada kondisi citra yang berbeda.

Pada sisi deteksi objek, area bakteri pada citra diberi anotasi agar dapat digunakan dalam pelatihan model YOLO11. Anotasi ini berfungsi untuk menunjukkan posisi objek bakteri yang akan dideteksi. Pada sisi klasifikasi, setiap citra atau hasil crop dari citra diberi label Gram-positif atau Gram-negatif sesuai karakteristik pewarnaannya.

### 3.2.2 Preprocessing Data

Preprocessing dilakukan untuk memastikan citra yang digunakan dalam pelatihan memiliki format, ukuran, dan distribusi nilai piksel yang sesuai dengan kebutuhan model. Tahapan preprocessing yang digunakan meliputi resize, normalisasi, pemisahan dataset, dan penyesuaian format data.

Citra dikonversi ke format RGB dan diubah ukurannya menjadi 224 x 224 piksel. Ukuran ini dipilih karena sesuai dengan konfigurasi umum arsitektur CNN pretrained seperti ResNet, EfficientNet, VGG, dan DenseNet. Selain itu, ukuran 224 x 224 piksel memungkinkan proses batch training dan inference berjalan lebih efisien.

Normalisasi dilakukan menggunakan nilai mean dan standard deviation yang umum digunakan pada model pretrained ImageNet. Normalisasi ini penting karena sebagian besar arsitektur transfer learning pada penelitian ini menggunakan bobot awal dari ImageNet, sehingga distribusi input perlu dibuat serupa dengan distribusi input saat pretraining.

Dataset dibagi menjadi train set, validation set, dan test set. Train set digunakan untuk melatih model, validation set digunakan untuk memantau performa selama pelatihan dan menentukan checkpoint terbaik, sedangkan test set digunakan untuk evaluasi akhir model.

### 3.2.3 Deteksi Objek Menggunakan YOLO11

YOLO11 digunakan sebagai model deteksi objek untuk menemukan lokasi bakteri pada citra mikroskopis. Model ini dipilih karena mampu melakukan deteksi secara cepat dalam satu tahap (one-stage detector), sehingga sesuai untuk sistem yang membutuhkan respons inference yang efisien.

Pada proses pelatihan YOLO11, citra dan anotasi bounding box digunakan untuk mengajarkan model mengenali posisi bakteri. Setelah model dilatih, YOLO11 menghasilkan koordinat bounding box dari objek yang terdeteksi. Koordinat tersebut kemudian digunakan untuk melakukan auto crop pada area bakteri.

Hasil crop dari YOLO11 menjadi input bagi model CNN classifier. Dengan cara ini, CNN tidak perlu memproses seluruh citra mikroskopis, tetapi hanya area yang memiliki kemungkinan besar berisi objek bakteri. Pendekatan ini membantu mengurangi noise dari latar belakang dan meningkatkan fokus model terhadap fitur visual yang relevan.

### 3.2.4 Pembentukan Model CNN

Model CNN digunakan untuk melakukan klasifikasi citra ke dalam dua kelas, yaitu Gram-positif dan Gram-negatif. Penelitian ini menggunakan dua kelompok arsitektur CNN. Kelompok pertama adalah Simple CNN yang dibangun dari awal sebagai baseline. Kelompok kedua adalah arsitektur transfer learning berbasis model pretrained ImageNet, yaitu ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121.

Simple CNN digunakan untuk mengetahui performa dasar model tanpa bantuan bobot pretrained. Arsitektur ini terdiri atas beberapa convolutional layer, activation function, pooling layer, dan fully connected layer. Model ini menjadi pembanding awal untuk melihat pengaruh augmentasi data, transfer learning, dan fine-tuning pada skenario eksperimen berikutnya.

Pada arsitektur transfer learning, bagian backbone model digunakan sebagai feature extractor. Classifier bawaan dari model pretrained diganti dengan classifier baru yang menghasilkan dua output kelas. Strategi ini memungkinkan model memanfaatkan fitur visual umum yang telah dipelajari dari ImageNet, kemudian menyesuaikannya dengan karakteristik citra Gram-stain.

Proses fine-tuning dilakukan dalam dua tahap. Tahap pertama melatih classifier head dengan backbone dalam kondisi frozen. Tahap kedua membuka sebagian layer akhir pada backbone agar fitur yang dipelajari dapat disesuaikan dengan domain citra bakteri Gram. Bagian backbone yang dibuka berbeda pada setiap arsitektur, menyesuaikan struktur model dan kebutuhan komputasi.

![Diagram Arsitektur Simple CNN](../images/simple_cnn_architecture.png)

*Gambar 3.1 Diagram Arsitektur Simple CNN*

Tabel 3.1 menunjukkan rancangan umum arsitektur Simple CNN yang digunakan sebagai baseline.

| Layer | Tipe | Konfigurasi | Output Shape |
|---|---|---|---|
| Input | Citra RGB | 224 x 224 x 3 | 224 x 224 x 3 |
| Conv1 + Pool1 | Conv2D + ReLU + MaxPool | 3 ke 32 channel | 112 x 112 x 32 |
| Conv2 + Pool2 | Conv2D + ReLU + MaxPool | 32 ke 64 channel | 56 x 56 x 64 |
| Conv3 + Pool3 | Conv2D + ReLU + MaxPool | 64 ke 128 channel | 28 x 28 x 128 |
| Conv4 + Pool4 | Conv2D + ReLU + MaxPool | 128 ke 256 channel | 14 x 14 x 256 |
| Conv5 + Pool5 | Conv2D + ReLU + MaxPool | 256 ke 512 channel | 7 x 7 x 512 |
| Flatten | Flatten | 7 x 7 x 512 ke 25.088 | 25.088 |
| FC1 | Fully Connected + ReLU | 25.088 ke 1.024 | 1.024 |
| FC2 | Fully Connected + ReLU | 1.024 ke 512 | 512 |
| FC3 | Fully Connected + Softmax | 512 ke 2 | 2 kelas |

Tabel 3.2 menunjukkan rancangan umum arsitektur transfer learning.

| Tahap | Komponen | Keterangan |
|---|---|---|
| Input | Citra 224 x 224 x 3 | Citra hasil preprocessing atau hasil crop YOLO11 |
| Backbone | ResNet, EfficientNet, VGG, DenseNet | Feature extractor pretrained ImageNet |
| Feature vector | Global pooling atau flatten | Mengubah feature map menjadi vektor fitur |
| Classifier head | Fully connected layer baru | Disesuaikan dengan dua kelas output |
| Output | Softmax | Gram-positif atau Gram-negatif beserta confidence score |

### 3.2.5 Skema Eksperimen

Eksperimen dirancang untuk mengevaluasi pengaruh beberapa pendekatan pelatihan terhadap performa klasifikasi. Skenario eksperimen mencakup pelatihan Simple CNN dari awal, Simple CNN dengan augmentasi data, pelatihan beberapa arsitektur CNN dari awal, transfer learning dengan fine-tuning, serta transfer learning dan fine-tuning pada beberapa arsitektur CNN populer.

Tabel 3.3 menunjukkan rancangan skenario uji coba yang digunakan.

| Skenario | Deskripsi | Arsitektur | Augmentasi | Transfer Learning | Fine-tuning |
|---|---|---|---|---|---|
| 1 | Baseline CNN dari awal | Simple CNN | Tidak | Tidak | Tidak |
| 2 | CNN dengan augmentasi data | Simple CNN | Ya | Tidak | Tidak |
| 3 | Pelatihan arsitektur CNN dari awal | ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, DenseNet121 | Tidak | Tidak | Tidak |
| 4 | Transfer learning dan fine-tuning terbatas | ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3 | Tidak | Ya | Ya |
| 5 | Transfer learning dan fine-tuning dengan augmentasi | ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, DenseNet121 | Ya | Ya | Ya |

## 3.3 Desain Sistem

Desain sistem menjelaskan hubungan antar komponen aplikasi yang dikembangkan. Sistem terdiri atas frontend, backend, basis data, model deteksi YOLO11, dan model klasifikasi CNN.

### 3.3.1 Design Top Level

Secara umum, pengguna mengakses sistem melalui frontend berbasis web. Pengguna mengunggah citra mikroskopis hasil pewarnaan Gram, kemudian frontend mengirimkan citra tersebut ke backend FastAPI. Backend menjalankan pipeline AI yang terdiri atas deteksi objek menggunakan YOLO11, auto crop area bakteri, preprocessing citra, dan klasifikasi menggunakan model CNN.

Hasil prediksi berupa label kelas dan confidence score dikembalikan dalam format JSON. Backend menyimpan hasil klasifikasi, metadata citra, dan informasi model ke dalam PostgreSQL. Data tersebut kemudian dapat ditampilkan kembali pada frontend sebagai riwayat klasifikasi atau bahan evaluasi model.

Komponen utama pada design top level adalah sebagai berikut.

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Frontend | HTML/CSS/JS statis | Antarmuka pengguna untuk upload citra dan melihat hasil klasifikasi |
| Backend | FastAPI | Endpoint API, validasi request, pemanggilan model, dan pengelolaan respons |
| Model deteksi | YOLO11 | Mendeteksi lokasi bakteri pada citra |
| Model klasifikasi | PyTorch CNN | Mengklasifikasikan citra menjadi Gram-positif atau Gram-negatif |
| Database | PostgreSQL | Menyimpan data pengguna, metadata citra, hasil klasifikasi, dan metadata model |

### 3.3.2 Design System Low Level

Pada level rendah, alur sistem dimulai dari citra yang dikirim oleh pengguna. Backend menerima file citra, melakukan validasi format, lalu meneruskannya ke pipeline AI. Pipeline AI menjalankan tahap deteksi objek, auto crop, resize, normalisasi, konversi tensor, dan inference model CNN.

Jika YOLO11 berhasil mendeteksi objek bakteri, area bounding box digunakan sebagai region of interest untuk klasifikasi. Jika sistem menggunakan mode manual atau citra tidak membutuhkan crop otomatis, citra input dapat langsung diproses oleh CNN setelah melalui preprocessing.

Alur low level sistem adalah sebagai berikut.

1. Pengguna mengunggah citra Gram-stain melalui frontend.
2. Frontend mengirim request ke endpoint FastAPI.
3. Backend melakukan validasi file dan menyimpan metadata awal.
4. YOLO11 mendeteksi area bakteri pada citra.
5. Sistem melakukan auto crop berdasarkan bounding box hasil deteksi.
6. Citra hasil crop diresize dan dinormalisasi.
7. CNN classifier melakukan prediksi kelas Gram-positif atau Gram-negatif.
8. Backend menyimpan hasil klasifikasi ke PostgreSQL.
9. Backend mengirim respons JSON ke frontend.
10. Frontend menampilkan label, confidence score, dan riwayat hasil.

### 3.3.3 Desain Database PostgreSQL

Basis data PostgreSQL digunakan untuk menyimpan data yang dibutuhkan sistem secara terstruktur. Desain basis data mencakup tabel pengguna, dataset, hasil klasifikasi, metadata model AI, dan status model.

Tabel utama yang digunakan adalah sebagai berikut.

| Tabel | Fungsi |
|---|---|
| users | Menyimpan data pengguna, seperti nama, email, password, dan role |
| datasets | Menyimpan metadata citra Gram-stain yang diunggah ke sistem |
| classifications | Menyimpan hasil klasifikasi, label prediksi, confidence score, dan relasi ke citra serta model |
| ai_models | Menyimpan metadata model AI, seperti nama model, versi, task type, metrik performa, dan status aktif |
| model_status | Menyimpan riwayat perubahan status model, misalnya aktif, tidak aktif, atau direkomendasikan |

Dengan rancangan ini, sistem dapat menyimpan hasil klasifikasi secara konsisten dan mendukung proses evaluasi atau retraining model di masa depan.

### 3.3.4 Alur Kerja Aplikasi

Alur kerja aplikasi menggambarkan proses dari input pengguna hingga output klasifikasi ditampilkan. Input sistem berupa citra mikroskopis hasil pewarnaan Gram. Citra tersebut diproses oleh backend dan pipeline AI, kemudian menghasilkan output berupa prediksi kelas dan confidence score.

Tahapan alur kerja aplikasi adalah sebagai berikut.

| Tahap | Proses | Output |
|---|---|---|
| Input | Pengguna mengunggah citra Gram-stain | File citra |
| Deteksi | YOLO11 mendeteksi area bakteri | Bounding box |
| Auto crop | Sistem memotong area bakteri | Citra region of interest |
| Preprocessing | Resize, normalisasi, dan konversi tensor | Tensor input model |
| Klasifikasi | CNN memprediksi kelas | Label dan confidence score |
| Penyimpanan | Backend menyimpan hasil ke PostgreSQL | Riwayat klasifikasi |
| Output | Frontend menampilkan hasil | Informasi klasifikasi |

## 3.4 Rancangan Antarmuka

Rancangan antarmuka dibuat untuk memberikan gambaran visual dan fungsionalitas sistem yang digunakan oleh pengguna. Antarmuka dirancang agar pengguna dapat mengunggah citra, melihat hasil klasifikasi, mengelola model, dan memantau performa model.

### 3.4.1 Mockup Beranda

*Gambar 3.2 Tampilan Beranda*

Halaman beranda berfungsi sebagai dashboard utama. Halaman ini menampilkan ringkasan performa model aktif, seperti akurasi, precision, recall, F1-score, serta grafik perkembangan performa model dari beberapa sesi pelatihan. Dashboard membantu admin atau analis melihat kondisi model secara cepat.

### 3.4.2 Mockup Model AI

*Gambar 3.3 Tampilan Fitur Model AI*

Halaman model AI digunakan untuk mengelola daftar model yang tersedia di sistem. Admin dapat melihat nama model, versi, arsitektur, status aktif, dan metrik performa model. Halaman ini juga digunakan untuk memilih model yang akan dipakai pada proses inference.

### 3.4.3 Mockup Konfigurasi Model

*Gambar 3.4 Tampilan Konfigurasi Model AI*

Halaman konfigurasi model menyediakan fasilitas untuk mengelola proses pelatihan ulang atau fine-tuning. Pada halaman ini, admin dapat memilih dataset, arsitektur model, parameter training, serta memantau progres pelatihan yang sedang berlangsung.

### 3.4.4 Mockup Klasifikasi

*Gambar 3.5 Tampilan Klasifikasi*

Halaman klasifikasi digunakan oleh analis untuk mengunggah citra mikroskopis dan menjalankan proses prediksi. Setelah citra diproses, halaman ini menampilkan hasil deteksi, hasil klasifikasi, confidence score, dan opsi anotasi ulang apabila hasil prediksi perlu dikoreksi oleh pengguna ahli.

## 3.5 Rancangan Skenario Uji Coba

Rancangan skenario uji coba digunakan untuk memastikan bahwa model dan sistem yang dikembangkan dapat dievaluasi secara terukur. Pengujian dilakukan terhadap model klasifikasi CNN dan integrasi sistem aplikasi.

### 3.5.1 Tujuan Uji Coba

Tujuan uji coba adalah mengukur kemampuan model dalam mengklasifikasikan citra bakteri Gram-positif dan Gram-negatif berdasarkan hasil pewarnaan Gram. Selain itu, uji coba juga bertujuan mengevaluasi pengaruh augmentasi data, transfer learning, fine-tuning, dan pemilihan arsitektur CNN terhadap performa model.

### 3.5.2 Metode Evaluasi

Evaluasi model dilakukan menggunakan metrik klasifikasi biner. Metrik yang digunakan meliputi accuracy, precision, recall, F1-score, confusion matrix, dan ROC-AUC jika tersedia. Accuracy digunakan untuk mengukur proporsi prediksi benar secara keseluruhan. Precision mengukur ketepatan prediksi kelas positif, recall mengukur kemampuan model mendeteksi seluruh sampel positif, sedangkan F1-score digunakan untuk menilai keseimbangan antara precision dan recall.

Confusion matrix digunakan untuk melihat jumlah true negative, false positive, false negative, dan true positive. Dari confusion matrix, dapat dianalisis kecenderungan kesalahan model, misalnya apakah model lebih sering salah mengklasifikasikan Gram-negatif sebagai Gram-positif atau sebaliknya.

### 3.5.3 Luaran Uji Coba

Luaran dari uji coba meliputi model terbaik pada setiap skenario, tabel perbandingan performa, grafik training loss dan validation loss, confusion matrix, serta dokumentasi konfigurasi pelatihan. Hasil uji coba tersebut digunakan sebagai dasar analisis pada BAB 4 untuk menentukan arsitektur dan strategi pelatihan yang paling sesuai dengan kebutuhan sistem klasifikasi bakteri Gram.
