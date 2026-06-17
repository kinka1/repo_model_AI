# BAB 3 DESAIN SISTEM

DESKRIPSI SISTEM

DESKRIPSI SOLUSI

Solusi yang ditawarkan dalam proyek akhir ini adalah pengembangan sistem berbasis kecerdasan buatan (AI) yang mampu mendeteksi jenis bakteri Gram-positif dan Gram-negatif secara otomatis dari citra mikroskopis hasil pewarnaan Gram. Sistem ini mengintegrasikan teknologi deep learning menggunakan arsitektur CNN, REST API berbasis framework backend modern, dan tampilan antarmuka berbasis web.

Solusi ini dirancang untuk mengotomatisasi proses identifikasi bakteri dalam lingkungan laboratorium dan klinik, dengan fitur-fitur utama sebagai berikut:

Model Deep Learning berbasis CNN (PyTorch)

Model dikembangkan menggunakan framework PyTorch dan Python, dengan arsitektur seperti DenseNet, VGG-16 atau ResNet50, ResNet101, EfficientNet, VGG, dan DenseNet121. Model dilatih untuk mengenali pola morfologis dan warna dari citra mikroskopis bakteri Gram-stain.

Backend REST API (FastAPI)

Backend dikembangkan menggunakan framework FastAPI yang berbasis Python. Backend bertanggung jawab untuk mengelola komunikasi antara frontend dan model AI. Menyediakan endpoint untuk:

Upload gambar

Prediksi hasil klasifikasi (inference)

Manajemen data pengguna dan riwayat klasifikasi

Database (PostgreSQL)

Seluruh data gambar, hasil klasifikasi, dan metadata disimpan secara terstruktur di PostgreSQL. Database digunakan untuk mencatat histori klasifikasi dan mendukung retraining model di masa depan.

Frontend (HTML/CSS/JS statis)

Sistem antarmuka dikembangkan menggunakan framework HTML/CSS/JS.

Pengguna dapat mengakses sistem melalui web untuk:

Mengunggah citra Gram-stain

Melihat hasil klasifikasi

Mengakses riwayat deteksi

Tampilan dirancang responsif dan user-friendly.

Deployment & Integrasi

Seluruh komponen diintegrasikan secara modular dengan komunikasi REST API. Sistem dapat dikembangkan ke arah microservice atau cloud-native sesuai kebutuhan ke depannya.

Sistem ini memungkinkan integrasi skala kecil maupun besar untuk kebutuhan laboratorium rumah sakit, serta mendukung peningkatan akurasi model secara bertahap melalui proses retraining berbasis data baru.

## DESAIN SISTEM

## DESIGN TOP LEVEL

Berikut ini merupakan desain sistem secara garis besar yang digunakan dalam pengembangan sistem.

*Gambar 3. 1 Design Sistem*

*Gambar 3.1 menjelaskan mengenai design sistem yang akan digunakan untuk mengembangkan system backend. Beberapa komponen yang terdapat pada design sistem di gambar 3.1, meliputi :*

Lapisan Frontend (HTML/CSS/JS statis)

Bagian Frontend dikembangkan menggunakan HTML/CSS/JS yang mana berfungsi sebagai antarmuka pengguna. Terdapat tiga jenis :

- Dokter: Antarmuka untuk dokter dirancang untuk menampilkan hasil klasifikasi yang telah diklasifikasi. Selain itu, dokter dapat memberikan umpan balik dengan melakukan anotasi ulang (re-annotated data) pada hasil klasifikasi, yang kemudian dikirim kembali ke backend untuk proses retraining model.
- Analis: Antarmuka untuk analis difokuskan pada proses klasifikasi yang dapat mengunggah gambar bakteri untuk mendapatkan hasil klasifikasi dari model AI yang aktif.
- Admin: Antarmuka untuk admin memungkinkan pengguna untuk mengelola seluruh aspek teknis sistem, termasuk mengunggah model AI baru, melakukan konfigurasi, dan memulai proses pelatihan ulang (retraining) model.
Lapisan Backend (FastAPI)

Bagian Backend dikembangkan menggunakan FastAPI yang berfungsi sebagai jalur komunikasi antara frontend dan model AI, serta menghasilkan endpoint API yang akan digunakan oleh frontend.

Lapisan Database

Sistem menggunakan PostgreSQL sebagai sistem manajemen basis data (RDBMS). Basis data ini bertanggung jawab untuk menyimpan seluruh data penting secara persisten dan terstruktur, seperti data gambar yang telah diklasifikasikan, riwayat hasil klasifikasi, informasi pengguna, dan metadata model AI.

Lapisan Model AI (Python + PyTorch)

Inti dari sistem klasifikasi ini adalah model deep learning yang dibangun menggunakan framework PyTorch dengan bahasa pemrograman Python. Model ini dirancang untuk melakukan klasifikasi citra mikroskopis hasil pewarnaan Gram ke dalam dua kelas utama: Gram-positif dan Gram-negatif.

Model menerima input berupa gambar dari frontend melalui backend API. Setelah diterima, gambar tersebut diproses dan dianalisis oleh model CNN (Convolutional Neural Network) yang telah dilatih sebelumnya menggunakan data set yang beranotasi. Proses klasifikasi menghasilkan label kelas dan confidence score, yang kemudian dikembalikan ke backend untuk disimpan dan ditampilkan kepada pengguna.

Beberapa poin penting dari lapisan model AI ini adalah:

Platform & Framework:

Model dikembangkan menggunakan PyTorch dan Python, memungkinkan fleksibilitas dalam pelatihan dan deployment.

Arsitektur Model:

Beberapa arsitektur CNN yang digunakan dalam eksperimen antara lain:

ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3

Arsitektur terbaik dipilih berdasarkan evaluasi akurasi dan waktu inferensi pada data validasi.

## Desain Arsitektur CNN

Sistem ini menggunakan dua kelompok desain arsitektur CNN yang berbeda, yaitu (1) arsitektur Simple CNN yang dibangun dari awal (from scratch) sebagai baseline, dan (2) arsitektur transfer learning berbasis backbone pretrained (ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121) yang dikombinasikan dengan classifier head khusus untuk klasifikasi dua kelas (Gram-positif dan Gram-negatif). Kedua bentuk arsitektur ini menerima input citra berukuran 224×224×3 (RGB) dan menghasilkan output berupa label kelas beserta confidence score.

Desain Arsitektur Simple CNN (Baseline)

Arsitektur Simple CNN dirancang dengan 5 convolutional layer yang disusun secara berurutan dengan peningkatan jumlah channel (channel progression) 3→32→64→128→256→512. Setiap convolutional layer menggunakan kernel 3×3 dengan padding 1 (mempertahankan ukuran spasial) dan diikuti oleh Max Pooling 2×2 yang mereduksi dimensi spasial menjadi setengahnya. Hasil ekstraksi fitur pada layer terakhir kemudian diratakan (flatten) dan diteruskan ke 3 fully connected layer dengan dimensi 25.088→1.024→512→2. Fungsi aktivasi ReLU digunakan pada seluruh hidden layer, sedangkan output layer menggunakan Softmax untuk menghasilkan probabilitas dua kelas (Gram-positif dan Gram-negatif). Diagram arsitektur ini ditampilkan pada Gambar 4.2 di BAB 4.

![Diagram Arsitektur Simple CNN](images/simple_cnn_architecture.png)

*Tabel 3.2 Desain Arsitektur Simple CNN*

| Layer | Tipe | Konfigurasi | Output Shape |
|---|---|---|---|
| Input | Citra | 224×224×3 | 224×224×3 |
| Conv1 + Pool1 | Conv2D 3×3 (pad 1) + ReLU + MaxPool 2×2 | 3→32 channel | 112×112×32 |
| Conv2 + Pool2 | Conv2D 3×3 (pad 1) + ReLU + MaxPool 2×2 | 32→64 channel | 56×56×64 |
| Conv3 + Pool3 | Conv2D 3×3 (pad 1) + ReLU + MaxPool 2×2 | 64→128 channel | 28×28×128 |
| Conv4 + Pool4 | Conv2D 3×3 (pad 1) + ReLU + MaxPool 2×2 | 128→256 channel | 14×14×256 |
| Conv5 + Pool5 | Conv2D 3×3 (pad 1) + ReLU + MaxPool 2×2 | 256→512 channel | 7×7×512 |
| Flatten | Flatten | 7×7×512 → 25.088 | 25.088 |
| FC1 | Fully Connected + ReLU | 25.088→1.024 | 1.024 |
| FC2 | Fully Connected + ReLU | 1.024→512 | 512 |
| FC3 (Output) | Fully Connected + Softmax | 512→2 | 2 (Gram-positif, Gram-negatif) |

Desain Arsitektur Transfer Learning (Backbone + Classifier Head)

Pada arsitektur transfer learning, struktur model dibagi menjadi dua bagian utama: backbone dan classifier head. Backbone merupakan bagian feature extractor dari arsitektur pretrained (ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, atau DenseNet121) yang sudah dilatih sebelumnya pada dataset ImageNet, sehingga sudah memiliki kemampuan mengenali fitur visual umum seperti tepi, tekstur, dan pola bentuk. Classifier head asli dari masing-masing arsitektur diganti dengan fully connected layer baru yang disesuaikan dengan jumlah kelas pada penelitian ini.

*Tabel 3.3 Desain Umum Arsitektur Transfer Learning*

| Tahap | Komponen | Keterangan |
|---|---|---|
| Input | Citra 224×224×3 | Hasil preprocessing/auto crop YOLO11 |
| Backbone | ResNet50 / ResNet101 / EfficientNet-B0 / EfficientNet-B3 / VGG-16 / VGG-19 / DenseNet121 (pretrained ImageNet) | Feature extractor; dibekukan (frozen) pada Skenario 3, dan dibuka sebagian (fine-tuning) pada Skenario 4-5 sesuai BAB 4 |
| Feature Map | Global Average Pooling / Flatten (sesuai arsitektur asli) | Mengubah feature map menjadi vektor fitur |
| Classifier Head | Fully Connected layer baru | Menggantikan classifier bawaan arsitektur, disesuaikan dengan 2 kelas |
| Output | Fully Connected + Softmax | 2 kelas: Gram-positif dan Gram-negatif, beserta confidence score |

Pemilihan bagian backbone yang dibekukan atau dibuka (fine-tuned) berbeda untuk setiap arsitektur, misalnya layer4 pada ResNet, beberapa MBConv block terakhir beserta conv_head pada EfficientNet, dan dense block terakhir beserta classifier pada DenseNet. Detail konfigurasi hyperparameter, strategi fine-tuning per arsitektur, serta perbandingan karakteristik tiap arsitektur dijelaskan lebih lanjut pada BAB 4 subbab Spesifikasi Arsitektur dan Konfigurasi Training serta Perbandingan Karakteristik Arsitektur CNN.

Implementasi CNN pada Sistem:

CNN digunakan sebagai model klasifikasi utama setelah citra bakteri diterima dan diproses oleh sistem. Input model berupa citra hasil preprocessing atau hasil crop dari area bakteri yang telah disesuaikan ke ukuran standar 224x224 piksel. Ukuran ini dipilih karena sesuai dengan konfigurasi umum arsitektur CNN pretrained seperti ResNet dan EfficientNet, serta memudahkan proses batch inference pada pipeline PyTorch.

Pada alur inference, citra dari frontend dikirim ke backend FastAPI, kemudian diteruskan ke pipeline AI untuk dilakukan preprocessing. Jika pengguna memilih proses otomatis, YOLO11 digunakan terlebih dahulu untuk mendeteksi lokasi bakteri dan menghasilkan area crop. Area tersebut kemudian menjadi input bagi CNN classifier. Jika pengguna menggunakan input manual, citra atau region yang dipilih langsung diproses oleh CNN untuk klasifikasi.

Output CNN berupa dua kelas utama, yaitu Gram-positif dan Gram-negatif. Selain label kelas, model juga menghasilkan confidence score yang menunjukkan tingkat keyakinan model terhadap prediksi. Hasil ini dikembalikan ke backend untuk disimpan di PostgreSQL dan ditampilkan kembali pada antarmuka pengguna.

Alur Klasifikasi CNN:

1. Citra mikroskopis diterima dari frontend melalui endpoint FastAPI.
2. Citra diproses melalui tahap preprocessing, seperti resize, normalisasi, dan penyesuaian format tensor.
3. Jika auto crop digunakan, YOLO11 mendeteksi area bakteri terlebih dahulu.
4. Citra hasil preprocessing atau crop dimasukkan ke CNN classifier berbasis PyTorch.
5. CNN menghasilkan label Gram-positif atau Gram-negatif beserta confidence score.
6. Backend menyimpan hasil klasifikasi dan mengirimkan respons ke frontend.

Retraining dan Fine-Tuning:

Model AI dapat diperbarui melalui proses retraining menggunakan data baru yang diperoleh dari rumah sakit atau hasil anotasi dokter. Hal ini memungkinkan sistem untuk terus beradaptasi terhadap data nyata dan meningkatkan akurasi klasifikasi.

Proses retraining dapat dilakukan dari awal (training from scratch) atau melalui pendekatan fine-tuning, di mana model yang telah ada disesuaikan kembali (adjusted) dengan data tambahan. Pendekatan fine-tuning ini lebih efisien secara waktu dan sumber daya, terutama saat model telah memiliki bobot awal yang stabil dari pelatihan sebelumnya.

Output Model:

Model mengembalikan hasil dalam format:

label: “Gram-positif” atau “Gram-negatif”

confidence: nilai probabilitas (%) dari keyakinan model terhadap hasil klasifikasi

Integrasi:

Model di-serve melalui backend menggunakan pipeline Python yang disambungkan ke API. Pipeline ini bertugas memanggil model, memproses input gambar, dan mengembalikan hasil klasifikasi ke frontend.

Dengan pendekatan ini, sistem tidak hanya melakukan klasifikasi satu arah, namun juga membangun siklus pembelajaran berkelanjutan (continuous learning), di mana kualitas model akan meningkat seiring pertambahan data yang divalidasi oleh ahli.

Data Source

Untuk meningkatkan performa model, sistem dirancang untuk dapat melakukan fine-tuning atau retraining menggunakan Data source. Data source ini dapat berupa data publik (Public Data) yang tersedia secara umum yang relevan untuk memperkaya data set pelatihan.

## DESIGN SYSTEM LOW LEVEL

Berikut merupakan detail design system

*Gambar 3. 2 Design System*

Pada gambar 3.2 dapat dilihat bahwa detail arsitektur ini terdiri dari lima komponen utama yaitu

Data Source

Bagian ini berisikan source data yang telah dikumpulkan yang mana berasal dari data public yang telah digunakan di jurnal IEEE. Dataset tersebut berasal dari Rumah Sakit Umum PLA Tiongkok yang diambil pada tahun 2018 hingga 2022.

Preprocessing

Bagian ini berisikan proses preprocessing data sebelum diolah menjadi sebuah model AI. Prosesnya ada resize, denoising, dan split data set. Resize: image dilakukan resize menjadi ukuran 224 x 224, lalu dilakukan denoising pada image untuk menghilangkan gangguan. Kemudian image dilakukan pemisahan menjadi train : val : test dengan masing-masing 70% : 20% : 10%.

Creating AI Model

Bagian ini merupakan tahapan selanjutnya setelah data dilakukan pemisahan. Data yang termasuk bagian train akan diolah untuk menghasilkan model YOLO dan CNN. Model YOLO akan digunakan untuk mendeteksi keberadaan bakterinya, sedangkan CNN akan digunakan untuk klasifikasi bakteri. Model YOLO yang digunakan ialah YOLO11.

Database

Bagianini berfungsi sebagai data base untuk menyimpan data yang akan digunakan. Database menggunakanPostgreSQL

Output

Bagian ini berisikan model AI yang akan dikonsumsi oleh backend. Backend akan menggunakan FastAPI dan menghasilkan sebuah endpoint yang nanti akan digunakan oleh frontend untuk diimplementasikan di websitenya.

## Design Database (PostgreSQL)

Bagian ini menjelaskan rancangan teknis dari basis data yang akan digunakan dalam sistem. Desain basis data ini dirancang untuk dapat menyimpan dan mengelola seluruh informasi yang berkaitan dengan pengguna, model AI, data set, dan hasil klasifikasi secara terstruktur dan efisien. Rancangan ini mencakup lima tabel utama yang saling berelasi untuk mendukung keseluruhan fungsionalitas aplikasi.

*Gambar 3. 3Tampilan Design Database*

*Tabel Users*

*Tabel Users digunakan untuk menyimpan data users yang meliputi id, name, password, email, role (analis, admin dan doctor) yang mana akan digunakan untuk mengakses web.*

*Tabel data sets*

*Tabel data sets berfungsi untuk mencatat metadata dari setiap gambar Gram Stain yang diunggah ke dalam sistem oleh Analis. Metadata ini penting untuk mendukung proses klasifikasi model.*

*Tabel classifications*

*Tabel classifications adalah tabel transaksional utama yang berfungsi untuk mencatat setiap hasil dari proses klasifikasi. Tabel ini menghubungkan data gambar, model AI yang digunakan, dan hasil anotasi ulang oleh dokter.*

*Tabel ai_model*

*Tabel ai_model digunakan untuk menyimpan metadata setiap model AI yang terdaftar dalam sistem, meliputi nama model, jenis tugas (task_type), versi, serta metrik performa seperti accuracy, f1-score, dan waktu inferensi. Tabel ini juga mencatat status model (aktif/tidak aktif) dan apakah model tersebut menjadi model yang direkomendasikan.*

*Tabel model_status*

*Tabel model_status berfungsi sebagai log untuk melacak riwayat aktivitas atau status dari setiap model AI, seperti status aktif atau tidak aktif. Tabel ini terhubung dengan tabel ai_models.*

## Alur kerja aplikasi

Berikut ini merupakan desain flow aplikasi :

*Gambar 3. 4 Design Flow Aplikasi*

*Gambar 3.4 menjelaskan mengenai design flow aplikasi yang akan digunakan untuk mengembangkan system backend. Beberapa komponen yang terdapat pada design sistem di gambar 3.4, meliputi :*

Input

User melakukan upload sebuah citra mikroskopis bakteri yang telah dilakukan pewarnaan.

Proses

Deteksi dengan Model YOLO

Melakukan deteksi bakteri dengan model YOLO yang telah dilakukan training.

Auto Crop

Melakukan auto crop dari hasil deteksi dengan model YOLO.

Klasifikasi

Melakukan klasifikasi dengan model CNN yang mana hasilnya akan berupa label bakteri dan confidence_score.

Simpan data

Backend akan melakukan penyimpanan hasil yang telah diklasifikasikan di PostgreSQL.

Output

Output berupa JSON yang mana akan digunakan oleh frontend untuk menampilkan hasil dari pengelompokan.

## Mockup

## Mockup atau purwarupa antarmuka pengguna dirancang untuk memberikan gambaran visual mengenai desain dan fungsionalitas dari aplikasi web yang akan dikembangkan.

## Mockup Beranda

*Gambar 3.5 Tampilan Beranda*

Halaman Beranda adalah dashboard pemantauan performa. Dasbor ini secara spesifik menampilkan ringkasan kinerja dari model AI yang sedang aktif. Informasi disajikan melalui: kartu metrik utama yang menunjukkan nilai Akurasi, Presisi, Recall, dan F1-Score, grafik garis 'Peningkatan Akurasi per Iterasi' untuk melacak kemajuan model setelah setiap sesi pelatihan ulang, serta diagram radar yang memvisualisasikan metrik performa model.

## Mockup Model Ai

*Gambar 3.6 Tampilan Fitur Model AI*

Halaman Model AI berfungsi sebagai pusat manajemen untuk semua model deep learning dalam sistem. Admin dapat melihat daftar seluruh model yang tersedia dalam format tabel, lengkap dengan informasi performa dan statusnya (Aktif atau Tidak Aktif). Pada halaman ini, Admin dapat melakukan dua aksi utama: masuk ke halaman konfigurasi untuk model tertentu atau menghapus model dari system.

## Mockup Konfigurasi Model

*Gambar 3.7 Tampilan Konfigurasi Model AI*

Halaman Konfigurasi Model AI menyediakan antarmuka untuk mengelola siklus hidup model. Terdapat fitur utama yaitu latih model untuk memulai proses pelatihan ulang pada model yang sudah ada dengan menggunakan data set baru. Selain itu, terdapat panel "Sesi Pelatihan yang Aktif" yang menampilkan progres real-time dari proses pelatihan yang sedang berjalan.

## Mockup Klasifikasi

*Gambar 3.8 Tampilan Klasifikasi*

Halaman Klasifikasi memiliki 2 fungsi utama yaitu upload gambar yang menyediakan fungsionalitas untuk mengunggah gambar Gram stain melalui dua cara: memilih dari direktori file lokal atau mengambil gambar secara langsung menggunakan kamera. Setelah gambar berhasil diunggah, Analis dapat menekan tombol "Mulai Klasifikasi" untuk memicu model AI melakukan analisis, serta fungsi Anotasi yang memungkinkan analis mengubah hasil dari model AI.

## Rancangan Skenario Uji Coba

## Tujuan Ujicoba

Menguji efektivitas model CNN dalam mengklasifikasikan citra bakteri Gram positif dan Gram negatif berdasarkan hasil pewarnaan Gram.

Design Uji Coba

Komponen: Rencana

Jenis Klasifikasi: Biner (Gram-positif vs Gram-negatif)

Data Masukan: Citra digital hasil pewarnaan Gram

Jumlah Kelas: 2

## Metode: CNN (dengan dan tanpa transfer learning)

Strategi Validasi: Hold-out validation dengan stratified split (70:15:15)

Skenario Uji Coba

Untuk menjawab pertanyaan riset secara sistem atis, uji coba akan dilakukan dalam beberapa skenario:

*Tabel 3.1 Tabel Skenario Uji Coba*

Nama Skenario

Deskripsi Singkat

## Tujuan

Arsitektur

Augmentasi

Transfer Learning

Fine-Tuning

CNN Sederhana dari Nol

Model CNN custom dibuat dari awal tanpa bobot pre-trained.

Menjadi baseline pembanding model lain.

CNN 4–5 layer konvolusi manual

Tidak

Tidak

–

CNN + Augmentasi

Model CNN sama seperti di skenario 1, tapi ditambah augmentasi data.

Melihat dampak augmentasi terhadap generalisasi model.

CNN manual

Ya

Tidak

–

Transfer Learning (ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3)

Menggunakan ResNet50, ResNet101, EfficientNet-B0, dan EfficientNet-B3 dengan bobot dari ImageNet (tanpa pelatihan awal).

Bandingkan performa arsitektur CNN populer pre-trained.

ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3

Tidak

Tidak

Tanpa fine-tuning (only FC)

Transfer Learning + Fine-tuning 5 Layer

Model seperti skenario 3, tapi dengan membuka 5 lapisan akhir untuk dilatih ulang.

Melihat efek fine-tuning terbatas terhadap performa akhir.

ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3

Tidak

Ya

Ya, 5 lapisan terakhir

Transfer Learning + Augmentasi

Kombinasi skenario 3 & 4, dengan augmentasi + fine-tuning + transfer learning.

Evaluasi kombinasi teknik terbaik yang realistis.

ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, DenseNet121

Ya

Ya

Ya

## Metode Evaluasi

Evaluasi dilakukan di data set uji (10%) menggunakan:

Akurasi

Precision, Recall, F1-Score untuk tiap kelas

- AUC-ROC: area di bawah kurva ROC
## Luaran Uji Coba

## Luarandari uji cobaakanberupa :

Akurasiterbaik daritiap skenario

Visualisasi performa model (grafik loss/accuracy, confusion matrix)

Dokumentasihasil uji coba dan konfigurasimasing-masing skenario

