# BAB 2 KAJIAN PUSTAKA

KAJIAN PUSTAKA

2.1 DESKRIPSI PERMASALAHAN

Identifikasi jenis bakteri Gram-positif dan Gram-negatif melalui pewarnaan Gram secara mikroskopis merupakan prosedur dasar dalam diagnosis laboratorium infeksi bakteri. Perbedaan struktur dinding sel antara kedua kelompok bakteri menghasilkan reaksi warna yang berbeda terhadap pewarnaan: Gram-positif akan tampak ungu karena menahan kristal violet, sedangkan Gram-negatif tampak merah muda setelah menyerap safranin akibat kehilangan warna primer [15]. Klasifikasi Gram ini bukan hanya bersifat morfologis, tetapi juga berimplikasi langsung pada pemilihan antibiotik yang sesuai, karena Gram-negatif cenderung lebih resisten akibat membran luarnya [4].

Meskipun prosedur pewarnaan relatif sederhana, interpretasi hasil pewarnaan masih sangat bergantung pada keahlian visual analis laboratorium. Beberapa faktor seperti intensitas pewarnaan, pencahayaan mikroskop, tingkat fokus, dan kondisi morfologi bakteri dapat memengaruhi akurasi pengamatan [19]. Penelitian oleh Borthakur D. (2025) menyoroti bahwa kondisi citra yang kabur, over-stained, atau under-stained dapat menimbulkan ketidakkonsistenan klasifikasi antarpengamat, terutama pada bentuk basil pleomorfik atau kokus dalam rantai [20].

Kompleksitas semakin meningkat dalam konteks volume sampel yang tinggi, keterbatasan waktu, serta fasilitas laboratorium dengan sumber daya manusia dan alat yang terbatas. Studi terbaru juga menunjukkan bahwa persentase kesalahan interpretasi dapat meningkat secara signifikan ketika pemeriksaan dilakukan secara manual dalam kondisi beban kerja tinggi [17]. Selain itu, kontaminasi visual pada preparat, seperti kehadiran leukosit, debris, atau artefak pewarnaan, dapat memperburuk akurasi identifikasi bakteri target [20].

Situasi tersebut mempertegas bahwa klasifikasi bakteri Gram melalui metode mikroskopis memerlukan pelatihan yang memadai, prosedur pewarnaan yang stabil, serta kondisi observasi yang optimal agar hasilnya dapat diandalkan sebagai dasar keputusan klinis.

2.2 TEORI PENUNJANG

Berikut ini beberapa teori penunjang yang akan digunakan sebagai

referensi dalam melakukan penelitian proyek akhir ini :

Pewarnaan Gram

*Gambar 2.1 Pewarnaan Gram*

Pewarnaan Gram adalah teknik diferensial yang membedakan bakteri berdasarkan komposisi dinding sel [15]. Langkah-langkahnya meliputi pemberian kristal violet, iodine, pelarut (dekolorisasi), dan safranin sebagai pewarna balik [15]. Bakteri Gram-positif memiliki lapisan peptidoglikan tebal yang mengikat kristal violet dan tampil berwarna ungu (biru) di bawah mikroskop [15]. Sebaliknya, bakteri Gram-negatif memiliki peptidoglikan tipis dan membran luar berlipid yang tidak menahan kristal violet; setelah pewarna balik safranin, sel ini tampak merah muda [15]. Perbedaan struktur ini diilustrasikan pada Gambar berikut.

*Gambar 2.2 Perbandingan struktur dinding sel bakteri Gram Positif (kiri) dan Gram Negatif (kanan)*

Klasifikasi ini memiliki implikasi langsung terhadap pemilihan antibiotik, karena bakteri Gram-positif dan Gram-negatif merespons secara berbeda terhadap jenis antibiotik tertentu. Oleh karena itu, identifikasi jenis Gram secara cepat dan akurat sangat penting dalam penanganan medis.

Citra Mikroskopik Digital dan Tantangan Kualitas Gambar

Deteksi objek (object detection) merupakan salah satu domain utama dalam Computer Vision yang bertujuan untuk mengidentifikasi kelas objek tertentu dalam citra digital serta menentukan lokasi tepatnya menggunakan kotak pembatas (bounding box) [1]. Dalam konteks analisis citra mikroskopis medis, deteksi objek memiliki peran yang lebih krusial dibandingkan klasifikasi citra biasa. Hal ini dikarenakan satu bidang pandang (Field of View) mikroskop sering kali memuat puluhan hingga ratusan sel bakteri yang tersebar secara acak. Oleh karena itu, diperlukan algoritma yang mampu memisahkan objek yang diminati (foreground) dari latar belakang (background) dan melakukan lokalisasi instansi bakteri secara individu [2].

Deteksi Objek (Object Detection) pada Citra Medis

Citra hasil pewarnaan Gram memiliki ciri visual khusus: bakteri Gram-positif berwarna ungu/biru dan berbentuk kokus (bulat) atau basil (batang), sedangkan Gram-negatif berwarna merah muda dengan bentuk serupa [15]. Sel kokus sering tersusun berkelompok (misal klaster Staphylococcus) atau berderetan (Streptococcus), sedangkan basil cenderung terpisah atau berantai pendek [12]. Misalnya, Kennenth P Smith. mengklasifikasikan citra menjadi kokus Gram-positif dalam klaster, kokus Gram-positif berantai, dan basil Gram-negatif [12]. Kondisi nyata pada preparat sering kompleks akibat latar belakang berwarna, sel darah putih, atau artefak stain [12]. Hal ini menuntut model mampu mengenali variasi morfologi dan kontras.

*Gambar 2. 3Tampilan deteksi bakteri*

*Gambar 2: Contoh citra mikroskopis pewarnaan Gram (Gram-stain) yang menunjukkan bakteri Gram-positif (ungu) dan Gram-negatif (merah muda). Pada gambar, kokus Gram-positif tampak berwarna ungu, sedangkan basil Gram-negatif berwarna pink [15]. Perhatikan pula keberadaan elemen lain seperti leukosit atau artefak pewarnaan yang menambah kompleksitas citra [12].*

Teknik XAI

Dalam bidang medis, kebutuhan akan Explainable AI (XAI) sangat tinggi karena model deep learning umumnya bersifat “kotak hitam” (black box). Ulasan terkini menyatakan bahwa kemampuan model AI untuk memberikan penjelasan kepada manusia bahkan lebih penting daripada akurasinya dalam aplikasi klinis [25]. XAI memastikan bahwa keputusan model dapat dimengerti oleh dokter atau ahli, sehingga mengurangi isu kepercayaan, bias, dan tanggung jawab hukum.

Grad-CAM (Gradient-weighted Class Activation Mapping) adalah salah satu teknik XAI yang populer di pengolahan citra. Grad-CAM menggunakan gradien target class yang mengalir ke lapisan konvolusi terakhir untuk menghasilkan peta aktivasi (lokalisasi kasar) yang menyoroti area penting pada citra untuk prediksi tersebut [26]. Dengan demikian, Grad-CAM menampilkan basis keputusan model dalam bentuk heatmap, sehingga secara intuitif menunjukkan fitur citra yang memicu klasifikasi tertentu [25]. Keunggulan Grad-CAM dibanding metode sebelumnya adalah tidak memerlukan perubahan arsitektur CNN (berbeda dengan metode CAM lama), sehingga dapat diterapkan pada banyak model CNN yang sudah ada [25]. Selain itu, Grad-CAM telah terbukti memberikan visualisasi yang lebih class-discriminative dan informatif: misalnya, peta warna “lebih hangat” (merah-kuning) menandakan fokus model yang lebih kuat [26]. Dengan kata lain, Grad-CAM secara luas digunakan dalam tugas visi komputer untuk meningkatkan interpretabilitas model CNN di bidang medis.

Dibandingkan dengan teknik XAI lain, seperti LIME atau SHAP, Grad-CAM dipilih karena sifatnya yang spesifik untuk data citra dan model CNN. LIME (Local Interpretable Model-agnostic Explanations) dan SHAP (SHapley Additive exPlanations) bersifat model-agnostik dan sering digunakan pada data terstruktur (misal EHR). Keduanya bekerja dengan mengganggu input dan membangun model linear lokal untuk menjelaskan prediksi. Meskipun berguna pada konteks umum, LIME/SHAP cenderung kurang intuitif untuk citra karena setiap piksel dipertimbangkan terpisah atau secara global, dan sering kali kurang menyoroti struktur spasial keseluruhan. Sebaliknya, Grad-CAM menghasilkan peta aktivasi spasial khusus kelas yang menyoroti wilayah citra yang benar-benar berkontribusi pada keputusan CNN [27]. Hal ini sangat membantu dalam konteks medis karena region yang ditampilkan (misalnya, lokasi koloni bakteri atau fitur morfologi) lebih mudah dipahami secara klinis. Sebagai contoh, meta-analisis XAI di klinik menegaskan bahwa Grad-CAM banyak digunakan pada tugas citra medis, karena “membuat heatmap khusus kelas yang secara visual menunjukkan wilayah input yang paling berpengaruh terhadap prediksi” [27]. Oleh karena itu, Grad-CAM dipandang lebih sesuai untuk aplikasi klasifikasi citra bakteri ini dibandingkan LIME atau SHAP, serta relatif mudah diimplementasikan untuk arsitektur CNN umum seperti DenseNet atau VGG.

YOLO

YOLO adalah keluarga algoritma deteksi objek berbasis Convolutional Neural Network (CNN) yang diperkenalkan pertama kali pada tahun 2016. YOLO merevolusi metode deteksi objek dengan pendekatan one-stage detector. Berbeda dengan metode dua tahap (two-stage) seperti Faster R-CNN yang memisahkan proses proposal wilayah (region proposal) dan klasifikasi, YOLO memandang deteksi objek sebagai masalah regresi tunggal [30].

Algoritma ini memproses keseluruhan citra dalam satu kali evaluasi jaringan (single forward pass), membagi citra menjadi grid S x S, dan memprediksi koordinat bounding box serta probabilitas kelas secara simultan. Pendekatan ini memungkinkan YOLO mencapai kecepatan inferensi yang sangat tinggi (real-time) dengan akurasi yang kompetitif, menjadikannya standar industri untuk aplikasi yang membutuhkan respons cepat [30].

Preproses Data

Sebelum citra dapat digunakan dalam pelatihan model deep learning, perlu dilakukan tahap pra-proses yang mencakup:

- Resize: Mengubah ukuran citra ke dimensi standar (misalnya 224x224 piksel) agar sesuai dengan input layer dari model CNN.
- Denoising: Menghilangkan noise menggunakan teknik filter atau reduksi berbasis gelombang untuk menjaga kejelasan fitur morfologis bakteri.
Normalisasi: Menyusun ulang nilai pixel ke dalam rentang tertentu (misal: 0–1) untuk mempercepat proses konvergensi saat pelatihan model.

Augmentasi Data: Proses memperbanyak variasi data dengan transformasi seperti rotasi, flipping, zooming, atau shifting untuk menghindari overfitting dan memperkaya representasi data [6].

Convolutional Neural Network (CNN)

CNN merupakan arsitektur jaringan saraf tiruan yang dirancang untuk menangani data spasial, khususnya citra. CNN terdiri dari beberapa lapisan utama:

Convolution Layer: Mengekstrak fitur lokal dari citra input.

Pooling Layer: Mengurangi dimensi spasial fitur dan menjaga fitur penting.

Fully Connected Layer: Melakukan klasifikasi berdasarkan fitur yang diekstraksi.

Model CNN terbukti unggul dalam tugas klasifikasi citra medis, termasuk dalam pengenalan pola visual dari hasil pewarnaan Gram [1]. Dalam proyek ini digunakan arsitektur CNN populer seperti:

VGG-16: Arsitektur dengan konvolusi bertumpuk 3x3 yang cocok untuk klasifikasi dua kelas.

DenseNet121: Memungkinkan propagasi fitur antar semua lapisan sehingga efisien dalam pelatihan.

Arsitektur Convolutional Neural Network

Arsitektur CNN secara umum tersusun dari beberapa blok pemrosesan yang bekerja secara bertahap untuk mengubah citra input menjadi representasi fitur yang dapat diklasifikasikan. Pada tahap awal, convolution layer mengekstraksi pola lokal seperti tepi, tekstur, warna, dan bentuk sederhana. Pada citra Gram-stain, pola ini penting karena perbedaan warna ungu dan merah muda, bentuk kokus atau basil, serta distribusi objek bakteri menjadi ciri utama dalam membedakan kelas Gram-positif dan Gram-negatif.

Setelah convolution layer, activation function seperti ReLU digunakan untuk menambahkan sifat non-linear pada jaringan sehingga model mampu mempelajari pola visual yang lebih kompleks. Pooling layer kemudian mengurangi dimensi spasial fitur agar proses komputasi menjadi lebih efisien dan model lebih tahan terhadap perubahan posisi kecil pada citra. Batch normalization dapat digunakan untuk menstabilkan distribusi aktivasi selama pelatihan, sedangkan dropout membantu mengurangi risiko overfitting dengan menonaktifkan sebagian neuron secara acak saat training. Pada bagian akhir, fully connected layer menggabungkan fitur yang telah diekstraksi menjadi prediksi kelas akhir.

Simple CNN

Simple CNN merupakan arsitektur CNN sederhana yang dibangun dari awal (from scratch) tanpa memanfaatkan bobot pretrained, dan digunakan sebagai baseline pada penelitian ini. Arsitektur ini terdiri dari lima convolutional layer dengan channel progression bertahap (3→32→64→128→256→512), masing-masing diikuti oleh Max Pooling 2x2 untuk mereduksi dimensi spasial, serta tiga fully connected layer (25.088→1.024→512→2) pada bagian akhir untuk menghasilkan dua kelas keluaran. Fungsi aktivasi ReLU digunakan pada hidden layer, sedangkan Softmax digunakan pada output layer. Karena tidak menggunakan mekanisme khusus seperti skip connection atau dense connection, seluruh representasi fitur harus dipelajari murni dari data latih yang tersedia, sehingga arsitektur ini cocok dijadikan acuan untuk mengukur kontribusi augmentasi data, pemilihan arsitektur yang lebih kompleks, transfer learning, dan fine-tuning pada skenario berikutnya.

VGG

VGG-16 dan VGG-19 merupakan arsitektur CNN yang menggunakan susunan convolution layer berukuran kecil, yaitu kernel 3x3 dengan stride 1, secara bertumpuk dan diselingi Max Pooling 2x2. VGG-16 terdiri dari 13 convolutional layer dan 3 fully connected layer (total 16 layer dengan bobot), sedangkan VGG-19 menambahkan 3 convolutional layer tambahan sehingga totalnya menjadi 19 layer. Desain ini sederhana dan mudah dipahami karena pola peningkatan kedalaman jaringan dilakukan secara bertahap dan konsisten pada setiap block. Kelebihan VGG adalah struktur layer yang konsisten, sehingga sering digunakan sebagai baseline dalam eksperimen klasifikasi citra dan transfer learning. Namun, sebagian besar dari sekitar 138 juta (VGG-16) dan 143 juta (VGG-19) parameter terkonsentrasi pada fully connected layer di bagian akhir, sehingga VGG membutuhkan sumber daya komputasi dan memori yang besar serta berpotensi mengalami overfitting pada data set berukuran sedang seperti pada penelitian ini.

ResNet

ResNet50 dan ResNet101 menggunakan konsep residual connection atau skip connection, yaitu setiap residual block menghitung output sebagai y = F(x) + x, di mana F(x) adalah transformasi yang dipelajari oleh beberapa convolutional layer dan x adalah input yang diteruskan langsung melalui shortcut connection. Mekanisme ini memungkinkan informasi dan gradien dari layer awal diteruskan langsung ke layer yang lebih dalam tanpa harus melewati seluruh transformasi non-linear, sehingga membantu mengurangi masalah vanishing gradient pada jaringan yang sangat dalam. ResNet50 dan ResNet101 menyusun residual block dalam bentuk bottleneck (kombinasi convolution 1x1, 3x3, dan 1x1) untuk menjaga efisiensi parameter meskipun jaringan sangat dalam. Dalam klasifikasi citra medis, ResNet sering digunakan karena mampu mempelajari fitur kompleks tanpa kehilangan informasi penting dari layer sebelumnya. Perbedaan utama ResNet50 dan ResNet101 terletak pada jumlah residual block pada beberapa stage jaringan, di mana ResNet101 memiliki lebih banyak block (101 layer dengan bobot, sekitar 43,5 juta parameter) dibandingkan ResNet50 (50 layer dengan bobot, sekitar 24,5 juta parameter), sehingga ResNet101 memiliki kapasitas representasi yang lebih besar namun membutuhkan komputasi yang lebih tinggi.

EfficientNet

EfficientNet-B0 dan EfficientNet-B3 menggunakan pendekatan compound scaling, yaitu peningkatan kedalaman (depth), lebar (width), dan resolusi input (resolution) jaringan secara seimbang berdasarkan satu koefisien skala. Tujuan pendekatan ini adalah memperoleh akurasi tinggi dengan penggunaan parameter dan komputasi yang lebih efisien dibandingkan penambahan kedalaman atau lebar jaringan secara terpisah. Blok dasar EfficientNet adalah MBConv (Mobile Inverted Bottleneck Convolution), yang menggunakan depthwise separable convolution untuk mengurangi jumlah komputasi serta squeeze-and-excitation untuk memberi bobot pada kanal fitur yang lebih informatif. EfficientNet-B0 merupakan varian dasar yang lebih ringan dengan sekitar 4,7 juta parameter, sedangkan EfficientNet-B3 memiliki kapasitas lebih besar dengan sekitar 11,5 juta parameter hasil scaling dari EfficientNet-B0. Arsitektur ini relevan untuk sistem klasifikasi citra bakteri karena dapat menyeimbangkan kebutuhan akurasi dan efisiensi inferensi, meskipun pada pelatihan from scratch kapasitas representasinya belum sepenuhnya termanfaatkan tanpa bobot pretrained.

DenseNet

DenseNet121 menggunakan dense connection, yaitu setiap layer dalam satu dense block menerima feature map dari seluruh layer sebelumnya pada block yang sama melalui concatenation (bukan penjumlahan seperti pada ResNet). Mekanisme ini mendorong pemanfaatan kembali fitur (feature reuse) di berbagai tingkat kedalaman dan membantu aliran gradien selama pelatihan, karena setiap layer memiliki akses langsung ke gradien dari loss function melalui dense connection. DenseNet121 terdiri dari empat dense block dengan jumlah layer 6, 12, 24, dan 16, yang dihubungkan oleh transition layer (convolution 1x1 dan average pooling) untuk mengurangi dimensi feature map antar block. Dengan mekanisme feature reuse tersebut, DenseNet121 hanya memiliki sekitar 8 juta parameter, jauh lebih sedikit dibandingkan VGG maupun ResNet101, namun tetap memiliki kedalaman efektif yang besar. DenseNet dapat efektif pada data citra karena fitur dari berbagai tingkat kedalaman jaringan tetap dapat dimanfaatkan oleh layer berikutnya, namun performanya tetap bergantung pada jumlah data, kualitas citra, dan strategi pelatihan yang digunakan.

Perbandingan Arsitektur CNN

Keempat kelompok arsitektur di atas memiliki karakter yang berbeda dalam menyeimbangkan kedalaman jaringan, jumlah parameter, dan mekanisme penanganan vanishing gradient. VGG mengandalkan kedalaman dan kernel kecil yang konsisten namun tanpa mekanisme khusus untuk jaringan dalam, sehingga jumlah parameternya besar dan rentan overfitting. ResNet menambahkan residual connection sehingga jaringan dapat dibuat sangat dalam tanpa kehilangan kemampuan optimasi. EfficientNet menyeimbangkan kedalaman, lebar, dan resolusi melalui compound scaling sehingga efisien secara parameter. DenseNet memanfaatkan dense connection untuk feature reuse sehingga parameter lebih hemat dibandingkan VGG maupun ResNet pada kedalaman yang sebanding. Perbedaan mekanisme ini menjadi dasar analisis ketika arsitektur-arsitektur tersebut dievaluasi pada skenario pelatihan from scratch, transfer learning, maupun fine-tuning untuk klasifikasi citra Gram-stain pada BAB 4.

Transfer Learning

Transfer learning adalah teknik di mana model CNN yang telah dilatih pada data set besar seperti ImageNet digunakan kembali untuk tugas klasifikasi baru. Dengan menyesuaikan lapisan akhir (fine-tuning), model dapat digunakan untuk tugas spesifik seperti klasifikasi bakteri Gram meskipun jumlah data terbatas [6]. Transfer learning mampu mencapai akurasi lebih dari 90% dalam klasifikasi citra Gram-stain [6].

Augmentasi Data

Augmentasi data dilakukan untuk memperkaya keragaman data set secara artifisial melalui transformasi citra [16]. Contohnya termasuk rotasi, flipping horizontal/vertikal, translasi, perubahan kontras atau kecerahan, zoom, dan penambahan noise ringan [16]. Teknik-teknik ini memperkenalkan invarian rotasi, orientasi, pencahayaan, dan skala ke dalam proses pelatihan, sehingga model belajar ciri yang lebih umum [16]. Dengan augmentasi, ukuran efektif data set meningkat, membantu menghindari overfitting dan meningkatkan kemampuan generalisasi model pada data tak terlihat sebelumnya [16].

Evaluasi Model

Kinerja model klasifikasi diukur dengan berbagai metrik. Akurasi adalah proporsi total prediksi yang benar (TP+TN)/jumlah sampel [17]. Presisi adalah rasio prediksi positif benar terhadap semua prediksi positif, sedangkan recall (sensitivitas) adalah rasio prediksi positif benar terhadap semua sampel positif [17]. F1-score merupakan rata-rata harmonis presisi dan recall [17]. Untuk klasifikasi biner, AUC (Area Under ROC Curve) digunakan untuk menilai kemampuan model memisahkan kelas independen terhadap threshold pengambilan keputusan [17]. Misalnya, F1-score didefinisikan sebagai [17].

PyTorch Serving untuk Integrasi Model

Model AI yang dilatih dengan Python dan PyTorch disimpan dalam format Saved Model dan dijalankan menggunakan PyTorch Serving. PyTorch Serving adalah framework produksi untuk menyajikan model pembelajaran mesin secara efisien melalui HTTP/gRPC API [13].

Keuntungan menggunakan PyTorch Serving:

Efisiensi Produksi: Tidak perlu menjalankan ulang Python script setiap kali ingin melakukan inferensi.

Scalability: Dapat digunakan dalam skenario cloud atau edge computing.

Flexibilitas Akses: Backend seperti FastAPI dapat mengakses model melalui endpoint REST.

Backend dengan FastAPI

Untuk menghubungkan frontend dengan model AI, sistem menggunakan FastAPI, yaitu framework berbasis Python yang mendukung pengembangan aplikasi web modern, ringan, dan cross-platform. FastAPI memiliki arsitektur modular, mendukung dependency injection secara bawaan, dan memudahkan pengembangan RESTful API, sehingga sangat cocok untuk membangun sistem klasifikasi medis yang skalabel dan mudah dipelihara [9].

Fungsi FastAPI dalam sistem ini:

Menerima file citra dari pengguna melalui endpoint upload.

Mengirim citra ke PyTorch Serving untuk proses inferensi model AI.

Menerima dan mengolah hasil klasifikasi dari PyTorch Serving.

Menyimpan hasil klasifikasi ke basis data.

Menyediakan endpoint API untuk menampilkan hasil klasifikasi kepada pengguna.

2.3 PENELITIAN TERKAIT

Lightweight Visual Transformers Outperform CNNs for Gram-Stained Image Classification

Penelitian ini memiliki tiga fokus utama, yaitu:

Membandingkan performa arsitektur CNN (ResNet, Conv Ne XT) dengan berbagai model visual transformer ringan (BEi T, Mobile Vi T, Pool Former, Vi T).

Menguji akurasi, efisiensi komputasi, dan ukuran model pada tugas klasifikasi Gram-stain.

Menilai apakah visual transformer dapat menggantikan CNN dalam tugas klasifikasi citra medis ringan.

Model Vision Transformer (seperti ViT) menunjukkan performa terbaik dengan akurasi hingga 98,3%, melampaui CNN konvensional seperti ResNet50. Studi ini juga mengevaluasi efisiensi model pada perangkat edge dan inferensi batch.

Studi ini berbeda dari proyek akhir Anda karena tidak membangun sistem aplikasi atau pipeline klinis. Fokus utama adalah eksperimen model dan pembuktian bahwa arsitektur ViT ringan dapat mengungguli CNN dalam klasifikasi citra Gram-stain.

Utilizing CNN-Based Architecture for Automated Differentiation between Gram-Positive and Gram-Negative Bacteria

Penelitian ini memiliki tiga fokus utama, yaitu:

Mengembangkan arsitektur CNN khusus untuk mengklasifikasikan citra Gram-stain menjadi Gram-positif dan Gram-negatif secara otomatis.

Mengimplementasikan pendekatan end-to-end deep learning untuk memproses data tanpa memerlukan ekstraksi fitur manual.

Mengoptimalkan akurasi dan presisi klasifikasi citra mikroskopis bakteri pewarnaan Gram menggunakan pendekatan CNN binari.

Model dilatih menggunakan data set Gram-stain berskala besar yang mencakup variasi pencahayaan dan morfologi bakteri. CNN yang dibangun dari awal ini disesuaikan untuk mengenali fitur visual spesifik dari bakteri berdasarkan pewarnaan Gram.

Hasil dari penelitian menunjukkan bahwa sistem mampu mencapai akurasi 95,7% dan presisi 96,97%, membuktikan keandalan model dalam membedakan bakteri Gram-positif dan Gram-negatif. Pendekatan ini sangat relevan dengan proposal Project Akhir ini, meskipun proyek Akhir lebih menitikberatkan pada penggunaan beberapa arsitektur pretrained serta integrasi ke dalam sistem berbasis web.

Vision Transformer Based Bacteria Classification Model for Gram-Stained Direct Smear Images

Penelitian ini memiliki tiga fokus utama, yaitu:

Mengembangkan model klasifikasi berbasis Vision Transformer (Vi T-B_16) untuk mendeteksi bakteri Gram-positif, Gram-negatif, dan neutrofil dari citra smear langsung.

Membandingkan performa Vi T dengan CNN lain seperti VGG-16 dan EfficientNet B0.

Meningkatkan akurasi klasifikasi pada data set terbatas menggunakan transfer learning dan teknik augmentasi.

Model ViT-B_16 yang digunakan mampu mencapai akurasi 96%, mengungguli VGG-16 (86%) dan EfficientNet B0 (88%). Model dilatih dengan data set smear langsung beranotasi secara manual, termasuk citra dari tiga kelas.

Berbeda dengan proyek akhir Anda yang mengklasifikasikan hanya dua kelas (Gram+ dan Gram–), penelitian ini mencakup kelas tambahan (neutrofil) dan hanya fokus pada eksperimen arsitektur model tanpa implementasi sistem berbasis web

Rapid Convolutional Neural Networks for Gram-Stained Image Classification at Inference Time on Mobile Devices

Penelitian ini memiliki tiga fokus utama, yaitu:

Mendesain dan mengoptimalkan arsitektur CNN (ResNet50, ResNet101, EfficientNet, VGG, dan DenseNet121, ResNet50, MobileNet) untuk klasifikasi bakteri Gram dari citra Gram-stain.

Menerapkan transfer learning, pruning, dan quantization untuk memperkecil ukuran model dan mempercepat waktu inferensi.

Menguji efektivitas model yang telah dioptimalkan pada perangkat mobile Android untuk mendukung klasifikasi secara cepat dan efisien.

Penelitian ini menggunakan data set citra mikroskopik Gram-stain dan melakukan pelatihan ulang pada arsitektur CNN populer. Model kemudian dikompresi dan diuji pada beberapa perangkat smartphone untuk mengevaluasi kecepatan inferensi dan akurasinya.

Hasil menunjukkan bahwa model CNN yang telah dioptimalkan mampu memproses satu citra Gram-stain dalam waktu kurang dari 0,6 detik, dan tetap mempertahankan akurasi yang sangat tinggi (AUC > 0,98). Penelitian ini berfokus pada penggunaan CNN dalam konteks mobile deployment, berbeda dengan pendekatan berbasis web atau server.

A Novel Framework for the Automated Characterization of Gram-Stained Blood Culture Slides Using a Large-Scale Vision Transformer

Penelitian ini memiliki tiga fokus utama, yaitu:

Mengklasifikasikan citra whole-slide hasil pewarnaan Gram dari kultur darah ke dalam lima kategori (GP kokus kluster, GP rantai, GP batang, GN batang, dan tidak ada bakteri).

Menerapkan model Vision Transformer berskala besar untuk menangani slide citra resolusi tinggi.

Menguji generalisasi model terhadap data dari institusi lain tanpa retraining.

Model Vi T yang digunakan dilatih pada 475 slide Gram-stain beresolusi tinggi. Model berhasil mencapai akurasi 85,8% dan AUC 0,952, serta menunjukkan ketahanan yang baik terhadap domain shift.

Perbedaan utama dengan proyek akhir Anda adalah pada jenis data (whole-slide vs citra mikroskopik biasa), jumlah kelas (>2), dan kebutuhan sumber daya tinggi. Selain itu, penelitian ini tidak menyertakan sistem backend maupun web interface.

*Tabel 2.1 Tabel Perbandingan*

Penelitian

Arsitektur CNN yang digunakan

Fokusan

Perbedaan

Lightweight Visual Transformers Outperform CNNs for Gram-Stained Image Classification

Beberapa model Transformer (BEi T, Dei T, Mobile Vi T, Pool Former, Swin, Vi T) dibandingkan dengan CNN (ResNet, Conv Ne XT)

Analisis komparatif performa model Vision Transformer (VT) vs CNN dalam klasifikasi citra Gram-stain.

Proyek akhir ini hanya menggunakan arsitektur CNN dan ingin menerapkan sistem berbasis web, sedangkan studi ini menunjukkan keunggulan VT secara eksperimental.

Utilizing CNN-Based Architecture for Automated Differentiation between Gram-Positive and Gram-Negative Bacteria

ResNet50 (tanpa lapisan atas) + beberapa lapisan tambahan CNN

Klasifikasi otomatis citra Gram-stain bakteri menjadi Gram-positif vs Gram-negatif menggunakan CNN.

Menggunakan arsitektur turunan ResNet50 (bukan DenseNet/ VGG/ Inception seperti proyek akhir ini), Tidak dibahas integrasi sistem web.

Vision Transformer Based Bacteria Classification Model for Gram-Stained Direct Smear Images

Vision Transformer (Vi TB_16)

Klasifikasi bakteri (Gram-positif kokus, Gram-negatif basil) dan neutrofil pada citra smear Gram langsung menggunakan Vision Transformer.

Memasukkan kelas neutrofil serta bakteri, bukan hanya Gram+ vs Gram–. Menggunakan ViT dan teknik segmentasi (GMM-EM) sebelum klasifikasi, berbeda dari CNN konvensional proyek akhir ini. Pendekatan menekankan transfer learning untuk jumlah data kecil, tanpa integrasi web, sementara project akhir ini berfokus klasifikasi dua kelas bakteri dengan CNN untuk integrasi web.

Rapid Convolutional Neural Networks for Gram-Stained Image Classification at Inference Time on Mobile Devices

Inception v3, ResNet50, MobileNet (pre-trained; dituning lalu dipruning/quantisasi)

Optimasi klasifikasi citra Gram-stain untuk inferensi cepat pada perangkat mobile melalui transfer learning, pruning, dan quantization.

Fokus pada mobilitas & kompresi model: bertujuan deploy ke smartphone (inference cepat, ukuran kecil), Proyek Akhir ini menargetkan sistem web, tidak memfokuskan optimisasi ukuran. Selain itu, mereka menggunakan arsitektur Inception/ResNet/MobileNet, sedangkan proyek akhir ini menggunakan DenseNet121/ VGG-16/ ResNet50, ResNet101, EfficientNet, VGG, dan DenseNet121.

·A Novel Framework for the Automated Characterization of Gram-Stained Blood Culture Slides Using a Large-Scale Vision Transformer

Vision Transformer berskala besar (masing-masing patch besar)

Klasifikasi otomatis whole-slide citra Gram-stain kultur darah (5 kategori: GP kokus kluster, GP kokus rantai, GP batang, GN batang, tanpa bakteri) menggunakan transformer berskala besar.

Melakukan klasifikasi multi-kelas (GP kelompok, GP pasang, GP rantai, GN, jamur, dll.) dengan sistem pemindaian otomatis, sedangkan project akhir ini menggunakan klasifikasi biner (Gram positif dan negatif) dengan antarmuka web.

