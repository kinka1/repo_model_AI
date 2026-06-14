# BAB 4 EKSPERIMEN DAN ANALISIS

## EKSPERIMEN DAN ANALISIS

Pada bab ini merupakan tahap uji coba dan analis a dari sistem yang telah dibuat. Eksperimen ini bertujuan guna mengetahui apakah sistem sudah berjalan sesuai dengan target awal Berdasarkan design sistem yang telah dirancang.

## PARAMETER EKSPERIMEN

Dalam penelitian ini, terdapat beberapa parameter yang digunakan untuk melakukan eksperimen. Parameter dan nilai-nilai yang diuji dapat dilihat pada Tabel 4.1.

*Tabel 4. 1Tabel Parameter Eksperiment*

No

## Parameter

Nilai

1

Dataset

Gram Bacteria Classification

2

Epoch

50-80

3

Total Image

11,824

4

Learning Rate

0.001

5

Optimizer

Adam

## KARAKTERISTIK DATA

*Gambar 4. 1 Gambar bakteri gram positif*

Dalam melakukan eksperimen sistem klasifikasi bakteri Gram menggunakan deep learning, data yang digunakan diperoleh dari gambar mikroskopi hasil pewarnaan Gram yang dikumpulkan dari berbagai sumber laboratorium mikrobiologi. Karakteristik data ini meliputi gambar bakteri yang telah melalui proses pewarnaan Gram standar, dengan pencahayaan mikroskop yang memadai dan resolusi yang cukup untuk mendeteksi morfologi bakteri secara akurat. Data dikumpulkan dari slide mikroskop yang telah disiapkan oleh ahli mikrobiologi, dimana setiap gambar telah melalui proses labeling oleh tenaga ahli untuk memastikan akurasi klasifikasi.

## Karakteristik data meliputi:

Sumber Data: Gambar mikroskopi hasil pewarnaan Gram dari laboratorium mikrobiologi klinis

Resolusi Asli: Bervariasi sesuai kameramikroskop (di-resize menjadi 224×224 piksel saat preprocessing)

Pencahayaan: Pencahayaan mikroskop standar dengan variasi intensitas pewarnaan

Morfologi: Gambar menangkap berbagai bentuk bakteri (kokus, basil, spiril) dengan karakteristik pewarnaan Gram Positif (ungu) dan Gram Negatif (merah/pink)

Kualitas: Gambar dengan rentang ukuran file 4,82 KB - 13,04 KB (rata-rata 8,12 KB)

Dataset yang telah dikumpulkan mencakup dua kelas yang relevan untuk mendukung sistem klasifikasi bakteri Gram. Kedua kelas tersebut meliputi Gram Positif dan Gram Negatif. Dataset dirancang sedemikian rupa agar dapat dilatih menjadi sebuah model yang mampu mengklasifikasikan bakteri berdasarkan hasil pewarnaan Gram secara akurat. Dalam penelitian ini, data set yang digunakan terdiri dari 11.824 gambar yang telah dilabeli oleh ahli mikrobiologi. Seluruh gambar tersebut memuat objek bakteri dari dua kelas yang relevan untuk mendukung sistem klasifikasi otomatis.

Class

Jumlah

Gram Positif

9.170

Gram Negatif

2.654

*Tabel 4. 2 Tabel Klasifikasi pada Dataset*

Berdasarkan Tabel 4.1, dapat diidentifikasi bahwa data set memiliki ketidak seimbangan kelas (class imbalance) dengan rasio 3,45:1 antara kelas Gram Negatif dan Gram Positif. Kondisi ini merupakanrefleksi dari distribusi natural sampel bakteri di laboratorium mikrobiologi, dimana bakteri Gram Negatif umumnya lebih banyak ditemukan dalam praktik klinis.

Dataset yang telah dikumpulkan dibagi menjadi tiga bagian yaitu train set, validation set, dan test set, dengan perbandingan 70% : 15% : 15%. Pembagian ini menggunakan metode stratified split untuk memastikan distribusi kelas yang proporsional pada setiap subset. Pembagian ini bertujuan untuk memastikan proses pelatihan model, validasi performa selama pelatihan, serta evaluasi akhir model dapat dilakukan secara optimal.

Gram Positif

Gram Negatif

Persentase

Train

6.418

1.858

70%

Val

1.376

398

15%

Test

1.376

398

15%

Total

9.170

2.654

100%

*Tabel 4. 3 Tabel Pembagian Dataset Penelitian*

Pada data set imbalance, data set condong ke bakteri gram negatif dengan perbandingan 1:3,45 antara bakteri gram positive dan negative. Untuk mengatasi hal tersebut, bisa menggunakan dua strategi utama yaitu :

Penggunaan Class Weights

Implementasi weighted loss function dengan bobot yang dihitung berdasarkan inverse frequency:

Dimana N adalah total sampel, nc adalah jumlah kelas, dan Ni adalah jumlah sampel kelas i.

## Hasil perhitungan:

Bobot untuk kelas Gram Negatif (mayoritas): 0,65

Bobot untuk kelas Gram Positif (minoritas): 2,23

Pemberian bobot yang lebih tinggi pada kelas minoritas bertujuan agar model tidak bias dalam memprediksi kelas mayoritas dan memberikan penalti lebih besar untuk kesalahan pada kelas Gram Positif.

Sharpness-based Filtering

Pada skenario baseline, dilakukan pengurangan sampel kelas Gram Negatif menggunakan metode sharpness-based filtering untuk menyeimbangkan jumlah sampel dengan kelas Gram Positif. Metode ini memilih sampel Gram Negatif dengan kualitas terbaik berdasarkan nilai Laplacian variance sebagai indikator ketajaman gambar, sehingga hanya gambar dengan kualitas visual optimal yang dipertahankan dalam data set training yang seimbang. Contoh code-nya :

sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

Yang mana library memiliki sebuah rumus :

Dimana :

xi adalah nilai pixel Laplacian ke-i

adalah mean dari semua nilai Laplacian

adalah total pixel

TEMPAT UJICOBA

WAKTU UJICOBA

Uji coba penelitian ini berlangsung dalam rentang waktu dari bulan Oktober 2025 hingga Desember 2025.

SPESIFIKASI PERALATAN UJICOBA

Pada penelitian ini membutuhkan sistem perangkat keras dan perangkat lunak dalam proses perancangan sistem deteksi kecurangan sampai dengan implementasinya. Adapun spesifikasi dari perangkat keras pada Tabel

*Tabel 4. 4 Tabel Spesifikasi*

No

Deskripsi

Spesifikasi

1

Laptop

LENOVO Idea Pad Gaming 3 15ACH6

2

CPU

AMD Ryzen 7 5800H

3

RAM

24 GB

4

SSD

512 GB

5

Graphic Card

NVIDIA Ge Force RTX 3050

6

Operating System

Windows 11 Home

7

Software

Visual Studio Code, Python,

## HASIL EKSPERIMEN

Pembagian Data

Pembagian data set adalah tahap penting untuk memastikan bahwa model yang dikembangkan dapat belajar dengan baik dari data latih dan data evaluasi. Dataset pada penelitian ini berjumlah total 9170 gambar, yang dibagi menjadi tiga subset: train, validation, dan test. Pembagian data dilakukan dengan skala 70% : 15% : 15% untuk memastikan distribusi data yang optimal sesuai dengan kebutuhan penelitian.

Pre-Processing Data

Pada tahap pre-processing data set, langkah utama yang diterapkan untuk memastikan kualitas dan konsistensi data sebelum digunakan dalam pelatihan model meliputi empat proses sistematis yang dirancang untuk mengoptimalkan kualitas input model. Tahapan Pre-processing :

Ekstraksi Region of Interest (ROI)

Isolasi area bakteri dari gambar mikroskopi menggunakan koordinat polygon yang telah dianotasi oleh ahli mikrobiologi. Proses ini menggunakan metode bounding box untuk mengidentifikasi area mini mal yang mencakup seluruh struktur bakteri, memastikan fokus model pada region yang mengandung informasi klasifikasi penting.

Penambahan Padding

Setelah ekstraksi ROI, ditambahkan padding sebesar 10% dari dimensi bounding box di setiap sisi. Padding ini bertujuan untuk memberikan konteks visual di sekitar bakteri dan menghindari kehilangan informasi penting pada boundary, serta mencegah cropping yang terlalu ketat yang dapat menghilangkan informasi morfologi penting.

Resize dan Normalisasi Dimensi

Seluruh gambar diresize menjadi ukuran standar 224×224 piksel menggunakan interpolasi bilinear untuk menjaga kualitas visual. Ukuran ini dipilih karena:

Kompatibel dengan arsitektur pre-trained model (ResNet50, ResNet101, EfficientNet)

Merupakan standar input untuk model yang telah dilatih pada data set ImageNet

Memungkinkan batch processing yang efisien

Normalisasi Intensitas Piksel

Nilai piksel dinormalisasi menggunakan nilai mean dan standard deviation dari data set ImageNet:

Mean: [0.485, 0.456, 0.406]

Standard Deviation: [0.229, 0.224, 0.225]

Normalisasi ini penting untuk transfer learning karena model pre-trained telah dilatihdengan normalisasi yang sama, memastikan distribusi input yang konsistendengan ekspektasi model.

Augmentasi Data

Pada tahap augmentasi, berbagai tekniktransformasi diterapkan untuk meningkatkan variasi dan jumlah data. Augmentasi data bertujuan untuk mengatasi overfitting dengan memperkenalkan variasi yang menyerupai kondisi nyata. Dapat dilihat kombinasi teknik augmentasi data pada table

*Tabel 4. 5 table augmentasi*

Augmentasi

Sebelum

Sesudah

Flip

Rotation (30)

Color jitter (Brightness, contrast, saturation, hue)

Resize (256 -> 244)

Skenario Eksperimen

Penelitianini mengimplementasikan lima skenario eksperimen dengan konfigurasi yang berbeda untuk mengevaluasi berbagai pendekatan deep learning dalam klasifikasi bakteri Gram. Setiap skenario dirancang untuk mengisolasi dan mengukur kontribusi spesifik dariteknik-teknik seperti data augmentation, transfer learning, dan fine-tuning.

*Tabel 4. 6 Tabel Skenario*

Skenario

Arsitektur

Augmentasi

Transfer Learning

Fine-tuning

Epochs

1

Simple CNN

X

X

X

40

2

Simple CNN

V

X

X

40

3

Rest Net50, ResNet101,

EfficientNet B0,

EfficientNet B3

X

X

X

50

4

Rest Net50, ResNet101,

EfficientNet B0,

EfficientNet B3

X

V

V

80

5

Rest Net50, ResNet101,

EfficientNet B0,

EfficientNet B3, VGG16, VGG19,

Densenet121

V

V

V

80

## Skenario 1: Baseline - Simple CNN from Scratch

Skenario ini mengimplementasikan arsitektur Convolutional Neural Network (CNN) sederhana yang dibangun dari awal (from scratch) tanpa menggunakan pre-trained weights. Arsitektur ini berfungsi sebagai baseline untuk membandingkan performa dengan skenario lainnya.

Simple CNN digunakan sebagai baseline karena arsitekturnya lebih sederhana dibandingkan model CNN standar seperti ResNet, EfficientNet, VGG, dan DenseNet. Dengan baseline ini, performa awal sistem dapat diamati tanpa pengaruh bobot pretrained atau desain arsitektur yang kompleks. Hasil dari skenario ini menjadi acuan untuk menilai seberapa besar peningkatan performa yang diperoleh dari augmentasi data, pemilihan arsitektur CNN yang lebih kuat, transfer learning, dan fine-tuning.

Spesifikasi Arsitektur :

Convolutional Layers: 5 layerdengan channel progression 3→32→64→128→256→512

Pooling: Max Pooling 2×2 setelah setiap convolutional layer

Fully Connected: 3 dense layers (25.088→1.024→512→2)

Aktivasi: Re LUuntuk hidden layers, Softmaxuntuk output layer

Implementasicodenya

*Gambar 4. 2 diagram arsitektur simple cnn*

Saya menggunakan library untuk Convolutional layer untukekstraksi fitur spatial denganparameter :

In_chanels :jumlahchanel input

Out_chanels: jumlah filter/feature maps

Kernel_size=3: filter 3x3 (standar untuk detail features)

Padding = 1 : zero-pading 1 pikseluntuk maintain spatial size

Komputasidengan convolution 2D dengan learnable kernels yang mana nantiakanmemiliki output feature maps yang mendekati patterns

## Skenario 2: CNN dengan Data Augmentation

Skenario ini menggunakan arsitektur yang identik dengan Skenario 1, namun dengan penambahan teknik augmentasi data untuk meningkatkan variasi training sampels dan kemampuan generalisasi model.

Pada skenario ini, augmentasi digunakan untuk memperkaya variasi citra tanpa menambah data asli baru. Teknik seperti rotasi, flipping, dan perubahan warna relevan untuk citra mikroskopis karena posisi bakteri, orientasi objek, dan pencahayaan dapat berbeda antar sampel. Dengan demikian, model diharapkan tidak hanya menghafal pola dari data latih, tetapi juga belajar fitur yang lebih stabil terhadap variasi visual.

Teknik Augmentasi:

Random Rotation (±30°)

Random Horizontal Flip (p=0,5)

Color Jitter (brightness, contrast, saturation: 0,2)

## Skenario 3: Transfer Learning dengan ResNet50, ResNet101, EfficientNet b0, dan EfficientNet b3

Kelompok Skenario 3 menggunakan pendekatan transfer learning tanpa fine-tuning: seluruh backbone jaringan yang telah di-pretrain pada ImageNet dibekukan (frozen), dan hanya fully connected layer (classifier) yang dilatih ulang untuk tugas klasifikasi bakteri Gram.

Pada pendekatan ini, backbone CNN digunakan sebagai feature extractor. Layer-layer awal hingga tengah dari model pretrained mempertahankan bobot ImageNet untuk mengekstraksi fitur umum seperti tepi, tekstur, dan pola bentuk. Bagian classifier diganti agar sesuai dengan jumlah kelas pada penelitian ini, yaitu Gram-positif dan Gram-negatif. Strategi ini mengurangi kebutuhan komputasi karena hanya classifier yang dilatih ulang.

Pada penelitian ini, empat backbone berbeda dievaluasi:

## Skenario 3a: ResNet50

## Skenario 3b: ResNet101

## Skenario 3c: EfficientNet-B0

## Skenario 3d: EfficientNet-B3

Keempat skenario menggunakan pola yang sama: backbone dibekukan, kemudian classifier baru (fully connected) ditambahkan di bagian akhir dan hanya bagian ini yang dilatih.

Perbandingan arsitektur pada skenario ini bertujuan untuk melihat kemampuan representasi fitur dari masing-masing backbone. ResNet mengandalkan residual connection untuk menjaga aliran informasi pada jaringan yang dalam, sedangkan EfficientNet menggunakan compound scaling untuk menyeimbangkan akurasi dan efisiensi. Perbedaan karakter arsitektur ini menjadi dasar analisis ketika hasil akurasi antar model menunjukkan performa yang berbeda.

Spesifikasi Arsitektur dan Kompleksitas Model Skenario 3:

*Tabel 4. 7 Tabel Spesifikasi Arsitektur*

Sub skenario

Backbone

Total parameter

## Parameter terlatih

Persentaseterlatih

3a

ResNet50

24.558.146

1.050.114

± 4,3%

3b

ResNet50

43.550.274

1.050.114

± 2,4%

3c

EfficientNet-B0

4.664.446

656.898

± 14,1%

3d

EfficientNet-B3

11.484.202

787.970

± 6,9%

Secara umum, Skenario 3 mengeksplorasi trade-off antara kompleksitas backbone dan efisiensi parameter terlatih. ResNet101 memiliki total parameter terbesar, sedangkan EfficientNet-B0 adalah model paling ringan namun dengan proporsi parameter terlatih yang relatiftinggi pada classifier.

Konfigurasi Training (berlaku untuk 3a–3d):

- Optimizer: Adam dengan learning rate 0,001
Maximum Epochs: 50

Tes Percobaan

Dari scenario di atas menghasilkan sebuah model yang akan dites dengan 10 citra mikroskopis di luar dari data set yang tersedia.

Skenario 1

## Hasil pengujian pada setiap data disajikan pada Gambar 4.2 dan Tabel 4.8 berikut

*Gambar 4.3 Grafik data akurasi dengan model Simple CNN*

*Tabel 4.8 Data Akurasi dengan Model Simple CNN*

| No. | Data Uji | Akurasi |
|---:|---|---:|
| 1 | Image 1 | 0.633 |
| 2 | Image 2 | 0.712 |
| 3 | Image 3 | 0.658 |
| 4 | Image 4 | 0.701 |
| 5 | Image 5 | 0.786 |
| 6 | Image 6 | 0.754 |
| 7 | Image 7 | 0.679 |
| 8 | Image 8 | 0.685 |
| 9 | Image 9 | 0.697 |
| 10 | Image 10 | 0.716 |

Berdasarkan data pada Gambar 4.2 dan Tabel 4.8, model Simple CNN yang dibangun dari awal menunjukkan performa yang masih bervariasi antar citra uji. Akurasi tertinggi diperoleh pada Image 5 dengan nilai 0,786 (78,6%), diikuti oleh Image 6 sebesar 0,754 (75,4%), Image 10 sebesar 0,716 (71,6%), dan Image 2 sebesar 0,712 (71,2%). Sebaliknya, akurasi terendah tercatat pada Image 1 dengan nilai 0,633 (63,3%), diikuti Image 3 sebesar 0,658 (65,8%). Rentang akurasi antara nilai tertinggi dan terendah adalah 0,153 atau 15,3 poin persentase, yang menunjukkan bahwa model baseline masih sensitif terhadap variasi kualitas dan karakteristik citra.

Keterbatasan Simple CNN merupakan konsekuensi dari pelatihan from scratch tanpa bobot pretrained. Seluruh representasi fitur harus dipelajari dari data yang tersedia, sehingga kemampuan ekstraksi fitur model masih terbatas dibandingkan arsitektur yang lebih dalam atau model yang memanfaatkan transfer learning. Meskipun arsitektur ini telah dilengkapi convolution layer, max pooling, Batch Normalization, dan Dropout, Simple CNN tetap berfungsi terutama sebagai baseline untuk menilai peningkatan performa pada skenario berikutnya.

## Skenario 2

## Hasil pengujian pada setiap data disajikan pada Gambar 4.3 dan Tabel 4.9 berikut

*Tabel 4.9 Data Akurasi Simple CNN dengan Augmentasi*

| No. | Data Uji | Akurasi (+Augmentasi) | Akurasi (Tanpa Aug.) | Perubahan |
|---:|---|---:|---:|---:|
| 1 | Image 1 | 0.759 | 0.633 | +12.6% |
| 2 | Image 2 | 0.813 | 0.712 | +10.1% |
| 3 | Image 3 | 0.782 | 0.658 | +12.4% |
| 4 | Image 4 | 0.752 | 0.701 | +5.1% |
| 5 | Image 5 | 0.803 | 0.786 | +1.7% |
| 6 | Image 6 | 0.741 | 0.754 | -1.3% |
| 7 | Image 7 | 0.713 | 0.679 | +3.4% |
| 8 | Image 8 | 0.726 | 0.685 | +4.1% |
| 9 | Image 9 | 0.743 | 0.697 | +4.6% |
| 10 | Image 10 | 0.764 | 0.716 | +4.8% |

Berdasarkan Tabel 4.9, penerapan augmentasi data memberikan peningkatan pada sebagian besar citra uji. Peningkatan terbesar terjadi pada Image 1 sebesar 12,6 poin persentase dan Image 3 sebesar 12,4 poin persentase. Image 2 menjadi data uji dengan akurasi tertinggi setelah augmentasi, yaitu 0,813 (81,3%), diikuti Image 5 sebesar 0,803 (80,3%) dan Image 3 sebesar 0,782 (78,2%). Satu penurunan terjadi pada Image 6, yaitu dari 0,754 menjadi 0,741, yang menunjukkan bahwa augmentasi tidak selalu meningkatkan performa untuk semua karakteristik citra.

Secara analitis, augmentasi membantu model mempelajari fitur yang lebih tahan terhadap perubahan orientasi, posisi, dan pencahayaan. Hal ini relevan pada citra mikroskopis karena objek bakteri dapat muncul dengan sudut, intensitas warna, dan kualitas fokus yang berbeda. Rentang akurasi setelah augmentasi berada pada 0,713 sampai 0,813, lebih sempit dibandingkan skenario tanpa augmentasi, sehingga performa model menjadi lebih stabil pada data uji yang beragam.

## Skenario 3

Pada scenario ini saya melakukan uji coba dengan menggunakan beberapa arsitektur CNN yang telah ada seperti ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121 dari awal (training from scratch), dan memiliki hasil seperti ini.

*Gambar 4.4 Grafik Data Akurasi Model CNN yang Dilatih dari Awal (From Scratch)*

Berdasarkan grafik pada Gambar 4.4, dapat dilihat perbandingan nilai akurasi dari tujuh arsitektur CNN yang dilatih dari awal. Data lengkap hasil evaluasi dirangkum dalam Tabel 4.10 berikut ini.

*Tabel 4.10 Data Akurasi Model CNN yang Dilatih dari Awal (From Scratch)*

| No. | Arsitektur Model | Akurasi |
|---:|---|---:|
| 1 | ResNet50 | 0.921273 |
| 2 | ResNet101 | 0.926173 |
| 3 | EfficientNet-B0 | 0.838358 |
| 4 | EfficientNet-B3 | 0.839127 |
| 5 | VGG-16 | 0.808688 |
| 6 | VGG-19 | 0.825571 |
| 7 | DenseNet121 | 0.782712 |

Berdasarkan data pada Gambar 4.4 dan Tabel 4.10, ResNet101 memperoleh akurasi tertinggi sebesar 0,926173 (92,62%), diikuti oleh ResNet50 sebesar 0,921273 (92,13%). Keunggulan kedua model ResNet menunjukkan bahwa residual connection efektif membantu proses pembelajaran ketika model dilatih dari awal. Mekanisme skip connection memungkinkan aliran informasi dan gradien tetap stabil pada jaringan yang dalam, sehingga proses optimasi dapat berjalan lebih baik meskipun bobot awal bersifat acak.

DenseNet121 memperoleh akurasi terendah sebesar 0,782712 (78,27%), sedangkan VGG-16, VGG-19, EfficientNet-B0, dan EfficientNet-B3 berada pada rentang 0,808688 sampai 0,839127. Pada skenario from scratch, arsitektur yang memiliki desain efisien atau koneksi padat tidak otomatis menghasilkan performa terbaik karena seluruh representasi fitur harus dipelajari dari data penelitian. Rentang antara model terbaik dan terendah adalah 0,143461 atau sekitar 14,35 poin persentase, yang menunjukkan bahwa pemilihan arsitektur sangat berpengaruh terhadap performa klasifikasi.

Analisis Pemilihan Arsitektur CNN

Berdasarkan hasil pelatihan from scratch, arsitektur ResNet menunjukkan performa yang lebih stabil dibandingkan arsitektur lain. Hal ini berkaitan dengan residual connection yang membantu proses pembelajaran pada jaringan dalam, terutama ketika model harus mempelajari representasi fitur dari bobot acak. Pada data set citra Gram-stain, fitur pembeda seperti warna, tekstur, bentuk bakteri, dan variasi pencahayaan perlu dipelajari secara bertahap. ResNet mampu mempertahankan aliran informasi dan gradien sehingga proses optimasi menjadi lebih efektif.

EfficientNet memiliki desain yang efisien melalui compound scaling, tetapi pada pelatihan from scratch performanya belum melampaui ResNet. Hal ini menunjukkan bahwa efisiensi arsitektur tidak selalu langsung menghasilkan akurasi tertinggi ketika data latih terbatas atau ketika model belum menggunakan bobot pretrained. Sementara itu, VGG memiliki struktur yang sederhana namun jumlah parameter besar, sehingga membutuhkan data dan regularisasi yang cukup agar tidak mudah overfitting. DenseNet secara teori mendukung feature reuse, tetapi performanya pada eksperimen ini menunjukkan bahwa koneksi padat antar layer tetap memerlukan strategi pelatihan dan jumlah data yang memadai untuk mencapai hasil optimal.

## Skenario 4

Pada scenario ini saya melakukan uji coba dengan menggunakan arsitektur seperti pada scenario ke-3, akan tetapi di sini setiap model dilakukan fine-tuning.

*Gambar 4.5 Grafik Data Akurasi Model CNN yang Dilatih dan Dilakukan Fine-Tuning*

*Tabel 4.11 Data Akurasi Model CNN yang Dilatih dan Dilakukan Fine-Tuning*

| No. | Arsitektur Model | Akurasi |
|---:|---|---:|
| 1 | ResNet50 | 0.941567 |
| 2 | ResNet101 | 0.918219 |
| 3 | EfficientNet-B0 | 0.868261 |
| 4 | EfficientNet-B3 | 0.876251 |
| 5 | VGG-16 | 0.789251 |
| 6 | VGG-19 | 0.852618 |
| 7 | DenseNet121 | 0.797219 |

Berdasarkan data pada grafik dan tabel hasil eksperimen, pendekatan transfer learning yang dikombinasikan dengan fine-tuning pada berbagai arsitektur CNN menunjukkan hasil yang beragam. ResNet50 menjadi model dengan performa terbaik pada skenario ini dengan akurasi 0,941567 (94,16%), diikuti oleh ResNet101 sebesar 0,918219 (91,82%). Hal ini menunjukkan bahwa struktur residual learning pada ResNet mampu memanfaatkan inisialisasi bobot pretrained secara efektif untuk mencapai konvergensi yang lebih baik.

EfficientNet-B3 memperoleh akurasi 0,876251 (87,63%) dan EfficientNet-B0 memperoleh akurasi 0,868261 (86,83%). Keduanya menunjukkan performa yang cukup stabil, tetapi belum melampaui seri ResNet pada skenario fine-tuning ini. VGG-16 dan DenseNet121 menjadi dua model dengan akurasi terendah, yaitu 0,789251 (78,93%) dan 0,797219 (79,72%). Hasil ini menunjukkan bahwa penggunaan bobot pretrained tidak selalu memberikan peningkatan yang sama pada setiap arsitektur, karena efektivitas fine-tuning tetap dipengaruhi oleh desain jaringan, jumlah parameter, dan kemampuan arsitektur dalam menyesuaikan fitur terhadap data set Gram-stain.

## Skenario 5

Pada scenario ini, tujuh arsitektur CNN yang sama dilatih menggunakan pendekatan transfer learning disertai dengan proses fine-tuning. Berbeda dengan pendekatan from scratch, metode ini memanfaatkan bobot pretrained yang telah dipelajari dari data set berskala besar (ImageNet) sebagai titik awal, kemudian melakukan penyesuaian bobot secara menyeluruh terhadap data set target melalui proses fine-tuning. Hasil evaluasi akurasi dari seluruh model disajikan pada Gambar 4.6 dan Tabel 4.12 berikut.

*Gambar 4.6 Grafik Data Akurasi Model CNN dengan Transfer Learning dan Fine-Tuning*

*Tabel 4.12 Data Akurasi Model CNN dengan Transfer Learning dan Fine-Tuning*

| No. | Arsitektur Model | Akurasi |
|---:|---|---:|
| 1 | ResNet50 | 0.951663 |
| 2 | ResNet101 | 0.968832 |
| 3 | EfficientNet-B0 | 0.879411 |
| 4 | EfficientNet-B3 | 0.984627 |
| 5 | VGG-16 | 0.808688 |
| 6 | VGG-19 | 0.875571 |
| 7 | DenseNet121 | 0.809248 |

Berdasarkan data pada Gambar 4.6 dan Tabel 4.12, pendekatan transfer learning yang dikombinasikan dengan fine-tuning menghasilkan peningkatan performa pada sebagian besar arsitektur dibandingkan pelatihan from scratch. Hal ini mengonfirmasi bahwa pemanfaatan bobot pretrained dari data set berskala besar memberikan fondasi representasi fitur yang lebih kuat, sehingga model dapat belajar fitur-fitur yang lebih diskriminatif dari data set target dengan lebih efisien.

Arsitektur EfficientNet-B3 menjadi model dengan performa terbaik pada skenario ini, mencatatkan akurasi tertinggi sebesar 0,9846 (98,46%), melonjak drastis dari 0,8391 (83,91%) pada skenario from scratch, atau meningkat sebesar 14,55 poin persentase. Peningkatan luar biasa ini menunjukkan bahwa EfficientNet-B3 memiliki kapasitas representasi fitur yang sangat tinggi yang baru dapat dioptimalkan secara penuh ketika diinisialisasi dengan bobot pretrained. Posisi kedua ditempati oleh ResNet101 dengan akurasi 0,9688 (96,88%), diikuti ResNet50 dengan 0,9517 (95,17%), yang juga mengalami peningkatan signifikan masing-masing sebesar 4,27 dan 3,04 poin persentase dari skenario sebelumnya.

Di sisi lain, meskipun sebagian besar model mengalami peningkatan, arsitektur VGG-16 tidak menunjukkan perubahan akurasi dan tetap berada di angka 0,808688 (80,87%), identik dengan nilai yang diperoleh pada skenario from scratch. Fenomena ini mengindikasikan bahwa arsitektur VGG-16 kemungkinan mengalami kendala dalam proses fine-tuning, seperti learning rate yang kurang optimal atau terjadinya catastrophic forgetting pada lapisan awal jaringan. DenseNet121 hanya meningkat sedikit dari 0,782712 menjadi 0,809248, sedangkan EfficientNet-B0 dan VGG-19 masing-masing memperoleh akurasi 0,879411 dan 0,875571.

Rentang akurasi pada skenario ini adalah 0,175939 atau sekitar 17,59 poin persentase antara model terbaik (EfficientNet-B3: 98,46%) dan model terendah (VGG-16: 80,87%). Hal ini menunjukkan bahwa pendekatan transfer learning dengan fine-tuning memberikan keuntungan yang tidak merata di antara arsitektur yang berbeda. Arsitektur dengan desain modern dan efisien seperti EfficientNet dan ResNet dapat memanfaatkan bobot pretrained secara lebih optimal dibandingkan arsitektur yang lebih lama seperti VGG dan DenseNet. Dengan demikian, temuan ini menegaskan bahwa kombinasi pemilihan arsitektur dan strategi pelatihan merupakan faktor penting dalam memaksimalkan performa model klasifikasi berbasis deep learning.

Pengaruh Transfer Learning dan Fine-Tuning

Transfer learning memberikan keuntungan karena model tidak memulai proses pembelajaran dari bobot acak, melainkan dari representasi visual umum yang telah dipelajari pada data set berskala besar seperti ImageNet. Representasi awal seperti deteksi tepi, tekstur, pola warna, dan bentuk dasar masih relevan untuk citra Gram-stain, meskipun domain citra medis berbeda dari citra natural. Oleh karena itu, model dapat lebih cepat beradaptasi terhadap tugas klasifikasi Gram-positif dan Gram-negatif.

Fine-tuning berperan untuk menyesuaikan bobot pretrained terhadap karakteristik data set target. Pada citra mikroskopis, perbedaan kelas tidak hanya ditentukan oleh bentuk objek, tetapi juga oleh intensitas warna hasil pewarnaan, kualitas fokus, dan variasi preparat. Dengan membuka sebagian atau seluruh layer untuk dilatih kembali, model dapat mempelajari fitur yang lebih spesifik terhadap domain Gram-stain. Namun, hasil eksperimen menunjukkan bahwa manfaat fine-tuning bergantung pada arsitektur dan konfigurasi pelatihan. Arsitektur seperti EfficientNet-B3 dan ResNet mampu memanfaatkan fine-tuning dengan baik, sedangkan beberapa arsitektur lain tidak memperoleh peningkatan yang sama besar.

## Template Confusion Matrix

Confusion matrix digunakan untuk melihat distribusi prediksi benar dan salah pada masing-masing kelas. Pada penelitian ini, kelas yang digunakan adalah Gram-negatif dan Gram-positif. Template berikut dapat digunakan sebagai landasan untuk menambahkan hasil confusion matrix dari setiap model setelah proses evaluasi selesai dilakukan.

*Tabel 4.X Template Confusion Matrix Model [Nama Model]*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | TN | FP |
| Gram-positif | FN | TP |

Keterangan:

- TN (True Negative): jumlah citra Gram-negatif yang diprediksi benar sebagai Gram-negatif.
- FP (False Positive): jumlah citra Gram-negatif yang salah diprediksi sebagai Gram-positif.
- FN (False Negative): jumlah citra Gram-positif yang salah diprediksi sebagai Gram-negatif.
- TP (True Positive): jumlah citra Gram-positif yang diprediksi benar sebagai Gram-positif.

Template tersebut dapat digunakan untuk masing-masing arsitektur, misalnya Simple CNN, ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121. Setelah nilai TN, FP, FN, dan TP tersedia, analisis dapat diarahkan pada pola kesalahan model, bukan hanya pada nilai akurasi.

### Template Analisis Confusion Matrix Model [Nama Model]

Model [Nama Model] menghasilkan nilai TN sebesar [...], FP sebesar [...], FN sebesar [...], dan TP sebesar [...]. Berdasarkan distribusi tersebut, model menunjukkan kemampuan yang [baik/cukup/kurang] dalam membedakan kelas Gram-negatif dan Gram-positif.

Kesalahan prediksi paling dominan terjadi pada bagian [...], yaitu ketika model memprediksi [...] sebagai [...]. Kondisi ini menunjukkan bahwa model masih mengalami kesulitan dalam mengenali karakteristik visual [...], terutama pada citra dengan kondisi [...].

Jika nilai FP tinggi, model cenderung salah mengklasifikasikan citra Gram-negatif sebagai Gram-positif. Jika nilai FN tinggi, model cenderung salah mengklasifikasikan citra Gram-positif sebagai Gram-negatif. Dalam konteks pemeriksaan bakteri, pola kesalahan ini penting karena dapat menunjukkan bias model terhadap salah satu kelas dan membantu menentukan apakah model lebih baik dalam mengenali kelas Gram-negatif atau Gram-positif.

Secara keseluruhan, confusion matrix model [Nama Model] menunjukkan kecenderungan [...]. Hasil ini dapat digunakan untuk mengevaluasi sensitivitas model terhadap kelas Gram-positif dan spesifisitas model terhadap kelas Gram-negatif.

## Metrik Evaluasi Berdasarkan Confusion Matrix

Berdasarkan nilai TP, TN, FP, dan FN, beberapa metrik evaluasi yang digunakan dalam analisis model adalah sebagai berikut.

Accuracy menunjukkan proporsi prediksi benar terhadap seluruh data uji.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision menunjukkan seberapa banyak prediksi positif yang benar-benar sesuai dengan kelas positif.

Precision = TP / (TP + FP)

Recall atau sensitivity menunjukkan kemampuan model dalam mengenali seluruh data kelas positif.

Recall = TP / (TP + FN)

Specificity menunjukkan kemampuan model dalam mengenali kelas negatif dengan benar.

Specificity = TN / (TN + FP)

F1-score merupakan rata-rata harmonis antara precision dan recall.

F1-score = 2 x (Precision x Recall) / (Precision + Recall)

Pada penelitian ini, metrik precision, recall, specificity, dan F1-score penting untuk melengkapi akurasi karena data bakteri Gram dapat memiliki distribusi kelas yang tidak seimbang. Model dengan akurasi tinggi belum tentu memiliki kemampuan yang seimbang dalam mengenali kedua kelas. Oleh karena itu, confusion matrix dan metrik turunannya perlu digunakan untuk melihat apakah model cenderung bias terhadap Gram-negatif atau Gram-positif.

## Loss Function

Loss function digunakan untuk mengukur besar kesalahan prediksi model selama proses pelatihan. Pada klasifikasi dua kelas Gram-negatif dan Gram-positif, loss function yang digunakan adalah CrossEntropyLoss. Fungsi ini membandingkan output logit dari model dengan label aktual, kemudian menghasilkan nilai loss yang digunakan untuk memperbarui bobot melalui proses backpropagation.

CrossEntropyLoss sesuai digunakan karena output model berupa dua kelas. Apabila data set memiliki ketidakseimbangan kelas, weighted CrossEntropyLoss dapat digunakan agar kelas minoritas memperoleh bobot lebih besar. Dengan demikian, kesalahan pada kelas minoritas akan memberikan penalti lebih tinggi selama proses pelatihan dan membantu mengurangi bias model terhadap kelas mayoritas.

Analisis loss dilakukan dengan membandingkan train loss dan validation loss selama proses pelatihan:

- Jika train loss menurun tetapi validation loss meningkat, model mengalami overfitting.
- Jika train loss dan validation loss sama-sama tinggi, model mengalami underfitting.
- Jika train loss dan validation loss sama-sama menurun secara stabil, proses pelatihan berjalan baik.
- Jika validation loss stagnan sementara train loss terus turun, model mulai terlalu menyesuaikan diri terhadap data latih.

Template analisis loss yang dapat digunakan setelah kurva loss tersedia adalah sebagai berikut:

Model [Nama Model] menunjukkan train loss sebesar [...] dan validation loss sebesar [...] pada akhir pelatihan. Pola perubahan loss menunjukkan bahwa model mengalami [konvergensi stabil/overfitting/underfitting]. Jika dibandingkan dengan akurasi dan confusion matrix, nilai loss ini menunjukkan bahwa model [mampu/belum mampu] mempelajari fitur pembeda antara Gram-negatif dan Gram-positif secara konsisten.

Visualisasi

Arsitektur antarmuka pada website ini dirancang untuk core fiturnya, yaitu klasifikasi specimen yang diupload oleh analis. Berikut ialah contoh tampilannya :

*Gambar 4.7 Tampilan saat user akan melakukan klasifikasi*

Pada gambar 4.x ini menunjukan fitur deteksi dan klasifikasi, di sini user bisa memilih pasien yang sampelnya akan diklasifikasi. Kemudian user akan melakukan upload sampel, lalu user akan memilih apakah menggunakan auto crop atau manual dengan Region of Interest (RoI). Kemudian user bisa memilih model apa yang akan digunakan untuk deteksi dan klasifikasi. Kemudian user akan mendapatkan hasil seperti gambar 4.x.

*Gambar 4.8 Tampilan hasil klasifikasi*

Pada gambar 4.x menunjukan hasil dari klasifikasi untuk setiap bakteri yang terdeteksi. Pada gambar ini juga menampilkan jumlah bakteri gram positif dan negatif serta gambar crop dan klasifikasinya.

