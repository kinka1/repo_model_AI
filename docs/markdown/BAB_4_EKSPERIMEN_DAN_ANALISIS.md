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

Kualitas: Gambar dengan rentang ukuran file 4,82 KB - 13,04 KB

Dataset yang telah dikumpulkan mencakup dua kelas yang relevan untuk mendukung sistem klasifikasi bakteri Gram. Kedua kelas tersebut meliputi Gram Positif dan Gram Negatif. Dataset dirancang sedemikian rupa agar dapat dilatih menjadi sebuah model yang mampu mengklasifikasikan bakteri berdasarkan hasil pewarnaan Gram secara akurat. Dalam penelitian ini, data set yang digunakan terdiri dari 11.824 gambar yang telah dilabeli oleh ahli mikrobiologi. Seluruh gambar tersebut memuat objek bakteri dari dua kelas yang relevan untuk mendukung sistem klasifikasi otomatis.

Class

Jumlah

Gram Positif

9.170

Gram Negatif

2.654

*Tabel 4. 2 Tabel Klasifikasi pada Dataset*

Berdasarkan Tabel 4.1, dapat diidentifikasi bahwa data set memiliki ketidak seimbangan kelas (class imbalance) dengan rasio 3,45:1 antara kelas Gram Positif dan Gram Negatif. Kondisi ini merupakanrefleksi dari distribusi natural sampel bakteri di laboratorium mikrobiologi, dimana bakteri Gram Positif umumnya lebih banyak ditemukan dalam praktik klinis.

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

Pada data set imbalance, data set condong ke bakteri gram positif dengan perbandingan 3,45:1 antara bakteri gram positif dan negatif. Untuk mengatasi hal tersebut, bisa menggunakan dua strategi utama yaitu :

Penggunaan Class Weights

Implementasi weighted loss function dengan bobot yang dihitung berdasarkan inverse frequency:

Dimana N adalah total sampel, nc adalah jumlah kelas, dan Ni adalah jumlah sampel kelas i.

## Hasil perhitungan:

Bobot untuk kelas Gram Positif (mayoritas): 0,65

Bobot untuk kelas Gram Negatif (minoritas): 2,23

Pemberian bobot yang lebih tinggi pada kelas minoritas bertujuan agar model tidak bias dalam memprediksi kelas mayoritas dan memberikan penalti lebih besar untuk kesalahan pada kelas Gram Positif.

Sharpness-based Filtering

Pada skenario baseline, dilakukan pengurangan sampel kelas Gram Positif menggunakan metode sharpness-based filtering untuk menyeimbangkan jumlah sampel dengan kelas Gram Negatif. Metode ini memilih sampel Gram Positif dengan kualitas terbaik berdasarkan nilai Laplacian variance sebagai indikator ketajaman gambar, sehingga hanya gambar dengan kualitas visual optimal yang dipertahankan dalam data set training yang seimbang. Contoh code-nya :

sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

Yang mana library memiliki sebuah rumus :

Dimana :

xi adalah nilai pixel Laplacian ke-i

adalah mean dari semua nilai Laplacian

adalah total pixel

TEMPAT UJICOBA

Ujicoba penelitian ini dilakukan secara mandiri menggunakan laptop pribadi penulis, dengan spesifikasi perangkat keras dan perangkat lunak sebagaimana dijelaskan pada Tabel 4.4.

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

Pembagian data set adalah tahap penting untuk memastikan bahwa model yang dikembangkan dapat belajar dengan baik dari data latih dan data evaluasi. Dataset pada penelitian ini berjumlah total 11.824 gambar, yang dibagi menjadi tiga subset: train, validation, dan test. Pembagian data dilakukan dengan skala 70% : 15% : 15% untuk memastikan distribusi data yang optimal sesuai dengan kebutuhan penelitian.

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

![Gambar 4.2 Diagram Arsitektur Simple CNN](../images/simple_cnn_architecture.png)

*Gambar 4. 2 Diagram arsitektur Simple CNN*

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

Spesifikasi Arsitektur dan Konfigurasi Training:

Arsitektur yang digunakan pada Skenario 2 identik dengan Skenario 1 (Simple CNN dengan 5 convolutional layer, channel progression 3→32→64→128→256→512, Max Pooling 2×2, dan 3 fully connected layer 25.088→1.024→512→2). Perbedaan satu-satunya terletak pada proses loading data, di mana teknik augmentasi di atas diterapkan secara on-the-fly pada setiap batch training sehingga model menerima variasi citra yang berbeda pada setiap epoch tanpa menambah jumlah file fisik pada data set.

Konfigurasi training yang digunakan: Optimizer Adam dengan learning rate 0,001, loss function CrossEntropyLoss dengan class weights berdasarkan inverse frequency (lihat subbab Karakteristik Data), dan epoch maksimum 40 sesuai Tabel 4.6. Proses pelatihan dipantau melalui train loss dan validation loss pada setiap epoch untuk memastikan augmentasi tidak menyebabkan model menjadi underfitting akibat variasi data yang terlalu besar.

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

ResNet101

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

## Skenario 1

Hasil pengujian pada setiap data disajikan pada Gambar 4.2 dan Tabel 4.8 berikut

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

Hasil pengujian pada setiap data disajikan pada Gambar 4.3 dan Tabel 4.9 berikut

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

Spesifikasi Konfigurasi Training dan Strategi Fine-Tuning Skenario 4:

Pelatihan pada Skenario 4 dilakukan dalam dua fase berurutan untuk setiap arsitektur (ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3):

- Fase 1 (Head Training): seluruh backbone pretrained dibekukan (frozen) dan hanya classifier head yang dilatih, menggunakan optimizer AdamW dengan learning rate 0,001 dan weight decay 0,0001. Fase ini bertujuan menstabilkan classifier baru sebelum bobot backbone ikut diperbarui.
- Fase 2 (Fine-tuning): sebagian layer terakhir backbone dibuka (unfrozen) bersama classifier head, kemudian dilatih kembali dengan learning rate yang lebih kecil yaitu 0,0001 agar representasi fitur pretrained tidak rusak (catastrophic forgetting) namun tetap dapat disesuaikan terhadap karakteristik citra Gram-stain.

Layer yang dibuka pada fase fine-tuning berbeda untuk setiap arsitektur, menyesuaikan struktur blok masing-masing:

- ResNet50 dan ResNet101: residual block terakhir (layer4) beserta fully connected classifier.
- EfficientNet-B0 dan EfficientNet-B3: beberapa MBConv block terakhir, conv_head, batch normalization terkait, beserta classifier.

Kedua fase menggunakan loss function weighted CrossEntropyLoss dengan bobot kelas berdasarkan inverse frequency (lihat subbab Karakteristik Data), serta learning rate scheduler ReduceLROnPlateau (factor 0,5, patience 3 epoch) yang menurunkan learning rate apabila macro F1-score pada validation set tidak meningkat. Early stopping dengan patience 10 epoch diterapkan untuk menghentikan pelatihan apabila tidak ada peningkatan performa, dengan batas epoch maksimum 80 sesuai Tabel 4.6.

Strategi dua fase ini bertujuan untuk memanfaatkan representasi fitur umum dari ImageNet pada fase awal, kemudian secara bertahap menyesuaikan fitur tersebut terhadap domain citra Gram-stain pada fase fine-tuning, tanpa mengorbankan stabilitas pelatihan akibat perubahan bobot yang terlalu drastis pada awal proses.

## Skenario 5: Transfer Learning dan Fine-Tuning Arsitektur CNN

Pada skenario ini, beberapa arsitektur CNN dilatih menggunakan pendekatan transfer learning dan fine-tuning. Berbeda dengan pelatihan from scratch, transfer learning memanfaatkan bobot pretrained ImageNet sebagai titik awal pembelajaran, kemudian bobot model disesuaikan kembali terhadap data set citra Gram-stain. Arsitektur yang diuji meliputi ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, VGG-16, VGG-19, dan DenseNet121.

Tujuan utama skenario ini adalah melihat kemampuan masing-masing arsitektur dalam memanfaatkan representasi visual pretrained untuk membedakan citra Gram-negatif dan Gram-positif. Analisis tidak hanya dilakukan berdasarkan akurasi, tetapi juga berdasarkan precision, recall, F1-score, confusion matrix, dan pola loss selama pelatihan.

### Konfigurasi Umum Pelatihan Skenario 5

Sebagian besar konfigurasi pelatihan pada Skenario 5 dibuat sama agar perbedaan hasil lebih banyak dipengaruhi oleh karakteristik arsitektur model. Konfigurasi umum yang digunakan adalah sebagai berikut.

| Komponen | Konfigurasi Umum |
|---|---|
| Framework | PyTorch |
| Input citra | RGB 224 x 224 piksel |
| Pretrained weights | ImageNet |
| Jumlah kelas output | 2 kelas: Gram-negatif dan Gram-positif |
| Loss function | Weighted CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Strategi checkpoint | Menyimpan model terbaik berdasarkan performa validasi |
| Metrik evaluasi | Accuracy, precision, recall, F1-score, ROC-AUC, dan confusion matrix |
| Augmentasi data | Transformasi citra latih seperti rotasi, flipping, color jitter, dan resize/crop sesuai script pelatihan masing-masing |

Secara umum, proses pelatihan dilakukan dalam dua tahap. Tahap pertama melatih classifier head dengan backbone pretrained sebagai feature extractor. Tahap kedua melakukan fine-tuning dengan membuka sebagian layer akhir atau seluruh model, bergantung pada arsitektur dan kebutuhan komputasi. Dengan strategi ini, model dapat mempertahankan fitur visual umum dari ImageNet sekaligus menyesuaikan fitur akhir terhadap karakteristik citra Gram-stain.

### Perbedaan Hyperparameter per Arsitektur

Tidak semua hyperparameter pada Skenario 5 sama. Perbedaan terutama terdapat pada optimizer, batch size, jumlah epoch maksimum, learning rate, dan bagian backbone yang dibuka saat fine-tuning. Perbedaan tersebut diperlukan karena setiap arsitektur memiliki ukuran parameter, kebutuhan memori, dan karakteristik optimasi yang berbeda.

| Arsitektur | Optimizer | Batch Size | Epoch Maksimum | Learning Rate | Strategi Fine-tuning |
|---|---|---:|---:|---|---|
| ResNet50 | AdamW | 32 | 40 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka `layer4` dan classifier |
| ResNet101 | AdamW | 32 | 60 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka blok residual terakhir dan classifier |
| EfficientNet-B0 | AdamW | 32 | 60 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka blok akhir EfficientNet dan classifier |
| EfficientNet-B3 | AdamW | 16 | 60 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka beberapa blok akhir EfficientNet dan classifier |
| VGG-16 | SGD momentum 0,9 | 8 | 80 | 0,00005 | Fine-tuning full model dengan classifier custom |
| VGG-19 | SGD momentum 0,9 | 8 | 80 | 0,00005 | Fine-tuning full model dengan classifier custom |
| DenseNet121 | AdamW | 16 | 48 | 0,001 lalu 0,0001 | Melatih classifier head, lalu membuka dense block terakhir dan classifier |

Perbedaan hyperparameter tersebut menunjukkan bahwa Skenario 5 tidak sepenuhnya menggunakan konfigurasi identik untuk setiap arsitektur. Model yang lebih besar seperti VGG-16 dan VGG-19 menggunakan batch size lebih kecil karena kebutuhan memori lebih tinggi. EfficientNet-B3 juga menggunakan batch size lebih kecil dibanding EfficientNet-B0 karena memiliki kapasitas model yang lebih besar. Sementara itu, ResNet dan DenseNet menggunakan pola dua tahap yang relatif serupa, yaitu head training diikuti fine-tuning pada blok akhir.

### Hasil Pelatihan ResNet50

Model ResNet50 pada Skenario 5 menggunakan hasil eksperimen `experiments/ta_resnet50_20260615_021715`. Model ini memperoleh akurasi sebesar 0,951696 dengan macro F1-score sebesar 0,951696 dan ROC-AUC sebesar 0,988272. Nilai tersebut menunjukkan bahwa ResNet50 mampu memanfaatkan residual connection dan bobot pretrained ImageNet secara efektif untuk mempelajari fitur pembeda antara Gram-negatif dan Gram-positif.

Confusion matrix ResNet50 menghasilkan TN sebesar 1.649, FP sebesar 90, FN sebesar 78, dan TP sebesar 1.661. Kesalahan klasifikasi relatif seimbang pada kedua kelas, dengan FP sedikit lebih tinggi dibanding FN. Hal ini menunjukkan bahwa model sedikit lebih sering mengklasifikasikan citra Gram-negatif sebagai Gram-positif, tetapi selisihnya kecil sehingga tidak menunjukkan bias yang kuat terhadap salah satu kelas.

Dari sisi pelatihan, ResNet50 mencapai epoch terbaik pada epoch ke-31 dengan train loss 0,1054 dan validation loss 0,1466. Train loss dan validation loss yang sama-sama rendah menunjukkan bahwa model telah konvergen dengan baik. Namun, validation loss yang mulai stagnan ketika train loss masih menurun mengindikasikan adanya mild overfitting pada akhir pelatihan, sehingga penggunaan checkpoint terbaik pada epoch ke-31 menjadi keputusan yang tepat.

### Hasil Pelatihan ResNet101

ResNet101 memiliki kedalaman lebih besar dibanding ResNet50, sehingga secara teoritis memiliki kapasitas representasi fitur yang lebih tinggi. Berdasarkan hasil eksperimen `experiments/resnet101_finetune_20260413_003414`, ResNet101 memperoleh akurasi sebesar 0,951409 dan macro F1-score sebesar 0,951404 pada epoch terbaik ke-60. Confusion matrix menunjukkan TN sebesar 1.672, FP sebesar 67, FN sebesar 102, dan TP sebesar 1.637.

Hasil tersebut menunjukkan bahwa ResNet101 memiliki performa yang sangat dekat dengan ResNet50. Nilai FP yang lebih rendah dibanding ResNet50 menunjukkan bahwa ResNet101 lebih baik dalam mempertahankan prediksi Gram-negatif, tetapi nilai FN yang lebih tinggi menunjukkan bahwa sebagian citra Gram-positif masih salah diklasifikasikan sebagai Gram-negatif. Dengan demikian, peningkatan kedalaman jaringan tidak otomatis menghasilkan peningkatan performa yang besar pada data set ini.

Dari sisi loss, ResNet101 memperoleh train loss 0,0496 dan validation loss 0,1867 pada epoch terbaik. Jarak antara train loss dan validation loss lebih besar dibanding ResNet50, sehingga model ini menunjukkan indikasi overfitting yang lebih kuat. Hal ini wajar karena ResNet101 memiliki jumlah parameter lebih besar, sehingga membutuhkan kontrol regularisasi dan data yang memadai agar kapasitas model tidak terlalu menyesuaikan diri terhadap data latih.

### Hasil Pelatihan EfficientNet-B0

EfficientNet-B0 merupakan arsitektur yang relatif ringan dan efisien secara parameter. Berdasarkan hasil eksperimen `experiments/efficientnet_b0_finetune_20260413_145240`, model ini memperoleh akurasi sebesar 0,910868 dan macro F1-score sebesar 0,910782 pada epoch terbaik ke-34. Confusion matrix menunjukkan TN sebesar 1.638, FP sebesar 101, FN sebesar 209, dan TP sebesar 1.530.

Hasil tersebut menunjukkan bahwa EfficientNet-B0 mampu mencapai performa yang baik dengan jumlah parameter yang lebih kecil dibanding ResNet dan VGG. Namun, nilai FN yang lebih tinggi dibanding FP menunjukkan bahwa model lebih sering gagal mengenali citra Gram-positif. Dengan kata lain, model cenderung lebih kuat dalam mengenali Gram-negatif daripada Gram-positif pada konfigurasi eksperimen ini.

Train loss pada epoch terbaik adalah 0,1713 dan validation loss sebesar 0,2389. Pola ini menunjukkan konvergensi yang cukup baik, tetapi performanya masih berada di bawah ResNet50 dan ResNet101. Hal ini dapat disebabkan oleh kapasitas EfficientNet-B0 yang lebih kecil, sehingga fitur yang dipelajari tidak sedalam model ResNet pada data set Gram-stain.

### Hasil Pelatihan EfficientNet-B3

EfficientNet-B3 memiliki kapasitas lebih besar dibanding EfficientNet-B0 melalui compound scaling pada depth, width, dan resolution. Berdasarkan hasil eksperimen `experiments/scenario_3d_efficientnet_b3`, model ini memperoleh akurasi sebesar 0,866967, precision sebesar 0,888376, recall sebesar 0,866967, F1-score sebesar 0,872866, dan ROC-AUC sebesar 0,944491. Confusion matrix menunjukkan TN sebesar 1.195, FP sebesar 181, FN sebesar 55, dan TP sebesar 343.

Hasil ini menunjukkan bahwa EfficientNet-B3 memiliki kemampuan pemisahan kelas yang cukup baik berdasarkan ROC-AUC, tetapi akurasinya belum melampaui EfficientNet-B0 maupun ResNet pada eksperimen yang tersedia. Nilai FP yang jauh lebih tinggi dibanding FN menunjukkan bahwa model lebih sering salah mengklasifikasikan Gram-negatif sebagai Gram-positif. Pola ini berbeda dari EfficientNet-B0, yang lebih banyak menghasilkan FN.

Performa EfficientNet-B3 yang belum optimal dapat dipengaruhi oleh konfigurasi pelatihan dan karakteristik data. Walaupun kapasitas model lebih besar, model yang lebih kompleks juga membutuhkan penyesuaian learning rate, jumlah epoch, dan strategi fine-tuning yang lebih tepat agar manfaat compound scaling dapat dimanfaatkan secara maksimal.

### Hasil Pelatihan VGG-16

VGG-16 pada Skenario 5 menggunakan hasil eksperimen `experiments/scenario_5a_vgg16`. Model ini memperoleh akurasi sebesar 0,953113, precision sebesar 0,954171, recall sebesar 0,953113, F1-score sebesar 0,953399, dan ROC-AUC sebesar 0,988942. Confusion matrix menunjukkan TN sebesar 2.509, FP sebesar 119, FN sebesar 61, dan TP sebesar 1.150.

Hasil tersebut menunjukkan bahwa VGG-16 mampu menghasilkan performa tinggi setelah dilakukan transfer learning dan fine-tuning penuh. Meskipun VGG-16 memiliki arsitektur yang lebih sederhana tanpa residual connection maupun dense connection, jumlah parameter yang besar dan classifier custom memungkinkan model mempelajari pola visual Gram-stain secara efektif pada eksperimen ini.

Kesalahan klasifikasi VGG-16 lebih banyak terjadi pada FP dibanding FN. Artinya, model lebih sering memprediksi citra Gram-negatif sebagai Gram-positif dibanding sebaliknya. Namun, secara umum nilai kesalahan relatif rendah dibanding jumlah data evaluasi, sehingga performa VGG-16 dapat dikategorikan baik. Kelemahannya adalah ukuran parameter yang sangat besar, sehingga kebutuhan memori dan waktu pelatihan lebih tinggi dibanding ResNet dan EfficientNet.

### Hasil Pelatihan VGG-19

VGG-19 menggunakan hasil eksperimen `experiments/scenario_5b_vgg19`. Model ini memperoleh akurasi sebesar 0,956239, precision sebesar 0,956847, recall sebesar 0,956239, F1-score sebesar 0,956426, dan ROC-AUC sebesar 0,990255. Confusion matrix menunjukkan TN sebesar 2.524, FP sebesar 104, FN sebesar 64, dan TP sebesar 1.147.

Jika dibandingkan dengan VGG-16, VGG-19 memperoleh akurasi dan F1-score sedikit lebih tinggi. Tambahan layer konvolusi pada VGG-19 tampaknya membantu model menangkap pola visual yang lebih kompleks pada citra Gram-stain. Nilai ROC-AUC yang mencapai 0,990255 juga menunjukkan kemampuan pemisahan kelas yang sangat baik.

Meskipun demikian, VGG-19 memiliki jumlah parameter lebih besar daripada VGG-16, sehingga biaya komputasi dan kebutuhan memori juga lebih tinggi. Peningkatan performa yang diperoleh relatif kecil dibanding tambahan kompleksitas model. Oleh karena itu, VGG-19 memberikan performa yang sangat baik, tetapi perlu dipertimbangkan dari sisi efisiensi jika model akan digunakan pada sistem dengan sumber daya terbatas.

### Hasil Pelatihan DenseNet121

DenseNet121 menggunakan hasil eksperimen `experiments/retrain_densenet121_20260414_171014`. Model ini memperoleh akurasi sebesar 0,868678 dan macro F1-score sebesar 0,868514 pada epoch terbaik ke-2. Confusion matrix menunjukkan TN sebesar 1.573, FP sebesar 166, FN sebesar 291, dan TP sebesar 1.450.

Hasil tersebut menunjukkan bahwa DenseNet121 mampu mengenali kedua kelas dengan performa cukup baik, tetapi masih berada di bawah ResNet50, ResNet101, VGG-16, dan VGG-19 pada hasil eksperimen yang tersedia. Nilai FN yang lebih tinggi dibanding FP menunjukkan bahwa DenseNet121 lebih sering salah mengklasifikasikan Gram-positif sebagai Gram-negatif. Dengan demikian, kelemahan utama model ini pada eksperimen tersebut adalah sensitivitas terhadap kelas Gram-positif.

Train loss pada epoch terbaik adalah 0,4002 dan validation loss sebesar 0,3044. Nilai validation loss yang lebih rendah daripada train loss dapat terjadi karena pengaruh augmentasi dan regularisasi pada data latih, sehingga data latih menjadi lebih sulit dibanding data validasi. Namun, karena epoch terbaik terjadi sangat awal, hasil ini juga menunjukkan bahwa pelatihan DenseNet121 masih perlu dieksplorasi lebih lanjut, misalnya dengan penyesuaian jumlah epoch, learning rate, dan strategi pembukaan dense block.

### Analisis Perbandingan Hasil Skenario 5

Ringkasan hasil setiap arsitektur CNN yang telah dicoba pada Skenario 5 disajikan pada Tabel 4.12. Beberapa hasil berasal dari folder eksperimen yang berbeda, sehingga ukuran data evaluasi dan konfigurasi detail tidak selalu identik. Oleh karena itu, tabel ini digunakan sebagai ringkasan hasil eksperimen yang tersedia, sedangkan interpretasi utama tetap memperhatikan analisis masing-masing arsitektur.

*Gambar 4.6 Grafik Data Akurasi Model CNN dengan Transfer Learning dan Fine-Tuning*

*Tabel 4.12 Ringkasan Hasil Model CNN dengan Transfer Learning dan Fine-Tuning*

| No. | Arsitektur Model | Akurasi | F1-score | ROC-AUC | Sumber Eksperimen |
|---:|---|---:|---:|---:|---|
| 1 | ResNet50 | 0,951696 | 0,951696 | 0,988272 | `ta_resnet50_20260615_021715` |
| 2 | ResNet101 | 0,951409 | 0,951404 | Tidak tersedia | `resnet101_finetune_20260413_003414` |
| 3 | EfficientNet-B0 | 0,910868 | 0,910782 | Tidak tersedia | `efficientnet_b0_finetune_20260413_145240` |
| 4 | EfficientNet-B3 | 0,866967 | 0,872866 | 0,944491 | `scenario_3d_efficientnet_b3` |
| 5 | VGG-16 | 0,953113 | 0,953399 | 0,988942 | `scenario_5a_vgg16` |
| 6 | VGG-19 | 0,956239 | 0,956426 | 0,990255 | `scenario_5b_vgg19` |
| 7 | DenseNet121 | 0,868678 | 0,868514 | Tidak tersedia | `retrain_densenet121_20260414_171014` |

Berdasarkan ringkasan tersebut, VGG-19 memperoleh akurasi tertinggi sebesar 0,956239, diikuti oleh VGG-16 sebesar 0,953113, ResNet50 sebesar 0,951696, dan ResNet101 sebesar 0,951409. Keempat model ini berada pada rentang performa yang sangat dekat, sehingga perbedaan akurasi tidak hanya perlu dilihat dari nilai akhir, tetapi juga dari stabilitas loss, jumlah parameter, dan pola kesalahan pada confusion matrix.

ResNet50 menjadi model yang seimbang antara performa dan efisiensi. Walaupun akurasinya sedikit di bawah VGG-16 dan VGG-19, jumlah parameternya jauh lebih kecil daripada VGG, dan pola confusion matrix menunjukkan kesalahan yang relatif seimbang antara kelas Gram-negatif dan Gram-positif. ResNet101 memiliki performa yang sangat dekat dengan ResNet50, tetapi indikasi overfitting lebih kuat karena train loss jauh lebih rendah daripada validation loss.

VGG-16 dan VGG-19 menunjukkan performa tinggi pada eksperimen ini. Namun, kedua model memiliki jumlah parameter sangat besar, sehingga membutuhkan memori dan waktu pelatihan lebih tinggi. VGG-19 memberikan hasil terbaik secara akurasi dan ROC-AUC, tetapi peningkatannya terhadap VGG-16 relatif kecil dibanding tambahan kompleksitas model.

EfficientNet-B0 dan EfficientNet-B3 menunjukkan performa lebih rendah dibanding ResNet dan VGG pada eksperimen yang tersedia. EfficientNet-B0 lebih stabil daripada EfficientNet-B3 pada hasil ini, meskipun EfficientNet-B3 memiliki kapasitas lebih besar. Hal ini menunjukkan bahwa kapasitas arsitektur yang lebih besar tidak otomatis menghasilkan performa lebih tinggi jika konfigurasi fine-tuning belum optimal.

DenseNet121 memperoleh performa cukup baik, tetapi masih memiliki kelemahan pada pengenalan kelas Gram-positif yang terlihat dari nilai FN yang tinggi. Meskipun DenseNet memiliki mekanisme dense connection dan jumlah parameter yang relatif efisien, hasil eksperimen menunjukkan bahwa strategi pelatihan dan penyesuaian hyperparameter masih sangat berpengaruh terhadap performanya pada citra Gram-stain.

Secara keseluruhan, hasil Skenario 5 menunjukkan bahwa transfer learning dan fine-tuning mampu meningkatkan kemampuan model dalam mempelajari fitur citra Gram-stain. Namun, keberhasilan setiap arsitektur tetap dipengaruhi oleh ukuran model, mekanisme koneksi internal, strategi fine-tuning, serta konfigurasi hyperparameter yang digunakan.

## Perbandingan Karakteristik Arsitektur CNN

Setiap arsitektur CNN yang digunakan pada penelitian ini memiliki mekanisme dan karakteristik desain yang berbeda, sehingga memengaruhi cara masing-masing model mempelajari fitur dari citra Gram-stain. Tabel berikut merangkum perbandingan karakteristik utama dari kedelapan arsitektur yang dievaluasi.

| Arsitektur | Kedalaman / Mekanisme Khas | Total Parameter (±) | Karakteristik untuk Citra Gram-stain |
|---|---|---:|---|
| Simple CNN | 5 convolutional layer + 3 fully connected layer, tanpa skip connection atau dense connection | < 50 juta | Baseline ringan, seluruh fitur dipelajari dari awal sehingga performa bergantung penuh pada jumlah dan kualitas data latih |
| ResNet50 | 50 layer dengan residual (skip) connection | 24,5 juta | Skip connection menjaga aliran gradien pada jaringan dalam, efektif mempelajari tekstur dan bentuk bakteri meski dilatih dari awal |
| ResNet101 | 101 layer dengan residual (skip) connection, lebih dalam dari ResNet50 | 43,5 juta | Kapasitas representasi lebih besar dari ResNet50, namun membutuhkan lebih banyak komputasi dan data agar tidak overfitting |
| EfficientNet-B0 | Compound scaling (depth, width, resolution) berbasis MBConv block | 4,7 juta | Arsitektur paling ringan, efisien secara parameter, namun pada pelatihan from scratch memerlukan representasi pretrained agar performa optimal |
| EfficientNet-B3 | Compound scaling dengan skala lebih besar dari EfficientNet-B0 | 11,5 juta | Kapasitas representasi tinggi yang baru terlihat optimal ketika dikombinasikan dengan transfer learning dan fine-tuning |
| VGG-16 | 16 layer dengan convolution 3x3 bertumpuk, tanpa skip/dense connection | ± 138 juta | Struktur sederhana namun parameter sangat besar, rentan overfitting pada data set berukuran sedang seperti pada penelitian ini |
| VGG-19 | 19 layer dengan convolution 3x3 bertumpuk, lebih dalam dari VGG-16 | ± 143 juta | Pola serupa VGG-16 dengan kapasitas sedikit lebih besar, tetap rentan terhadap overfitting dan kendala fine-tuning |
| DenseNet121 | 121 layer dengan dense connection (feature reuse antar layer) | ± 8 juta | Parameter efisien melalui feature reuse, tetapi performa tetap bergantung pada strategi pelatihan dan jumlah data yang memadai |

Berdasarkan perbandingan tersebut, terdapat tiga kelompok karakter arsitektur yang relevan terhadap hasil eksperimen pada Skenario 3-5:

1. Arsitektur dengan residual connection (ResNet50, ResNet101) cenderung stabil baik pada pelatihan from scratch maupun setelah fine-tuning, karena mekanisme skip connection membantu aliran gradien pada jaringan yang dalam.
2. Arsitektur dengan compound scaling (EfficientNet-B0, EfficientNet-B3) memiliki desain efisien parameter, tetapi hasil eksperimen menunjukkan bahwa kapasitas tersebut tetap membutuhkan konfigurasi fine-tuning yang tepat agar performanya optimal pada data set Gram-stain.
3. Arsitektur VGG-16 dan VGG-19 menunjukkan performa tinggi setelah fine-tuning penuh, tetapi membutuhkan jumlah parameter dan komputasi yang besar. Sebaliknya, DenseNet121 memiliki parameter lebih efisien melalui feature reuse, tetapi performanya masih bergantung kuat pada strategi pelatihan dan penyesuaian hyperparameter.

Perbandingan ini menegaskan bahwa pemilihan arsitektur CNN untuk klasifikasi bakteri Gram tidak hanya bergantung pada jumlah parameter atau kedalaman jaringan, tetapi juga pada kesesuaian mekanisme arsitektur (residual connection, compound scaling, atau dense connection) dengan strategi pelatihan yang digunakan (from scratch, transfer learning, atau fine-tuning).

## Confusion Matrix

Confusion matrix digunakan untuk melihat distribusi prediksi benar dan salah pada masing-masing kelas. Pada penelitian ini, kelas yang digunakan adalah Gram-negatif dan Gram-positif. Format dasar confusion matrix ditunjukkan terlebih dahulu, kemudian diikuti oleh hasil confusion matrix dari model yang sudah memiliki artefak evaluasi.

*Tabel 4.X Format Confusion Matrix*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | TN | FP |
| Gram-positif | FN | TP |

Keterangan:

- TN (True Negative): jumlah citra Gram-negatif yang diprediksi benar sebagai Gram-negatif.
- FP (False Positive): jumlah citra Gram-negatif yang salah diprediksi sebagai Gram-positif.
- FN (False Negative): jumlah citra Gram-positif yang salah diprediksi sebagai Gram-negatif.
- TP (True Positive): jumlah citra Gram-positif yang diprediksi benar sebagai Gram-positif.

Format di atas digunakan untuk mencatat hasil confusion matrix dari setiap arsitektur yang dievaluasi. Pada bagian ini, nilai confusion matrix diisi untuk arsitektur yang sudah memiliki artefak evaluasi lengkap. Simple CNN tetap disediakan sebagai template karena nilai TP, TN, FP, dan FN belum tersedia pada format evaluasi yang sama.

*Tabel 4.X.1 Confusion Matrix Model Simple CNN*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | Belum tersedia | Belum tersedia |
| Gram-positif | Belum tersedia | Belum tersedia |

*Tabel 4.X.2 Confusion Matrix Model ResNet50*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 1.649 | 90 |
| Gram-positif | 78 | 1.661 |

![Confusion Matrix ResNet50](../images/resnet50_confusion_matrix.png)

Catatan: Hasil pada Tabel 4.X.2 di atas berasal dari evaluasi tambahan (pelengkap) model ResNet50 hasil fine-tuning pada data validasi sebanyak 3.478 citra (1.739 Gram-negatif dan 1.739 Gram-positif), terpisah dari pengujian 10 citra pada Tabel 4.10. Evaluasi ini bertujuan memberikan gambaran performa model pada data dalam jumlah lebih besar.

### Analisis Confusion Matrix Model ResNet50 (Hasil Evaluasi)

Model ResNet50 menghasilkan nilai TN sebesar 1.649, FP sebesar 90, FN sebesar 78, dan TP sebesar 1.661. Berdasarkan distribusi tersebut, model menunjukkan kemampuan yang baik dalam membedakan kelas Gram-negatif dan Gram-positif, dengan total kesalahan klasifikasi hanya 168 dari 3.478 citra (akurasi 95,17%).

Kesalahan prediksi paling dominan terjadi pada FP (90 citra), yaitu ketika model memprediksi citra Gram-negatif sebagai Gram-positif, sedikit lebih banyak dibandingkan FN (78 citra) di mana citra Gram-positif diprediksi sebagai Gram-negatif. Kondisi ini menunjukkan bahwa model sedikit lebih sering salah pada citra Gram-negatif dengan karakteristik visual yang menyerupai Gram-positif.

Nilai FP (90) sedikit lebih tinggi dibandingkan FN (78), sehingga model memiliki kecenderungan kecil untuk salah mengklasifikasikan citra Gram-negatif sebagai Gram-positif. Namun demikian, recall kelas Gram-positif (sensitivity = TP/(TP+FN) = 1.661/1.739 = 0,9551) dan recall kelas Gram-negatif (specificity = TN/(TN+FP) = 1.649/1.739 = 0,9482) berada pada rentang yang berdekatan, sehingga model tidak menunjukkan bias signifikan terhadap salah satu kelas.

Secara keseluruhan, confusion matrix model ResNet50 menunjukkan kecenderungan kinerja yang seimbang antara kedua kelas dengan tingkat kesalahan yang relatif kecil (4,83%). Nilai ROC-AUC sebesar 0,9883 turut mengonfirmasi bahwa model memiliki kemampuan pemisahan kelas yang sangat baik.

*Tabel 4.X.3 Confusion Matrix Model ResNet101*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 1.672 | 67 |
| Gram-positif | 102 | 1.637 |

*Tabel 4.X.4 Confusion Matrix Model EfficientNet-B0*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 1.638 | 101 |
| Gram-positif | 209 | 1.530 |

*Tabel 4.X.5 Confusion Matrix Model EfficientNet-B3*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 1.195 | 181 |
| Gram-positif | 55 | 343 |

*Tabel 4.X.6 Confusion Matrix Model VGG-16*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 2.509 | 119 |
| Gram-positif | 61 | 1.150 |

*Tabel 4.X.7 Confusion Matrix Model VGG-19*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 2.524 | 104 |
| Gram-positif | 64 | 1.147 |

*Tabel 4.X.8 Confusion Matrix Model DenseNet121*

| Aktual \ Prediksi | Gram-negatif | Gram-positif |
|---|---:|---:|
| Gram-negatif | 1.573 | 166 |
| Gram-positif | 291 | 1.450 |

Analisis confusion matrix untuk setiap arsitektur telah dijabarkan pada subbagian hasil pelatihan masing-masing model. Secara umum, nilai FP menunjukkan kesalahan ketika citra Gram-negatif diprediksi sebagai Gram-positif, sedangkan nilai FN menunjukkan kesalahan ketika citra Gram-positif diprediksi sebagai Gram-negatif. Pola FP dan FN ini digunakan untuk melihat kecenderungan bias model terhadap salah satu kelas.

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

*Tabel 4.Y Ringkasan Loss per Arsitektur*

| Arsitektur | Best Epoch | Train Loss | Validation Loss | Status (Konvergen/Overfitting/Underfitting) |
|---|---:|---:|---:|---|
| Simple CNN | Belum tersedia | Belum tersedia | Belum tersedia | Belum tersedia |
| ResNet50 | 31 | 0,1054 | 0,1466 | Konvergen (indikasi mild overfitting setelah epoch ke-31) |
| ResNet101 | 60 | 0,0496 | 0,1867 | Konvergen, tetapi terdapat indikasi overfitting karena selisih train loss dan validation loss cukup besar |
| EfficientNet-B0 | 34 | 0,1713 | 0,2389 | Konvergen cukup stabil |
| EfficientNet-B3 | 15 | 0,3304 | 0,2678 | Konvergen, tetapi performa validasi belum optimal dibanding model lain |
| VGG-16 | 20 | 0,1664 | 0,1286 | Konvergen dengan performa validasi tinggi |
| VGG-19 | 29 | 0,1513 | 0,1241 | Konvergen dengan performa validasi paling tinggi pada ringkasan eksperimen |
| DenseNet121 | 2 | 0,4002 | 0,3044 | Konvergen awal, tetapi masih perlu eksplorasi epoch dan fine-tuning lebih lanjut |

Ringkasan loss pada Tabel 4.Y menunjukkan bahwa model dengan akurasi tinggi umumnya memiliki validation loss yang rendah. ResNet101 memiliki train loss paling rendah, tetapi selisihnya terhadap validation loss lebih besar sehingga indikasi overfitting lebih kuat. VGG-16 dan VGG-19 menunjukkan validation loss rendah dan performa validasi tinggi, sedangkan DenseNet121 masih memerlukan eksplorasi pelatihan lanjutan karena checkpoint terbaik muncul sangat awal.

### Analisis Loss Model ResNet50 (Hasil Evaluasi)

![Training Curves ResNet50](../images/resnet50_training_curves.png)

Model ResNet50 menunjukkan train loss sebesar 0,1054 dan validation loss sebesar 0,1466 pada epoch terbaik (epoch ke-31, fase fine-tuning). Pada grafik loss, train loss dan validation loss menurun bersama secara stabil hingga sekitar epoch ke-15 dan kemudian konvergen di kisaran 0,14-0,17. Setelah epoch ke-31, train loss terus menurun hingga sekitar 0,07-0,08 pada epoch ke-37-39, sedangkan validation loss relatif stagnan atau sedikit naik. Pola ini menunjukkan model mengalami konvergensi yang baik dengan indikasi mild overfitting pada epoch-epoch akhir, sehingga pemilihan checkpoint terbaik pada epoch ke-31 (berdasarkan macro F1-score validasi tertinggi) tepat digunakan untuk menghindari overfitting lebih lanjut.

Jika dibandingkan dengan confusion matrix pada Tabel 4.X.2 (akurasi 95,17%, macro F1-score 0,9517, ROC-AUC 0,9883), nilai loss ini menunjukkan bahwa model mampu mempelajari fitur pembeda antara Gram-negatif dan Gram-positif secara konsisten, dengan tingkat kesalahan yang kecil dan seimbang pada kedua kelas.

Visualisasi

Arsitektur antarmuka pada website ini dirancang untuk core fiturnya, yaitu klasifikasi specimen yang diupload oleh analis. Berikut ialah contoh tampilannya :

*Gambar 4.7 Tampilan saat user akan melakukan klasifikasi*

Pada gambar 4.x ini menunjukan fitur deteksi dan klasifikasi, di sini user bisa memilih pasien yang sampelnya akan diklasifikasi. Kemudian user akan melakukan upload sampel, lalu user akan memilih apakah menggunakan auto crop atau manual dengan Region of Interest (RoI). Kemudian user bisa memilih model apa yang akan digunakan untuk deteksi dan klasifikasi. Kemudian user akan mendapatkan hasil seperti gambar 4.x.

*Gambar 4.8 Tampilan hasil klasifikasi*

Pada gambar 4.x menunjukan hasil dari klasifikasi untuk setiap bakteri yang terdeteksi. Pada gambar ini juga menampilkan jumlah bakteri gram positif dan negatif serta gambar crop dan klasifikasinya.

